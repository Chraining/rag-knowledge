# 文档处理的核心逻辑：读文件 → 提取文本 → 切分 → 生成向量 → 写入 chunks → 更新状态。
# 这里的函数都接收一个数据库会话（db）作为参数，不关心自己是被谁调用的：
#   - 后台任务（tasks.py）调用它们时，传入连接开发数据库的会话
#   - 测试调用它们时，传入连接测试数据库的会话
#
# 状态机（只允许这几种变化）：
#   pending ──> processing ──> ready
#                   │
#                   └────────> failed ──(重新投递)──> processing
#
# 幂等：同一个文档的任务可能被执行多次（重复投递、worker 崩溃后重新投递、失败重试），
# 下面的设计保证无论执行几次，最终结果都和执行一次一样：
#   1. 已经是 ready 的文档直接跳过
#   2. 写 chunks 时先锁住文档这一行（SELECT ... FOR UPDATE），两个任务同时写时只能一个一个来
#   3. 写入前先删掉这个文档的旧 chunks，所以不会出现重复的段落
#   4. (document_id, chunk_index) 有唯一约束，作为数据库层面的最后一道保险

# hashlib：重新计算文件哈希，检查文件有没有损坏
import hashlib
# datetime、UTC：记录处理完成的时间（带时区）
from datetime import UTC, datetime
# batched：把一个序列按固定大小分组，例如 batched([1,2,3,4,5], 2) → (1,2) (3,4) (5,)
from itertools import batched
# PurePath：从路径字符串里取扩展名
from pathlib import PurePath

# Vector：向量类型（见 llm.embed_texts）
from pgvector import Vector
# delete、insert、select、update：构造对应的 SQL 语句
from sqlalchemy import delete, insert, select, update
# Session：数据库会话的类型
from sqlalchemy.orm import Session

# 调用 embedding 服务。写成 from app import llm 再用 llm.embed_texts，测试才能把它替换成假的
from app import llm
# 提取文字、永久性错误、表格类格式。PermanentError 和 extract_text 也通过这个模块对外提供（tasks.py 和测试在用）
from app.extract import TABLE_EXTENSIONS, PermanentError, extract_text
# 模型
from app.models import Chunk, Document, DocumentStatus
# 读取磁盘上的文件
from app.storage import read_file

# 每段最多多少个字符
CHUNK_SIZE = 500
# 写数据库时每批插入多少段。见 save_chunks
SAVE_BATCH_SIZE = 200
# 相邻两段重叠多少个字符。
# 为什么要重叠：如果一句话正好被切在两段的边界上，两段各拿到半句，检索时哪段都匹配不好；
# 重叠一部分，边界附近的内容会同时出现在两段里
CHUNK_OVERLAP = 50


# 把长文本切成多段，相邻两段有重叠
# 例如 size=5, overlap=2，文本 "abcdefghij" 切成 "abcde"、"defgh"、"ghij"
def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    # 每次窗口往后移动的距离
    step = size - overlap
    # 存放切出来的段落
    chunks: list[str] = []
    # range(0, 总长度, 步长)：start 依次是 0, step, 2*step, ...
    for start in range(0, len(text), step):
        # 取出 [start, start+size) 这一段，去掉首尾的空白
        piece = text[start : start + size].strip()
        # 只有空白的段落没有意义，跳过
        if piece:
            chunks.append(piece)
        # 这一段已经到了文本末尾，后面不用再切了（否则最后会多出一段完全被包含的小尾巴）
        if start + size >= len(text):
            break
    return chunks


# 表格专用的切分：按整行装进每一段，一行不会被从中间切断。
# 每一行都带着列名（见 extract.rows_to_lines），所以不需要重叠，任何一段单独拿出来都看得懂。
# 例如 size=10，行 "aaaa"、"bbbb"、"cccc" → "aaaa\nbbbb"、"cccc"
def split_lines(text: str, size: int = CHUNK_SIZE) -> list[str]:
    chunks: list[str] = []
    # 正在装的这一段
    current = ""
    for line in text.splitlines():
        # 单独一行就超过了长度上限（比如某个单元格里有一大段文字）：先收起当前段，这一行按普通方式切
        if len(line) > size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_text(line, size))
        # 加上这一行会超长：当前段装满了，收起来，这一行作为新一段的开头
        elif current and len(current) + 1 + len(line) > size:
            chunks.append(current)
            current = line
        # 还装得下：接在后面（+1 是换行符）
        else:
            current = f"{current}\n{line}" if current else line
    # 最后一段
    if current:
        chunks.append(current)
    return chunks


# 第一步（事务 1）：把文档状态改成 processing。返回文档对象；返回 None 表示"不需要处理"
def start_processing(db: Session, document_id: int) -> Document | None:
    # SELECT ... FOR UPDATE：查询的同时锁住这一行，直到 commit 才释放。
    # 另一个任务如果也想锁这一行，必须等这里提交，所以两个任务不会同时修改状态
    doc = db.scalar(select(Document).where(Document.id == document_id).with_for_update())
    # 文档不存在（任务排队期间用户把它删了），或者已经处理完了（重复投递）：什么都不做
    if doc is None or doc.status == DocumentStatus.READY:
        # 结束事务，释放锁
        db.rollback()
        return None
    # pending 或 failed（重试）或 processing（worker 崩溃后重新投递）都可以进入 processing
    doc.status = DocumentStatus.PROCESSING
    # 清掉上一次失败留下的错误信息
    doc.error_message = None
    # 立刻提交，让用户查询时能看到"处理中"
    db.commit()
    return doc


# 第二步（不碰数据库）：读文件、校验、提取文字、切分。这是最耗时的部分，所以放在事务外面，
# 避免长时间锁住数据库的行
def read_and_split(doc: Document) -> list[str]:
    try:
        content = read_file(doc.storage_path)
    except FileNotFoundError as e:
        # 记录还在但文件没了，重试也找不到
        raise PermanentError("File is missing from storage") from e
    # 这里用上 content_hash：重新计算哈希，和上传时记下的比较。
    # 不一样说明文件在磁盘上被改动或损坏了，不能把错误的内容存进知识库
    if hashlib.sha256(content).hexdigest() != doc.content_hash:
        raise PermanentError("File content does not match its hash")
    # 取扩展名（例如 ".pdf"），决定用哪种方式提取文字。存储路径里的扩展名是上传时校验过的
    extension = PurePath(doc.storage_path).suffix
    text = extract_text(content, extension)
    # 表格按整行切，其他按字数切
    chunks = split_lines(text) if extension in TABLE_EXTENSIONS else split_text(text)
    # 一段文字都没有：通常是扫描版 PDF（每页都是图片，需要 OCR 才能识别，目前不支持），
    # 或者是只有图片、没有文字的 Word / PPT
    if not chunks:
        raise PermanentError("No text could be extracted (scanned PDF or images only?)")
    return chunks


# 第三步（事务 2）：写入 chunks 和它们的向量，状态改成 ready。返回 True 表示这次写入了，False 表示跳过了
def save_chunks(db: Session, document_id: int, chunks: list[str], vectors: list[Vector]) -> bool:
    # 再次锁住文档这一行。第一步的锁在 commit 时已经释放了，这里必须重新锁：
    # 如果两个任务同时走到这里，第二个会等第一个提交，然后看到状态已经是 ready，直接跳过
    doc = db.scalar(select(Document).where(Document.id == document_id).with_for_update())
    # 处理期间文档被删了，或者另一个任务已经先完成了：什么都不写
    if doc is None or doc.status == DocumentStatus.READY:
        db.rollback()
        return False
    # 删掉这个文档的旧 chunks（如果有的话），保证不会重复
    db.execute(delete(Chunk).where(Chunk.document_id == document_id))
    # 每段文字和它的向量配成一行。zip 把两者配对；enumerate 再加上序号 i（从 0 开始）。
    # 这里用的是生成器（圆括号），不会一次性把所有行都造出来
    rows = (
        {"document_id": document_id, "chunk_index": i, "content": text, "embedding": vector}
        for i, (text, vector) in enumerate(zip(chunks, vectors, strict=True))
    )
    # 每次插入 200 行，而不是一次插入全部。
    # 原因：写入时每个向量都要先转成约 19KB 的文本，一本长小说有上万段，一次性全部转换要几百 MB 内存，
    # 实测 500 万字的文档会让 worker 超过 400MB 的内存上限被系统杀掉。分批之后内存峰值和文档长度基本无关。
    # 所有批次仍然在同一个事务里，最后一起 commit，"要么全写进去，要么都不生效"的保证不变
    for batch in batched(rows, SAVE_BATCH_SIZE):
        db.execute(insert(Chunk), list(batch))
    # 更新文档的状态和统计信息
    doc.status = DocumentStatus.READY
    doc.chunk_count = len(chunks)
    doc.processed_at = datetime.now(UTC)
    # 写 chunks 和改状态在同一个事务里：要么都成功，要么都不生效。
    # 不会出现"状态是 ready，但 chunks 只写了一半"的情况
    db.commit()
    return True


# 把文档标记为失败，并记下原因
def mark_failed(db: Session, document_id: int, message: str) -> None:
    # 用一条 UPDATE 语句直接改，不用先查出来。
    # 条件 status != 'ready'：如果另一个重复的任务已经处理成功了，不能把成功的结果改成失败
    db.execute(
        update(Document)
        .where(Document.id == document_id, Document.status != DocumentStatus.READY)
        # 错误信息截断到 1000 个字符，避免把超长的报错堆栈全部存进数据库
        .values(status=DocumentStatus.FAILED, error_message=message[:1000])
    )
    db.commit()


# 处理一个文档的完整流程。后台任务调用的就是这个函数
def process_document(db: Session, document_id: int) -> None:
    doc = start_processing(db, document_id)
    # 不需要处理，直接结束
    if doc is None:
        return
    chunks = read_and_split(doc)
    # 调用 embedding 服务把每段文字转成向量。这是整个流程里最慢的一步（网络调用），
    # 所以和读文件一样放在两个事务之间，不占着数据库的锁。
    # 失败时抛出 LLMUnavailable，属于"临时性错误"，tasks.py 会让 Celery 过一会儿重试
    vectors = llm.embed_texts(chunks)
    save_chunks(db, document_id, chunks, vectors)
