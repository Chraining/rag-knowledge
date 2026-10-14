# 测试后台文档处理：切分、解析、状态流转、幂等、失败与重试、和上传接口的衔接

# contextlib.nullcontext：一个"什么也不做"的 with 上下文，用来把测试的 db 会话包装成 SessionLocal() 的样子
import contextlib
# hashlib：计算测试文件的哈希
import hashlib
# io：在内存里生成 PDF
import io

# pytest：测试框架
import pytest
# PdfWriter：生成一个只有空白页的 PDF（模拟扫描件）
from pypdf import PdfWriter
# select、func：直接查数据库做验证
from sqlalchemy import func, select

# 被测试的模块
from app import processing, tasks
# 读取配置（临时修改存储目录）
from app.config import settings
# 模型
from app.models import Chunk, Document, DocumentStatus
# 把测试文件写到磁盘上
from app.storage import save_file


# ============================ 辅助函数 ============================


# 手工拼一个最小的、带文字的 PDF 文件。真实的 PDF 由很多"对象"组成，最后有一张"目录表"（xref）
# 记录每个对象在文件里的字节位置。这里只是为了测试，不需要看懂每一行的 PDF 语法
def make_pdf(text: str) -> bytes:
    # 页面内容：BT/ET 是"开始/结束文字"，Tf 选字体和字号，Td 移动到坐标，Tj 画出文字
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    # 五个对象：目录、页面列表、页面、页面内容、字体
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R"
        b" /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    # 文件头
    out = b"%PDF-1.4\n"
    # 记录每个对象的起始位置
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    # 目录表的位置
    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    # 文件尾。%%%%EOF 经过 % 格式化后变成 %%EOF
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref_at)
    return out


# 生成一个只有一页空白页的 PDF：能正常打开，但一个字都提取不出来，相当于扫描件
def make_blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


# 工厂 fixture：把内容存到临时目录，并在数据库里建一条 pending 状态的文档记录
@pytest.fixture
def make_document(db, make_user, monkeypatch, tmp_path):
    # 存储目录换成临时目录
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    user = make_user()

    def _make(content: bytes, extension: str = ".md") -> Document:
        doc = Document(
            user_id=user.id,
            filename=f"test{extension}",
            storage_path=save_file(user.id, extension, content),
            file_size=len(content),
            content_hash=hashlib.sha256(content).hexdigest(),
        )
        db.add(doc)
        # 必须 commit 而不是 flush：被测代码会调用 db.rollback()，
        # 没提交的数据会被一起回滚掉（测试结束时外层事务仍然会撤销所有数据）
        db.commit()
        return doc

    return _make


# 让 tasks.py 使用测试的 db 会话，而不是连开发数据库。
# 原代码是 with SessionLocal() as db，这里把 SessionLocal() 换成一个直接产出测试会话的 with 上下文
@pytest.fixture
def task_db(db, monkeypatch):
    monkeypatch.setattr(tasks, "SessionLocal", lambda: contextlib.nullcontext(db))
    return db


# 查某个文档的所有段落内容，按序号排列
def chunk_contents(db, document_id: int) -> list[str]:
    stmt = select(Chunk.content).where(Chunk.document_id == document_id).order_by(Chunk.chunk_index)
    return list(db.scalars(stmt))


# ============================ 切分 ============================


# 测试：按窗口切分，相邻两段有重叠，最后不多出一段小尾巴
def test_split_text_with_overlap():
    assert processing.split_text("abcdefghij", size=5, overlap=2) == ["abcde", "defgh", "ghij"]


# 测试：短文本只切出一段；空文本和纯空白切不出任何段
def test_split_text_short_and_empty():
    assert processing.split_text("hello", size=5, overlap=2) == ["hello"]
    assert processing.split_text("") == []
    assert processing.split_text("   \n\n  ") == []


# 测试：切分后不丢内容。把每段去掉开头的重叠部分再拼起来，应该等于原文
def test_split_text_covers_everything():
    text = "".join(str(i % 10) for i in range(1234))  # 1234 个字符，没有空白
    chunks = processing.split_text(text, size=100, overlap=10)
    # 第一段完整保留，后面每段去掉开头 10 个重叠字符
    assert chunks[0] + "".join(c[10:] for c in chunks[1:]) == text


# ============================ 解析 ============================


# 测试：能从 PDF 里提取出文字
def test_extract_text_from_pdf():
    assert "Hello PDF" in processing.extract_text(make_pdf("Hello PDF"), ".pdf")


# 测试：提取结果里的空字符（\x00）被去掉，否则 PostgreSQL 存不进去
def test_extract_text_removes_nul():
    assert processing.extract_text(b"a\x00b", ".txt") == "ab"


# ============================ 正常处理 ============================


# 测试：处理成功，状态变成 ready，段落按顺序写入，记录了段数和完成时间
def test_process_document_success(db, make_document):
    doc = make_document(("x" * 1200).encode())
    processing.process_document(db, doc.id)
    db.refresh(doc)  # 从数据库重新读取最新的值
    assert doc.status == DocumentStatus.READY
    assert doc.chunk_count == 3  # 1200 个字符，每段 500、重叠 50：0-500、450-950、900-1200
    assert doc.processed_at is not None
    assert doc.error_message is None
    assert len(chunk_contents(db, doc.id)) == 3


# 测试：PDF 文档也能走通完整流程
def test_process_pdf_document(db, make_document):
    doc = make_document(make_pdf("Hello PDF"), ".pdf")
    processing.process_document(db, doc.id)
    db.refresh(doc)
    assert doc.status == DocumentStatus.READY
    assert "Hello PDF" in chunk_contents(db, doc.id)[0]


# ============================ 幂等 ============================


# 测试：同一个文档处理两次（重复投递），结果和处理一次完全一样，没有重复段落
def test_process_twice_is_idempotent(db, make_document):
    doc = make_document(("y" * 1200).encode())
    processing.process_document(db, doc.id)
    first = chunk_contents(db, doc.id)
    processing.process_document(db, doc.id)
    assert chunk_contents(db, doc.id) == first
    assert db.scalar(select(func.count()).select_from(Chunk).where(Chunk.document_id == doc.id)) == 3


# 测试：worker 崩溃后消息被重新投递（文档停在 processing，可能还留着旧段落），
# 再处理一次会把旧段落替换掉，而不是叠加
def test_reprocess_after_crash_replaces_old_chunks(db, make_document):
    doc = make_document(b"real content")
    # 模拟上一次执行留下的残留数据
    db.add(Chunk(document_id=doc.id, chunk_index=0, content="stale"))
    db.add(Chunk(document_id=doc.id, chunk_index=1, content="stale"))
    doc.status = DocumentStatus.PROCESSING
    db.commit()
    processing.process_document(db, doc.id)
    assert chunk_contents(db, doc.id) == ["real content"]


# 测试：已经失败的文档可以重新处理，成功后错误信息被清掉
def test_failed_document_can_be_reprocessed(db, make_document):
    doc = make_document(b"content")
    doc.status = DocumentStatus.FAILED
    doc.error_message = "old error"
    db.commit()
    processing.process_document(db, doc.id)
    db.refresh(doc)
    assert doc.status == DocumentStatus.READY
    assert doc.error_message is None


# 测试：任务排队期间文档被删除，任务直接结束，不报错
def test_deleted_document_is_skipped(db):
    processing.process_document(db, 999_999)  # 不存在的 id，不应该抛异常


# 测试：标记失败时，不会把已经 ready 的文档改成 failed（另一个重复任务已经成功了）
def test_mark_failed_does_not_overwrite_ready(db, make_document):
    doc = make_document(b"content")
    processing.process_document(db, doc.id)
    processing.mark_failed(db, doc.id, "late error from a duplicate task")
    db.refresh(doc)
    assert doc.status == DocumentStatus.READY


# ============================ 永久性错误：不重试，直接失败 ============================


# parametrize：同一个测试用不同的参数跑多次，每组参数算一个独立的测试
# 每组是 (怎么制造问题, 错误信息里应该包含的关键词)
@pytest.mark.parametrize(
    ("break_it", "expected"),
    [
        # 磁盘上的文件被删掉了
        (lambda doc, tmp: next(tmp.rglob("*.*")).unlink(), "missing"),
        # 文件内容被改动了，和哈希对不上
        (lambda doc, tmp: next(tmp.rglob("*.*")).write_bytes(b"tampered"), "hash"),
    ],
)
def test_permanent_errors_mark_failed(task_db, make_document, tmp_path, break_it, expected):
    doc = make_document(b"content")
    break_it(doc, tmp_path)
    # apply()：在当前进程里同步执行任务（eager 模式），不经过 Redis
    tasks.process_document.apply(args=[doc.id])
    task_db.refresh(doc)
    assert doc.status == DocumentStatus.FAILED
    assert expected in doc.error_message


# 测试：扫描版 PDF（提取不出文字）和损坏的 PDF，都标记为失败
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (make_blank_pdf(), "No text"),
        (b"%PDF-1.4 this is not really a pdf", "Cannot parse PDF"),
    ],
)
def test_bad_pdf_marks_failed(task_db, make_document, content, expected):
    doc = make_document(content, ".pdf")
    tasks.process_document.apply(args=[doc.id])
    task_db.refresh(doc)
    assert doc.status == DocumentStatus.FAILED
    assert expected in doc.error_message


# 测试：永久性错误不会重试（只执行一次）
def test_permanent_error_is_not_retried(task_db, make_document, monkeypatch):
    doc = make_document(b"content")
    attempts = []

    # 假的 read_and_split：记录被调用了几次，每次都抛永久性错误
    def always_permanent(d):
        attempts.append(1)
        raise processing.PermanentError("bad file")

    monkeypatch.setattr(processing, "read_and_split", always_permanent)
    tasks.process_document.apply(args=[doc.id])
    assert len(attempts) == 1


# ============================ 临时性错误：重试 ============================


# 测试：前两次遇到临时错误，第三次成功，最终状态是 ready
def test_transient_error_is_retried_then_succeeds(task_db, make_document, monkeypatch):
    doc = make_document(b"content")
    real = processing.read_and_split  # 先保存真正的函数
    attempts = []

    # 假的 read_and_split：前两次抛 OSError（模拟磁盘暂时读不了），之后调用真正的函数
    def flaky(d):
        attempts.append(1)
        if len(attempts) <= 2:
            raise OSError("disk busy")
        return real(d)

    monkeypatch.setattr(processing, "read_and_split", flaky)
    tasks.process_document.apply(args=[doc.id])
    task_db.refresh(doc)
    assert len(attempts) == 3
    assert doc.status == DocumentStatus.READY


# 测试：一直失败，重试次数用完后标记为 failed（一共执行 1 + 3 次）
def test_transient_error_gives_up_after_max_retries(task_db, make_document, monkeypatch):
    doc = make_document(b"content")
    attempts = []

    def always_broken(d):
        attempts.append(1)
        raise OSError("disk busy")

    monkeypatch.setattr(processing, "read_and_split", always_broken)
    result = tasks.process_document.apply(args=[doc.id])
    task_db.refresh(doc)
    assert len(attempts) == 1 + tasks.MAX_RETRIES
    assert doc.status == DocumentStatus.FAILED
    assert "Failed after 3 retries" in doc.error_message
    # Celery 把这次任务记为失败
    assert result.failed()


# ============================ 和上传接口的衔接 ============================


# 测试：上传成功后投递了任务，投递的是新文档的 id
def test_upload_enqueues_processing(client, register_and_login, enqueued):
    headers = register_and_login("alice")
    r = client.post("/documents", headers=headers, files={"file": ("a.md", b"# hi", "text/markdown")})
    assert enqueued == [r.json()["id"]]


# 测试：重复上传相同内容（返回已有文档），不会再投递一次
def test_duplicate_upload_does_not_enqueue_again(client, register_and_login, enqueued):
    headers = register_and_login("alice")
    for _ in range(2):
        client.post("/documents", headers=headers, files={"file": ("a.md", b"# hi", "text/markdown")})
    assert len(enqueued) == 1


# 测试：投递失败（Redis 挂了），上传仍然成功，文档停在 pending 等待重新投递
def test_upload_succeeds_when_queue_is_down(client, register_and_login, monkeypatch):
    # 假的任务对象：delay 一调用就抛出连接错误
    class BrokenQueue:
        @staticmethod
        def delay(document_id):
            raise ConnectionError("redis is down")

    monkeypatch.setattr("app.routers.documents.process_document", BrokenQueue)
    headers = register_and_login("alice")
    r = client.post("/documents", headers=headers, files={"file": ("a.md", b"# hi", "text/markdown")})
    assert r.status_code == 201
    assert r.json()["status"] == "pending"


# 测试：完整流程——上传 → 后台处理 → 查询详情能看到 ready（这就是前端轮询进度的方式）
def test_progress_visible_through_api(client, db, register_and_login, enqueued):
    headers = register_and_login("alice")
    doc_id = client.post(
        "/documents", headers=headers, files={"file": ("a.md", b"# hello world", "text/markdown")}
    ).json()["id"]
    # 处理之前：pending
    assert client.get(f"/documents/{doc_id}", headers=headers).json()["status"] == "pending"
    # 模拟 worker 执行刚才投递的任务
    processing.process_document(db, enqueued[0])
    body = client.get(f"/documents/{doc_id}", headers=headers).json()
    assert body["status"] == "ready"
    assert body["chunk_count"] == 1


# 测试：embedding 服务挂了属于临时性错误，任务会重试；一直挂着就在重试用完后标记为失败
def test_embedding_outage_is_retried_then_fails(task_db, make_document, fake_llm):
    doc = make_document(b"content")
    fake_llm.fail = True
    tasks.process_document.apply(args=[doc.id])
    task_db.refresh(doc)
    assert doc.status == DocumentStatus.FAILED
    assert "Failed after 3 retries" in doc.error_message
    assert "LLMUnavailable" in doc.error_message
