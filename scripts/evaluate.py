"""Measure retrieval quality on a labeled question set.

    uv run python -m scripts.evaluate --check-only       # 只检查标注，不需要数据库
    uv run python -m scripts.evaluate                    # 用线上的切分参数评测（段长 500、重叠 50）
    uv run python -m scripts.evaluate --chunk-size 300   # 换一个段长，和上面的结果对比

评测检索效果。用法见上面的命令。

测试数据在 eval/ 目录：
  eval/corpus/          测试文档（虚构公司的制度、说明书、会议纪要、价格表）
  eval/questions.jsonl  标注好的问题，每行一个：问题、答案片段、类型

怎么判断"找到了"：每个问题标注了正确内容里的一小段文字（答案片段），检索结果里某一段包含这段文字就算命中。
不按段落 id 标注，是因为改了段长之后段落 id 全变了，而答案片段不受影响，不同的切分方式可以用同一份标注比较。

两个指标：
  Recall@k：正确段落排进前 k 名的问题占多少比例。线上给模型的是前 5 段，所以 Recall@5 最重要
  MRR：每个问题取 1 ÷ 正确段落的排名（第 1 名得 1 分，第 2 名 0.5 分，没找到 0 分），再求平均。衡量排得靠不靠前

数据写进单独的数据库 rag_eval（不存在会自动创建），不影响开发库 rag 和测试库 rag_test。
文档向量调用真实的百炼 embedding（每次运行约几十段，费用不到一分钱）；问题向量有 Redis 缓存，重复运行不再花钱。
"""

# argparse：解析命令行参数
import argparse
# hashlib：计算文档的 sha256（documents 表要求有 content_hash）
import hashlib
# json：读取 questions.jsonl
import json
# dataclass：定义只装数据的小类
from dataclasses import dataclass
# Path：处理文件路径
from pathlib import Path

# command、Config：在代码里执行 alembic upgrade head，给评测库建表
from alembic import command
from alembic.config import Config
# create_engine：连接数据库；text：原生 SQL
from sqlalchemy import create_engine, text
# make_url：解析数据库地址，把其中的库名换成 rag_eval
from sqlalchemy.engine import make_url
# Session：数据库会话
from sqlalchemy.orm import Session

# llm：文档转向量；queries：线上用的向量检索；rag：问题转向量（带缓存）
from app import llm, queries, rag
# 读取配置（数据库地址、最低相似度）
from app.config import settings
# 提取文字；表格类格式（按行切分）
from app.extract import TABLE_EXTENSIONS, extract_text
# 模型
from app.models import Document, DocumentStatus, User
# 线上用的切分函数和默认参数；save_chunks：线上写入段落的函数（写段落 + 把文档改成 ready）
from app.processing import CHUNK_OVERLAP, CHUNK_SIZE, save_chunks, split_lines, split_text
# 检索结果中的一段
from app.queries import RetrievedChunk

# eval 目录（相对于项目根目录）
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
# 测试文档目录
CORPUS_DIR = EVAL_DIR / "corpus"
# 标注文件
QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"
# 评测专用的数据库名
EVAL_DATABASE = "rag_eval"
# 每个问题取前几名。MRR 也只在前 10 名里算，排在 10 名以后当作没找到
DEPTH = 10
# 报告哪几个 Recall@k
RECALL_AT = (1, 3, 5, 10)


# 一个标注好的问题
@dataclass
class Question:
    question: str  # 问题
    answer: str  # 答案片段：正确段落里的一小段原文
    type: str  # keyword：问题里有文档中的编号、型号等原词；semantic：换了一种说法提问


# 一个问题的评测结果
@dataclass
class Result:
    question: Question
    rank: int | None  # 正确段落排第几（从 1 开始）；前 DEPTH 名里没有就是 None
    score: float | None  # 正确段落的相似度


# 读取全部测试文档，返回 {文件名: 提取出的文字}。用线上同一个函数提取，CSV 会变成"列名: 值"的格式
def load_corpus() -> dict[str, str]:
    return {path.name: extract_text(path.read_bytes(), path.suffix) for path in sorted(CORPUS_DIR.iterdir())}


# 读取标注文件，每行一个 JSON
def load_questions() -> list[Question]:
    lines = QUESTIONS_FILE.read_text(encoding="utf-8").splitlines()
    # 跳过空行。**json.loads(...) 把字典展开成关键字参数，例如 Question(question=..., answer=..., type=...)
    return [Question(**json.loads(line)) for line in lines if line.strip()]


# 按线上的规则切分：表格按整行切，其他按字数切
def split(filename: str, text_: str, size: int, overlap: int) -> list[str]:
    return split_lines(text_, size) if Path(filename).suffix in TABLE_EXTENSIONS else split_text(text_, size, overlap)


# 检查标注，返回发现的问题（空列表表示没问题）。
# 标注错了，算出来的分数也是错的，而且看不出来，所以运行前一定要先检查
def check_labels(corpus: dict[str, str], questions: list[Question], overlap: int) -> list[str]:
    errors = []
    for q in questions:
        # 答案片段在所有文档里一共出现几次
        count = sum(text_.count(q.answer) for text_ in corpus.values())
        # 0 次：片段写错了，永远不可能命中；多次：片段有歧义，命中了也不一定是对的那段
        if count != 1:
            errors.append(f"答案片段出现了 {count} 次（应该恰好 1 次）：{q.answer!r}")
        # 片段比重叠部分长，就可能被切在两段的边界上，哪一段都不完整包含它，明明找到了也算没找到。
        # 不超过重叠长度时，一定有一段完整包含它（表格按整行切，一行不会被切开，也满足）
        if len(q.answer) > overlap:
            errors.append(f"答案片段比重叠长度 {overlap} 还长：{q.answer!r}")
    return errors


# 准备评测数据库：不存在就创建，然后执行迁移建表。返回连接它的引擎
def prepare_database():
    # 把开发库地址里的库名换成 rag_eval，用户名、密码、端口不变
    url = make_url(settings.database_url).set(database=EVAL_DATABASE)
    # CREATE DATABASE 不能在事务里执行，所以用自动提交模式连接开发库来执行它
    admin = create_engine(settings.database_url, isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        # pg_database：PostgreSQL 记录所有数据库的系统表
        if conn.scalar(text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": EVAL_DATABASE}) is None:
            conn.execute(text(f"CREATE DATABASE {EVAL_DATABASE}"))
    admin.dispose()
    # 和测试一样，用真正的迁移文件建表
    cfg = Config("alembic.ini")
    # hide_password=False：默认会把密码显示成 ***，这里需要真实的地址
    cfg.set_main_option("sqlalchemy.url", url.render_as_string(hide_password=False))
    command.upgrade(cfg, "head")
    return create_engine(url)


# 清空评测库，重新写入全部测试文档的段落和向量。返回评测用户的 id
def build_index(db: Session, corpus: dict[str, str], size: int, overlap: int) -> int:
    # 每次都从头建：段长变了，段落就全变了。CASCADE：连带清空引用这些表的其他表
    db.execute(text("TRUNCATE chunks, documents, users RESTART IDENTITY CASCADE"))
    user = User(username="eval", email="eval@example.com", password_hash="x")
    db.add(user)
    # flush：执行 INSERT，拿到 user.id
    db.flush()
    for filename, text_ in corpus.items():
        content = (CORPUS_DIR / filename).read_bytes()
        doc = Document(
            user_id=user.id,
            filename=filename,
            storage_path=f"eval/corpus/{filename}",  # 评测不读这个路径，只是表里要求有值
            file_size=len(content),
            content_hash=hashlib.sha256(content).hexdigest(),
            status=DocumentStatus.PROCESSING,
        )
        db.add(doc)
        db.flush()
        chunks = split(filename, text_, size, overlap)
        # 和线上一样：调用 embedding，再用 save_chunks 写入段落、把文档改成 ready（它会 commit）
        save_chunks(db, doc.id, chunks, llm.embed_texts(chunks))
    return user.id


# 找出第一个包含答案片段的段落，返回（它排第几（从 1 开始），这一段）；没有就返回 (None, None)
def rank_of(answer: str, chunks: list[RetrievedChunk]) -> tuple[int | None, RetrievedChunk | None]:
    # enumerate(..., start=1)：序号从 1 开始
    for rank, chunk in enumerate(chunks, start=1):
        if answer in chunk.content:
            return rank, chunk
    return None, None


# 对每个问题执行一次线上的检索，记录正确段落的排名
def evaluate(db: Session, user_id: int, questions: list[Question]) -> list[Result]:
    results = []
    for q in questions:
        # 和线上完全相同的两步：问题转向量（带缓存）→ 向量检索（按用户过滤、HNSW 索引）
        chunks = queries.search_chunks(db, user_id, rag.embed_query(q.question), DEPTH)
        rank, chunk = rank_of(q.answer, chunks)
        results.append(Result(q, rank, chunk.score if chunk else None))
    return results


# 打印一组结果的各项指标：Recall@1/3/5/10 和 MRR
def summary_row(label: str, results: list[Result]) -> str:
    n = len(results)
    # 每个 k：排名不为 None 且不超过 k 的问题占多少
    recalls = [sum(r.rank is not None and r.rank <= k for r in results) / n for k in RECALL_AT]
    # 每个问题 1 ÷ 排名，没找到算 0，再求平均
    mrr = sum(1 / r.rank if r.rank else 0 for r in results) / n
    cells = "".join(f"{v:>11.1%}" for v in recalls)
    return f"{label:<10}{n:>6}{cells}{mrr:>9.3f}"


# 打印报告
def report(results: list[Result]) -> None:
    header = "".join(f"{f'Recall@{k}':>11}" for k in RECALL_AT)
    print(f"{'类型':<10}{'问题数':>6}{header}{'MRR':>9}")
    print(summary_row("全部", results))
    # 按问题类型分别统计（keyword、semantic）
    for type_ in sorted({r.question.type for r in results}):
        print(summary_row(type_, [r for r in results if r.question.type == type_]))
    # 线上只把前 5 段给模型，没进前 5 的就是"漏召回"，逐个列出来，方便分析原因
    misses = [r for r in results if r.rank is None or r.rank > 5]
    print(f"\n没进前 5 名的问题（{len(misses)} 个）：")
    for r in misses:
        where = f"第 {r.rank} 名" if r.rank else f"前 {DEPTH} 名里没有"
        print(f"  [{r.question.type}] {r.question.question}  →  {where}")
    # 进了前 5，但相似度低于线上的阈值：线上会被当作"不相关"丢掉，模型同样看不到
    low = [r for r in results if r.rank is not None and r.rank <= 5 and r.score < settings.retrieval_min_score]
    print(f"\n进了前 5 但相似度低于阈值 {settings.retrieval_min_score} 的问题（{len(low)} 个）：")
    for r in low:
        print(f"  [{r.question.type}] {r.question.question}  →  相似度 {r.score:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="评测检索效果")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="每段最多多少个字符")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="相邻两段重叠多少个字符")
    parser.add_argument("--check-only", action="store_true", help="只检查标注和切分，不连数据库、不调用 embedding")
    args = parser.parse_args()

    corpus = load_corpus()
    questions = load_questions()
    errors = check_labels(corpus, questions, args.overlap)
    if errors:
        # SystemExit：打印信息并以非 0 状态退出
        raise SystemExit("标注有问题：\n" + "\n".join(errors))
    total = sum(len(split(name, text_, args.chunk_size, args.overlap)) for name, text_ in corpus.items())
    print(f"标注检查通过：{len(corpus)} 个文档，{len(questions)} 个问题；段长 {args.chunk_size}、重叠 {args.overlap}，共 {total} 段")
    if args.check_only:
        return

    engine = prepare_database()
    with Session(engine) as db:
        user_id = build_index(db, corpus, args.chunk_size, args.overlap)
        print()
        report(evaluate(db, user_id, questions))
    engine.dispose()


if __name__ == "__main__":
    main()
