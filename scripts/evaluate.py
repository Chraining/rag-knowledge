"""Measure retrieval quality on a labeled question set.

    uv run python -m scripts.evaluate --check-only                # 只检查标注，不需要数据库
    uv run python -m scripts.evaluate                             # 自建评测集，线上的切分参数（段长 500、重叠 50）
    uv run python -m scripts.evaluate --chunk-size 300            # 换一个段长，和上面的结果对比
    uv run python -m scripts.evaluate --dataset duretrieval       # 公开数据集 DuRetrieval 的子集
    uv run python -m scripts.evaluate --mode hybrid               # 混合检索（向量 + 关键词），和默认的向量检索对比
    uv run python -m scripts.evaluate --mode rerank               # 向量检索前 20 段，再用重排序模型排序（调用付费接口）

评测检索效果。用法见上面的命令。

两份评测集：
  self（默认）：自建的，在 eval/ 目录
    eval/corpus/          测试文档（虚构公司的制度、说明书、会议纪要、价格表）
    eval/questions.jsonl  标注好的问题，每行一个：问题、答案片段、类型（keyword / semantic）
    判断"找到了"：检索结果里某一段包含答案片段。不按段落 id 标注，是因为改了段长之后段落 id 全变了，
    而答案片段不受影响，不同的切分方式可以用同一份标注比较
  duretrieval：公开数据集 DuRetrieval 的子集（300 个查询、1 万段），先用 scripts/build_duretrieval.py 生成
    每个原始段落存成一个文档，标注的是"哪些段落和这个查询相关"（平均约 5 段）
    判断"找到了"：检索结果里的某一段来自一个相关段落

两个指标：
  Recall@k：相关内容中有多少比例排进了前 k 名。自建评测集每题只有 1 个答案，就是"找没找到"。
            线上给模型的是前 5 段，所以 Recall@5 最重要
  MRR：每个问题取 1 ÷ 第一个相关段落的排名（第 1 名得 1 分，第 2 名 0.5 分，没找到 0 分），再求平均。衡量排得靠不靠前

数据写进单独的数据库 rag_eval（不存在会自动创建），不影响开发库 rag 和测试库 rag_test。
文档向量调用真实的百炼 embedding，算好的向量存进评测库的缓存表（见 embedding_cache）：
中途失败时已经算好的不会丢；同一份评测集、同样的段长重复运行不再花钱。问题向量有 Redis 缓存。
"""

# argparse：解析命令行参数
import argparse
# hashlib：计算文档的 sha256（documents 表要求有 content_hash）
import hashlib
# json：读取 jsonl 文件
import json
# dataclass：定义只装数据的小类；field：给列表类型的字段设置默认值
from dataclasses import dataclass, field
# Path：处理文件路径
from pathlib import Path

# command、Config：在代码里执行 alembic upgrade head，给评测库建表
from alembic import command
from alembic.config import Config
# Vector：pgvector 的向量列类型（向量缓存表用）
from pgvector.sqlalchemy import Vector
# create_engine：连接数据库；select：查询；text：原生 SQL；Table 等：定义向量缓存表和重排序缓存表
from sqlalchemy import Column, Float, MetaData, String, Table, create_engine, select, text
# insert：PostgreSQL 专用的 INSERT，支持 ON CONFLICT DO NOTHING
from sqlalchemy.dialects.postgresql import insert
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
from app.models import EMBEDDING_DIM, Document, DocumentStatus, User
# 线上用的切分函数和默认参数；save_chunks：线上写入段落的函数（写段落 + 把文档改成 ready）
from app.processing import CHUNK_OVERLAP, CHUNK_SIZE, save_chunks, split_lines, split_text
# 检索结果中的一段
from app.queries import RetrievedChunk

# eval 目录（相对于项目根目录）
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
# 自建评测集：测试文档目录、标注文件
CORPUS_DIR = EVAL_DIR / "corpus"
QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"
# 公开数据集子集的目录（由 scripts/build_duretrieval.py 生成）
DURETRIEVAL_DIR = EVAL_DIR / "duretrieval"
# 评测专用的数据库名
EVAL_DATABASE = "rag_eval"
# 每个问题取前几名。MRR 也只在前 10 名里算，排在 10 名以后当作没找到
DEPTH = 10
# 报告哪几个 Recall@k
RECALL_AT = (1, 3, 5, 10)
# 一次调用 embedding 的段数。分批调用只是为了能打印进度（1 万段要几分钟）
EMBED_PROGRESS_BATCH = 500
# rerank 模式：先用向量检索取多少段候选，再交给重排序模型。
# 公开数据集上向量检索前 20 名已经包含 97% 的相关段落，再多取，费用成倍增加，能多找到的很少
RERANK_CANDIDATES = 20
# 没进前 5 名的问题最多列出多少个（公开数据集可能有几十个，全列出来太长）
MAX_MISSES_SHOWN = 20


# 一个标注好的问题。两份评测集的标注方式不同，所以 answer 和 relevant 只会用到其中一个
@dataclass
class Question:
    question: str  # 问题
    type: str  # 自建：keyword（问题里有文档中的编号、型号等原词）或 semantic（换了一种说法提问）；公开：duretrieval
    answer: str = ""  # 自建评测集：答案片段，正确段落里的一小段原文
    relevant: list[str] = field(default_factory=list)  # 公开数据集：相关段落的 id（也是文档的文件名）


# 一个问题的评测结果
@dataclass
class Result:
    question: Question
    rank: int | None  # 第一个相关段落排第几（从 1 开始）；前 DEPTH 名里没有就是 None
    score: float | None  # 这一段的相似度
    recall: dict[int, float]  # {k: Recall@k}


# 读 jsonl：每行一个 JSON，跳过空行。
# 用 split("\n") 而不是 splitlines()：splitlines 还会在   这类不常见的换行符处切开，
# 网页段落里偶尔有这种字符（JSON 里允许原样出现），会把一行 JSON 切成两半
def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


# 读取一份评测集，返回（{文档名: 文字}，问题列表）
def load_dataset(name: str) -> tuple[dict[str, str], list[Question]]:
    if name == "self":
        # 用线上同一个函数提取文字，CSV 会变成"列名: 值"的格式
        corpus = {path.name: extract_text(path.read_bytes(), path.suffix) for path in sorted(CORPUS_DIR.iterdir())}
        questions_file = QUESTIONS_FILE
    else:
        if not (DURETRIEVAL_DIR / "corpus.jsonl").exists():
            raise SystemExit("还没有生成公开数据集子集，先运行：uv run --with pyarrow python -m scripts.build_duretrieval")
        # 每个段落当作一个文档，文件名就是段落 id
        corpus = {row["id"]: row["text"] for row in read_jsonl(DURETRIEVAL_DIR / "corpus.jsonl")}
        questions_file = DURETRIEVAL_DIR / "questions.jsonl"
    # **row 把字典展开成关键字参数，例如 Question(question=..., answer=..., type=...)
    return corpus, [Question(**row) for row in read_jsonl(questions_file)]


# 按线上的规则切分：表格按整行切，其他按字数切
def split(filename: str, text_: str, size: int, overlap: int) -> list[str]:
    return split_lines(text_, size) if Path(filename).suffix in TABLE_EXTENSIONS else split_text(text_, size, overlap)


# 检查标注，返回发现的问题（空列表表示没问题）。
# 标注错了，算出来的分数也是错的，而且看不出来，所以运行前一定要先检查
def check_labels(corpus: dict[str, str], questions: list[Question], overlap: int) -> list[str]:
    errors = []
    for q in questions:
        # 公开数据集：相关段落必须都在子集里
        if not q.answer:
            missing = [pid for pid in q.relevant if pid not in corpus]
            if not q.relevant or missing:
                errors.append(f"相关段落不在文档里：{q.question!r} {missing}")
            continue
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


# 向量缓存表（只在评测库里，不属于应用的表，所以不走迁移）：{模型名 + 文字的哈希: 向量}。
# 每算完一批就存进来并提交：中途失败（网络断了、账户欠费）时，已经算好的向量不会丢，下次运行接着算；
# 换段长、换检索方式重跑时，文字相同的段落也直接用缓存，不再花钱
embedding_cache = Table(
    "embedding_cache",
    MetaData(),
    Column("key", String(64), primary_key=True),
    Column("embedding", Vector(EMBEDDING_DIM)),
)


# 缓存的键：模型名 + 文字一起算哈希。换了 embedding 模型，向量就不一样了，不能用旧缓存
def cache_key(text_: str) -> str:
    return hashlib.sha256(f"{settings.embedding_model}\n{text_}".encode()).hexdigest()


# 返回 {段落文字: 向量}。缓存里有的直接读，没有的调用 embedding，分批算、每批存进缓存并打印进度
def embed_all(db: Session, texts: list[str]) -> dict:
    # checkfirst=True：表已经存在就不建
    embedding_cache.create(db.connection(), checkfirst=True)
    # dict.fromkeys 去重并保持顺序：同样的文字只算一次
    keys = {t: cache_key(t) for t in dict.fromkeys(texts)}
    # 一次查出缓存里已有的：{键: 向量}
    rows = db.execute(select(embedding_cache).where(embedding_cache.c.key.in_(list(keys.values()))))
    cached = {key: vector for key, vector in rows}
    missing = [t for t, key in keys.items() if key not in cached]
    print(f"向量：{len(keys) - len(missing)} 段用缓存，{len(missing)} 段需要调用 embedding")
    for start in range(0, len(missing), EMBED_PROGRESS_BATCH):
        batch = missing[start : start + EMBED_PROGRESS_BATCH]
        new = [{"key": keys[t], "embedding": v} for t, v in zip(batch, llm.embed_texts(batch), strict=True)]
        # on_conflict_do_nothing：万一键已经存在（比如两个评测同时在跑）就跳过，不报错
        db.execute(insert(embedding_cache).on_conflict_do_nothing(), new)
        # 每批立刻提交，存进数据库
        db.commit()
        cached.update((row["key"], row["embedding"]) for row in new)
        print(f"  {start + len(batch)} / {len(missing)}")
    return {t: cached[key] for t, key in keys.items()}


# 重排序缓存表，和向量缓存表一样只在评测库里：{模型名 + 问题 + 段落的哈希: 相关度分数}。
# 每个问题算完就提交：中途失败时已经算好的不会丢；重复运行、换别的评测集再换回来，都不再花钱
rerank_cache = Table(
    "rerank_cache",
    MetaData(),
    Column("key", String(64), primary_key=True),
    Column("score", Float),
)


# 给每段打重排序分数，返回的顺序和 chunks 一致。缓存里有的直接读，没有的才调用接口
def rerank_cached(db: Session, question: str, chunks: list[RetrievedChunk]) -> list[float]:
    rerank_cache.create(db.connection(), checkfirst=True)
    # 键里带上模型名和问题：同一段文字，对不同的问题分数不一样
    keys = [hashlib.sha256(f"{settings.rerank_model}\n{question}\n{c.content}".encode()).hexdigest() for c in chunks]
    cached = {key: score for key, score in db.execute(select(rerank_cache).where(rerank_cache.c.key.in_(keys)))}
    # 缓存里没有的段落（dict.fromkeys 去重：同样的键只算一次）
    missing = [(key, c.content) for key, c in dict(zip(keys, chunks)).items() if key not in cached]
    if missing:
        # 只把缺的段落发给模型。重排序模型对每一对（问题, 段落）单独打分，分开发送不影响分数
        scores = llm.rerank(question, [content for _, content in missing])
        new = [{"key": key, "score": score} for (key, _), score in zip(missing, scores, strict=True)]
        db.execute(insert(rerank_cache).on_conflict_do_nothing(), new)
        db.commit()
        cached.update((row["key"], row["score"]) for row in new)
    return [cached[key] for key in keys]


# 清空评测库，重新写入全部文档的段落和向量。返回评测用户的 id
def build_index(db: Session, corpus: dict[str, str], size: int, overlap: int) -> int:
    # 先切好所有文档：[(文档名, 文字, 段落列表), ...]
    docs = [(filename, text_, split(filename, text_, size, overlap)) for filename, text_ in corpus.items()]
    # 先把所有段落的向量准备好，再动数据库里的文档。所有段落一起算，而不是每个文档调用一次：
    # 公开数据集有 1 万个文档，逐个调用要多花很多倍时间
    vectors = embed_all(db, [chunk for _, _, chunks in docs for chunk in chunks])
    print("写入数据库 ...")
    # 每次都从头建：段长变了，段落就全变了。CASCADE：连带清空引用这些表的其他表
    db.execute(text("TRUNCATE chunks, documents, users RESTART IDENTITY CASCADE"))
    user = User(username="eval", email="eval@example.com", password_hash="x")
    db.add(user)
    # flush：执行 INSERT，拿到 user.id
    db.flush()
    for filename, text_, chunks in docs:
        content = text_.encode()
        doc = Document(
            user_id=user.id,
            filename=filename,
            storage_path=f"eval/{filename}",  # 评测不读这个路径，只是表里要求有值
            file_size=len(content),
            content_hash=hashlib.sha256(content).hexdigest(),
            status=DocumentStatus.PROCESSING,
        )
        db.add(doc)
        db.flush()
        # 和线上一样用 save_chunks 写入段落、把文档改成 ready（它会 commit）
        save_chunks(db, doc.id, chunks, [vectors[chunk] for chunk in chunks])
    return user.id


# 一个段落里找到了这个问题的哪些相关内容（返回集合，空集合表示这一段不相关）
def relevant_items(q: Question, chunk: RetrievedChunk) -> set[str]:
    # 自建评测集：段落里包含答案片段，就算找到了这个答案
    if q.answer:
        return {q.answer} if q.answer in chunk.content else set()
    # 公开数据集：这一段来自某个相关段落（每个原始段落存成一个文档，文件名就是它的 id）。& 是求交集
    return {chunk.filename} & set(q.relevant)


# 用哪种方式检索，前 DEPTH 段。vector 是线上用的方式；keyword、hybrid、rerank 是试过的改进，和它对比
def search(db: Session, user_id: int, mode: str, question: str) -> list[queries.RetrievedChunk]:
    # 问题转向量（带缓存），和线上一样
    vector = rag.embed_query(question)
    if mode == "vector":
        return queries.search_chunks(db, user_id, vector, DEPTH)
    if mode == "rerank":
        candidates = queries.search_chunks(db, user_id, vector, RERANK_CANDIDATES)
        scores = rerank_cached(db, question, candidates)
        # 按重排序分数从高到低重新排列候选
        order = sorted(range(len(candidates)), key=scores.__getitem__, reverse=True)
        return [candidates[i] for i in order[:DEPTH]]
    if mode == "keyword":
        return queries.keyword_search_chunks(db, user_id, question, vector, DEPTH)
    return queries.hybrid_search_chunks(db, user_id, question, vector, DEPTH)


# 对每个问题执行一次检索，算出它的排名和各个 Recall@k
def evaluate(db: Session, user_id: int, questions: list[Question], mode: str) -> list[Result]:
    results = []
    for q in questions:
        chunks = search(db, user_id, mode, q.question)
        # 一共有几个相关内容：自建评测集是 1 个答案，公开数据集是相关段落的个数
        total = 1 if q.answer else len(q.relevant)
        rank, score, found, recall = None, None, set(), {}
        # enumerate(..., start=1)：序号从 1 开始，就是排名
        for i, chunk in enumerate(chunks, start=1):
            items = relevant_items(q, chunk)
            # 第一个相关段落：记下排名和相似度
            if items and rank is None:
                rank, score = i, chunk.score
            # |= 求并集：累计到目前为止找到的相关内容（同一个相关段落被切成两段时，只算一次）
            found |= items
            if i in RECALL_AT:
                recall[i] = len(found) / total
        # 检索结果不够 k 个时（文档很少），没算到的 k 就用最终找到的数量
        for k in RECALL_AT:
            recall.setdefault(k, len(found) / total)
        results.append(Result(q, rank, score, recall))
    return results


# 一组结果的各项指标：Recall@1/3/5/10 和 MRR
def summary_row(label: str, results: list[Result]) -> str:
    n = len(results)
    # 每个 k：所有问题的 Recall@k 求平均
    recalls = [sum(r.recall[k] for r in results) / n for k in RECALL_AT]
    # 每个问题 1 ÷ 排名，没找到算 0，再求平均
    mrr = sum(1 / r.rank if r.rank else 0 for r in results) / n
    cells = "".join(f"{v:>11.1%}" for v in recalls)
    return f"{label:<12}{n:>6}{cells}{mrr:>9.3f}"


# 打印报告
def report(results: list[Result]) -> None:
    header = "".join(f"{f'Recall@{k}':>11}" for k in RECALL_AT)
    print(f"{'类型':<12}{'问题数':>6}{header}{'MRR':>9}")
    print(summary_row("全部", results))
    # 有多种问题类型时（自建评测集的 keyword、semantic），分别统计
    types = sorted({r.question.type for r in results})
    if len(types) > 1:
        for type_ in types:
            print(summary_row(type_, [r for r in results if r.question.type == type_]))
    # 线上只把前 5 段给模型，前 5 段里一个相关的都没有，就是"漏召回"。列出来，方便分析原因
    misses = [r for r in results if r.rank is None or r.rank > 5]
    print(f"\n前 5 名里没有相关段落的问题（{len(misses)} 个{'，只列出前 ' + str(MAX_MISSES_SHOWN) + ' 个' if len(misses) > MAX_MISSES_SHOWN else ''}）：")
    for r in misses[:MAX_MISSES_SHOWN]:
        where = f"第 {r.rank} 名" if r.rank else f"前 {DEPTH} 名里没有"
        print(f"  [{r.question.type}] {r.question.question}  →  {where}")
    # 进了前 5，但相似度低于线上的阈值：线上会被当作"不相关"丢掉，模型同样看不到
    low = [r for r in results if r.rank is not None and r.rank <= 5 and r.score < settings.retrieval_min_score]
    print(f"\n第一个相关段落进了前 5 但相似度低于阈值 {settings.retrieval_min_score} 的问题：{len(low)} 个")


def main() -> None:
    parser = argparse.ArgumentParser(description="评测检索效果")
    parser.add_argument("--dataset", choices=["self", "duretrieval"], default="self", help="用哪份评测集")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="每段最多多少个字符")
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP, help="相邻两段重叠多少个字符")
    parser.add_argument(
        "--mode", choices=["vector", "keyword", "hybrid", "rerank"], default="vector", help="检索方式，默认是线上用的向量检索"
    )
    parser.add_argument("--check-only", action="store_true", help="只检查标注和切分，不连数据库、不调用 embedding")
    args = parser.parse_args()

    corpus, questions = load_dataset(args.dataset)
    errors = check_labels(corpus, questions, args.overlap)
    if errors:
        # SystemExit：打印信息并以非 0 状态退出
        raise SystemExit("标注有问题：\n" + "\n".join(errors))
    total = sum(len(split(name, text_, args.chunk_size, args.overlap)) for name, text_ in corpus.items())
    print(
        f"标注检查通过（{args.dataset}）：{len(corpus)} 个文档，{len(questions)} 个问题；"
        f"段长 {args.chunk_size}、重叠 {args.overlap}，共 {total} 段"
    )
    if args.check_only:
        return

    engine = prepare_database()
    with Session(engine) as db:
        user_id = build_index(db, corpus, args.chunk_size, args.overlap)
        print(f"\n检索方式：{args.mode}")
        report(evaluate(db, user_id, questions, args.mode))
    engine.dispose()


if __name__ == "__main__":
    main()
