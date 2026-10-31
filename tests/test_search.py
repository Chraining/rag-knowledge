# 测试混合检索：中文分词、关键词检索（权限隔离、IDF）、RRF 融合

# select：直接查数据库做验证
from sqlalchemy import select
# Vector：save_chunks 需要的向量类型
from pgvector import Vector

# 被测试的模块
from app import queries
from app.models import Chunk, Document, DocumentStatus
from app.processing import save_chunks
from app.queries import RetrievedChunk
from app.segment import segment
# 假 embedding：测试里直接算问题的向量
from tests.conftest import fake_embedding


# ============================ 辅助函数 ============================


# 给 user 建一个已处理完成的文档，每个字符串是一段，返回文档
def add_doc(db, user, *chunks: str, name: str = "a.md") -> Document:
    doc = Document(
        user_id=user.id,
        filename=name,
        storage_path=f"storage/{user.id}/{name}",
        file_size=1,
        # 每个文档的哈希都不同，不会触发唯一约束
        content_hash=f"{user.id}/{name}".ljust(64, "0"),
        status=DocumentStatus.PROCESSING,
    )
    db.add(doc)
    db.commit()
    # 用线上写入段落的函数，这样 terms 列也是按线上的方式生成的
    save_chunks(db, doc.id, list(chunks), [Vector(fake_embedding(c)) for c in chunks])
    return doc


# 关键词检索，只返回段落内容
def keyword(db, user, query: str, top_k: int = 5) -> list[str]:
    return [c.content for c in queries.keyword_search_chunks(db, user.id, query, fake_embedding(query), top_k)]


# 造一个检索结果，只关心 id
def chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(chunk_id, 1, "a.md", f"chunk {chunk_id}", 0.5)


# ============================ 分词 ============================


# 测试：分词去掉标点和空格，统一小写，搜索模式下长词里的短词也切出来
def test_segment():
    words = segment("路由器报E171怎么处理？")
    assert "e171" in words
    assert {"路由", "路由器"} <= set(words)
    assert "？" not in words


# 测试：写入段落时同时写入了分词结果
def test_save_chunks_stores_terms(db, make_user):
    doc = add_doc(db, make_user(), "故障代码 E171 表示 DHCP 地址池已满")
    terms = db.scalar(select(Chunk.terms).where(Chunk.document_id == doc.id))
    # tsvector 转成文字的样子，例如 'dhcp':4 'e171':3 ……
    assert "'e171'" in terms and "'dhcp'" in terms


# ============================ 关键词检索 ============================


# 测试：型号只差一个字也能分清楚（E171 和 E117），这正是向量检索容易混淆的地方
def test_keyword_search_matches_exact_code(db, make_user):
    user = make_user()
    add_doc(db, user, "E117 IP 地址冲突", "E171 DHCP 地址池已满")
    assert keyword(db, user, "E171 是什么故障") == ["E171 DHCP 地址池已满"]


# 测试：IDF 起作用。常见词出现得再多，也比不上命中一个少见的关键词
def test_keyword_search_prefers_rare_words(db, make_user):
    user = make_user()
    # "怎么样" 在很多段落里都有，"8218" 只有一段有
    common = [f"这款{i}号产品怎么样？怎么样都行，怎么样都好" for i in range(10)]
    add_doc(db, user, *common, "老板8218 油烟机评测")
    assert keyword(db, user, "老板8218怎么样", top_k=1) == ["老板8218 油烟机评测"]


# 测试：关键词检索也只能搜到自己的、已处理完成的文档
def test_keyword_search_only_own_ready_documents(db, make_user):
    alice, bob = make_user(), make_user()
    add_doc(db, bob, "bob 的工资是 50000")
    pending = add_doc(db, alice, "alice 的工资条", name="pending.md")
    pending.status = DocumentStatus.PENDING
    db.commit()
    assert keyword(db, alice, "工资") == []


# 测试：问题里一个词都没有（只有标点）时返回空，不报错
def test_keyword_search_without_words(db, make_user):
    user = make_user()
    add_doc(db, user, "一些内容")
    assert keyword(db, user, "？？！") == []


# ============================ 融合 ============================


# 测试：RRF 只看名次。两路都排得靠前的段落排第一；只在一路里出现的按名次排在后面
def test_fuse_prefers_chunks_found_by_both():
    vector = [chunk(1), chunk(2), chunk(3)]
    keyword_ = [chunk(3), chunk(4)]
    assert [c.chunk_id for c in queries.fuse([vector, keyword_], top_k=3)] == [3, 1, 2]


# 测试：只有关键词检索找得到的段落，混合检索的结果里也有它。
# 假 embedding 按 \w+ 切词，"E171怎么办" 被当成一个整体，和任何段落都没有共同的词，向量检索的顺序等于随机；
# 关键词检索靠 jieba 切出的 "e171" 把它排在第 1。
# RRF 下关键词第 1 名的得分（1/61）不低于向量第 1 名，所以前 2 名里一定有它
def test_hybrid_search_includes_keyword_hits(db, make_user):
    user = make_user()
    add_doc(db, user, "故障代码E171表示地址池已满", *[f"其他说明第{i}条" for i in range(30)])
    query = "E171怎么办"
    results = queries.hybrid_search_chunks(db, user.id, query, fake_embedding(query), top_k=2)
    assert "故障代码E171表示地址池已满" in [c.content for c in results]
    # 返回的 score 仍然是余弦相似度（0-1），不是 RRF 得分
    assert all(0 <= c.score <= 1 for c in results)

