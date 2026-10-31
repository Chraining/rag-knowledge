"""Query functions.

Rule: every function that reads user-owned data takes `user_id` and filters on it
in SQL. Never load a row by id and check ownership afterwards in Python.

查询函数。
规则：凡是读取"属于某个用户的数据"的函数，都必须接收 user_id 参数，并在 SQL 里按它过滤。
绝不能先按 id 把数据查出来，再在 Python 里判断"是不是这个用户的"——
那种写法在复杂查询里迟早会漏掉一处，造成越权（用户 A 看到用户 B 的数据）。
"""

# math.log：计算 IDF（见 keyword_search_chunks）
import math
# dataclass：一个装饰器，能自动给类生成 __init__ 等方法，适合定义"只装数据的小类"
from dataclasses import dataclass
# datetime：用于类型标注
from datetime import datetime

# func：调用 SQL 函数（这里用 func.count()）
# select：构造 SELECT 查询
# true：SQL 里的布尔值 true（注意不是 Python 的 True）
from sqlalchemy import func, select, text, true
# Session：数据库会话的类型
# aliased：给一个模型（或子查询）起"别名"，让同一个模型能在一条 SQL 里以不同身份出现
from sqlalchemy.orm import Session, aliased

# 导入要查询的模型
from app.models import (
    Chunk,
    Conversation,
    Document,
    DocumentStatus,
    Message,
    UsageAction,
    UsageLog,
    User,
)
# 中文分词，关键词检索用
from app.segment import segment


# ============================ 用户 ============================
# 用户表的查询不需要 user_id 过滤：这些函数用在登录、注册阶段，此时还没有"当前用户"。


# 按用户名查用户，登录时用。用户名已在注册时转成小写，所以这里也要传小写
def get_user_by_username(db: Session, username: str) -> User | None:
    # username 有唯一约束，自带索引，查询很快
    return db.scalar(select(User).where(User.username == username))


# 检查用户名或邮箱是否已被占用，注册时用
def username_or_email_taken(db: Session, username: str, email: str) -> bool:
    # SELECT id FROM users WHERE username = ? OR email = ? LIMIT 1
    # 只要找到任意一行就说明被占用了，所以 LIMIT 1，不需要查出全部
    stmt = select(User.id).where((User.username == username) | (User.email == email)).limit(1)
    # 查到了返回 id（不是 None），否则返回 None
    return db.scalar(stmt) is not None


# ============================ 文档 ============================


# 查询某个用户的文档列表（分页）
# 参数：db 是数据库会话；user_id 是当前用户；limit 是每页条数；offset 是跳过前面多少条
# 返回值类型 list[Document]：返回一个 Document 对象的列表
def list_documents(db: Session, user_id: int, limit: int = 20, offset: int = 0) -> list[Document]:
    # 用"链式调用"一步步构造查询语句。外层的括号只是为了能换行书写
    stmt = (
        select(Document)  # SELECT * FROM documents
        .where(Document.user_id == user_id)  # WHERE user_id = ?  （权限过滤，必须有）
        # ORDER BY created_at DESC, id DESC：新的在前
        # 加上 id 是为了在创建时间相同时也有确定的顺序，否则翻页时可能重复或漏掉数据
        .order_by(Document.created_at.desc(), Document.id.desc())
        .limit(limit)  # LIMIT 20：最多取 20 条
        .offset(offset)  # OFFSET 0：跳过前 0 条（翻到第二页时传 20）
    )
    # db.scalars(stmt) 执行查询，返回每行的第一列（这里就是 Document 对象）
    # list(...) 把结果转成普通列表
    return list(db.scalars(stmt))


# 按 id 查询单个文档。返回类型 Document | None：查到就返回文档，查不到返回 None
def get_document(db: Session, user_id: int, document_id: int) -> Document | None:
    """Returns None both for "doesn't exist" and "belongs to someone else".

    "不存在"和"属于别人"都返回 None，调用方无法区分这两种情况。
    这是故意的：如果对"属于别人"返回不同的错误，攻击者就能借此探测哪些 id 存在。
    """
    # WHERE id = ? AND user_id = ?：两个条件同时满足才返回
    # 拿到别人的 document_id 也查不到，因为 user_id 对不上
    stmt = select(Document).where(Document.id == document_id, Document.user_id == user_id)
    # db.scalar(stmt) 只取第一行第一列；没有结果时返回 None
    return db.scalar(stmt)


# 按内容哈希查某个用户的文档：上传时用来判断"这个用户是不是已经传过同样的文件"
def get_document_by_hash(db: Session, user_id: int, content_hash: str) -> Document | None:
    # WHERE user_id = ? AND content_hash = ?
    # 这两列正好有唯一约束 uq_documents_user_hash，查询会直接用上它的索引
    stmt = select(Document).where(Document.user_id == user_id, Document.content_hash == content_hash)
    return db.scalar(stmt)


# 按 id 查询某个用户的会话。不存在或属于别人，都返回 None（和 get_document 的理由一样）
def get_conversation(db: Session, user_id: int, conversation_id: int) -> Conversation | None:
    stmt = select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user_id)
    return db.scalar(stmt)


# 查询某个会话的全部消息
def list_messages(db: Session, user_id: int, conversation_id: int) -> list[Message]:
    """Messages of a conversation, oldest first. Empty if the conversation isn't the user's.

    返回会话的消息，最早的在前。如果会话不属于这个用户，返回空列表。
    """
    stmt = (
        select(Message)  # SELECT messages.*
        # messages 表里没有 user_id，所以要 JOIN（连接）conversations 表才能知道会话属于谁
        # JOIN conversations ON messages.conversation_id = conversations.id
        .join(Conversation, Message.conversation_id == Conversation.id)
        # WHERE conversations.id = ? AND conversations.user_id = ?（权限过滤）
        .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        # ORDER BY created_at, id：按时间从早到晚排，时间相同再按 id 排
        .order_by(Message.created_at, Message.id)
    )
    return list(db.scalars(stmt))


# @dataclass 会自动生成 __init__，所以可以直接写 ConversationSummary(会话, 消息)
# 这个类用来装"一个会话 + 它的最后一条消息"
@dataclass
class ConversationSummary:
    conversation: Conversation  # 会话对象
    last_message: Message | None  # 最后一条消息；空会话没有消息，所以可能是 None


# 查询会话列表，并带上每个会话的最后一条消息（类似聊天软件的会话列表页）
def list_conversations_with_last_message(
    db: Session, user_id: int, limit: int = 20
) -> list[ConversationSummary]:
    """One SQL query, not 1 + N.

    The naive version loops over conversations and runs one "latest message" query
    per conversation. Here a LATERAL subquery picks the latest message per
    conversation, using the (conversation_id, created_at) index for each lookup.

    只执行 1 条 SQL，而不是 1 + N 条。
    朴素写法：先查出 N 个会话（1 条 SQL），再循环每个会话各查一次最后一条消息（N 条 SQL）。
    这就是"N+1 问题"：会话越多，查询次数越多，越慢。
    这里用 LATERAL 子查询，在一条 SQL 里为每个会话取出最新的一条消息，
    每次查找都能用上 (conversation_id, created_at) 索引。
    """
    # 第一步：构造子查询"某个会话的最后一条消息"
    last_msg = (
        select(Message)  # SELECT messages.*
        # 关键：这里引用了外层查询的 Conversation.id。
        # 普通子查询不能引用外层查询的表，LATERAL 子查询可以。
        # 效果是：对外层的每一个会话，都单独执行一次这个子查询
        .where(Message.conversation_id == Conversation.id)
        # 按时间倒序，最新的在最前面
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(1)  # 只要第一条，也就是最新的那条
        .lateral("last_msg")  # 把它标记为 LATERAL 子查询，别名叫 last_msg
    )
    # 第二步：让子查询的结果能当作 Message 对象使用
    # 这样查询结果里拿到的是 Message 对象，而不是一堆零散的列
    last_msg_entity = aliased(Message, last_msg)
    # 第三步：构造主查询
    stmt = (
        # 每行结果包含两样东西：会话对象、它的最后一条消息
        select(Conversation, last_msg_entity)
        # LEFT OUTER JOIN LATERAL (...) ON true
        # 用 LEFT（左连接）而不是普通 JOIN：没有消息的会话也要出现在结果里，此时消息为 None
        # ON true：连接条件已经写在子查询的 where 里了，所以这里写"永远成立"
        .outerjoin(last_msg_entity, true())
        # WHERE conversations.user_id = ?（权限过滤）
        .where(Conversation.user_id == user_id)
        # 最近更新的会话排在前面
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .limit(limit)  # 最多取 limit 个会话
    )
    # db.execute(stmt) 返回多行，每行是 (会话, 消息) 两个值
    # 下面是"列表推导式"：遍历每一行，拆成 conv 和 msg，组装成 ConversationSummary
    return [ConversationSummary(conv, msg) for conv, msg in db.execute(stmt)]


# 统计某个用户从某个时间点开始做了多少次某种操作（用来算今日配额）
def count_actions_since(db: Session, user_id: int, action: UsageAction, since: datetime) -> int:
    """Used for daily quota. Stage 4 replaces the hot path with a Redis counter.

    用于每日配额。阶段 4 会把这个高频调用的地方换成 Redis 计数器，避免每次提问都查数据库。
    """
    # SELECT count(*) FROM usage_logs WHERE ...
    stmt = select(func.count()).where(
        UsageLog.user_id == user_id,  # 是这个用户
        UsageLog.action == action,  # 是这种操作（如 query）
        UsageLog.created_at >= since,  # 在 since 之后（如今天零点之后）
    )
    # 返回计数。"or 0" 是保险：万一结果是 None 就返回 0
    return db.scalar(stmt) or 0


# ============================ 检索 ============================


# 检索结果中的一段
@dataclass
class RetrievedChunk:
    chunk_id: int  # 段落 id
    document_id: int  # 所属文档 id
    filename: str  # 所属文档的文件名，展示来源时用
    content: str  # 段落内容，拼进给 LLM 的提示词里
    score: float | None  # 相似度，1 表示完全相同，越小越不相关；按顺序读取（不是检索）的段落没有相似度，为 None


# 向量检索：在这个用户"已处理完成"的文档里，找出和问题向量最相似的 top_k 段。
# 传了 document_id 就只在这一个文档里找
def search_chunks(
    db: Session, user_id: int, query_vector: list[float], top_k: int, document_id: int | None = None
) -> list[RetrievedChunk]:
    # 多用户场景下 HNSW 索引的一个坑：
    #   索引是所有用户共用的。默认情况下，索引只先找出全局最相似的 40 个段落，再按 WHERE 条件过滤。
    #   如果这 40 个都是别人的，过滤完一个都不剩，这个用户就搜不到任何结果，哪怕他的文档里有答案。
    # iterative_scan（pgvector 0.8 的新功能）：过滤后不够 top_k 个时，继续在索引里往下找，直到够数。
    # strict_order：保证结果严格按距离排序。
    # SET LOCAL：只在当前事务里生效，事务结束后自动恢复，不影响连接池里这个连接的下一次使用
    db.execute(text("SET LOCAL hnsw.iterative_scan = strict_order"))
    # 余弦距离：0 表示方向完全相同，越大越不相似。用 .label 给这个计算结果起个名字
    distance = Chunk.embedding.cosine_distance(query_vector).label("distance")
    stmt = (
        select(Chunk.id, Chunk.document_id, Document.filename, Chunk.content, distance)
        # chunks 表没有 user_id，要连接 documents 表才知道段落属于谁
        .join(Document, Chunk.document_id == Document.id)
        .where(
            Document.user_id == user_id,  # 权限过滤：只搜自己的文档
            Document.status == DocumentStatus.READY,  # 只搜处理完成的文档
            Chunk.embedding.is_not(None),  # 跳过没有向量的段落
        )
        # 按距离从小到大排序，也就是最相似的在前
        .order_by(distance)
        .limit(top_k)
    )
    # 限定文档：在上面的条件基础上再加一个条件（权限过滤仍然在，传别人的文档 id 什么也查不到）
    if document_id is not None:
        stmt = stmt.where(Chunk.document_id == document_id)
    return [
        # 相似度 = 1 - 余弦距离，更直观：越大越相似
        RetrievedChunk(chunk_id, document_id, filename, content, 1 - dist)
        for chunk_id, document_id, filename, content, dist in db.execute(stmt)
    ]


# 关键词检索：找出包含问题里的词的段落，按 BM25 的思路打分。
# 和向量检索互补：向量擅长"意思相近、用词不同"，但对型号、缩写、专有名词（"E171"、"8218"）这种
# 没什么"意思"可言的词很弱，这些词只能按字面匹配。
#
# 打分：每个词的得分 = IDF × 这个词在段落里的词频得分，所有词加起来。
#   IDF（逆文档频率）：越少的段落包含这个词，这个词越重要。"老板8218怎么样"里，"怎么样"到处都有，
#     几乎没有区分度；"8218"只出现在极少数段落里，命中它才说明找对了。
#   词频得分：用 PostgreSQL 自带的 ts_rank 对单个词打分。出现次数越多分越高，但增长越来越慢
#     （出现 10 次不会是 1 次的 10 倍）；参数 1 表示再除以段落长度的对数，长段落不会因为字多而占便宜。
# 为什么不直接用 ts_rank 给整个问题打分：它只看词出现了几次，没有 IDF，常见词会把关键的词淹没。
# 在 DuRetrieval 上离线实测：只数次数 Recall@5 约 27%，加上 IDF 约 61%，和标准 BM25 公式（60.5%）基本一样。
#
# 也传入问题向量：返回的每段都带上余弦相似度，和向量检索的结果统一，调用方才能用同一条最低分数线过滤
def keyword_search_chunks(
    db: Session, user_id: int, query: str, query_vector: list[float], top_k: int, document_id: int | None = None
) -> list[RetrievedChunk]:
    # dict.fromkeys：去掉重复的词，保留顺序（同一个词出现两次不重复计分）
    words = list(dict.fromkeys(segment(query)))
    # 问题里没有一个词（比如只有标点）：没法做关键词检索
    if not words:
        return []
    # 每个词一个检索条件。词都是 \w 组成的（见 segment），不会含有 & | ! 这类在检索语法里有特殊含义的符号
    word_queries = [func.to_tsquery("simple", word) for word in words]
    # 检索的范围：这个用户已处理完成的文档（传了 document_id 就只是这一个文档）。
    # 下面统计 IDF 和真正检索都在这个范围里，所以单独写成一个函数
    def in_scope(stmt):
        stmt = stmt.join(Document, Chunk.document_id == Document.id).where(
            Document.user_id == user_id,  # 权限过滤：只搜自己的文档
            Document.status == DocumentStatus.READY,  # 只搜处理完成的文档
            Chunk.embedding.is_not(None),  # 和向量检索一样跳过没有向量的段落（要计算相似度）
        )
        return stmt if document_id is None else stmt.where(Chunk.document_id == document_id)

    # 第一条 SQL：一共多少段，以及每个词出现在多少段里（文档频率 df）。
    # count().filter(条件)：只数满足条件的行，一次扫描就能把每个词的 df 都数出来。
    # @@：全文检索的"匹配"运算符，terms 里包含这个词才算匹配
    total, *dfs = db.execute(
        in_scope(select(func.count(), *[func.count().filter(Chunk.terms.op("@@")(q)) for q in word_queries]).select_from(Chunk))
    ).one()
    # IDF，和 BM25 用的公式一样：包含这个词的段落越少，值越大；几乎每段都有的词接近 0
    idfs = [math.log(1 + (total - df + 0.5) / (df + 0.5)) for df in dfs]
    # 每段的得分：所有词的 IDF × 词频得分 之和
    score = sum(idf * func.ts_rank(Chunk.terms, q, 1) for idf, q in zip(idfs, word_queries, strict=True))
    # 检索条件：包含任意一个词就算匹配（| 表示"或"），例如 "路由 | 路由器 | 报 | e171"。
    # 用"或"而不是"且"：问题里的"怎么"、"什么"这类词段落里通常没有，要求全部包含的话几乎什么都搜不到。
    # 这一步会用上 GIN 索引
    any_word = func.to_tsquery("simple", " | ".join(words))
    distance = Chunk.embedding.cosine_distance(query_vector)
    stmt = (
        in_scope(select(Chunk.id, Chunk.document_id, Document.filename, Chunk.content, distance).select_from(Chunk))
        .where(Chunk.terms.op("@@")(any_word))
        # 得分从高到低
        .order_by(score.desc())
        .limit(top_k)
    )
    return [
        RetrievedChunk(chunk_id, document_id, filename, content, 1 - dist)
        for chunk_id, document_id, filename, content, dist in db.execute(stmt)
    ]


# RRF（Reciprocal Rank Fusion，倒数排名融合）：把几个排好序的列表合并成一个。
# 每段的得分 = 它在每个列表里的 1 / (RRF_K + 名次) 之和，名次从 1 开始。
# 只看名次，不看原来的分数：向量的相似度（0-1）和关键词的 ts_rank 量纲完全不同，没法直接相加，名次可以。
# 两个列表都排得靠前的段落得分最高；只出现在一个列表里的，也能凭名次排进来
RRF_K = 60


def fuse(result_lists: list[list[RetrievedChunk]], top_k: int) -> list[RetrievedChunk]:
    # {段落 id: 累计得分}
    scores: dict[int, float] = {}
    # {段落 id: 段落}，同一段在两个列表里都出现时，内容一样，留哪个都行
    chunks: dict[int, RetrievedChunk] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1 / (RRF_K + rank)
            chunks[chunk.chunk_id] = chunk
    # 按得分从高到低排，取前 top_k 段
    best = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return [chunks[chunk_id] for chunk_id in best]


# 混合检索每一路取多少段候选。比最终要的多取一些：某段在一路里排第 8，在另一路里排第 2，
# 融合后可能排进前 5，只各取 5 段的话它在第一路里就被截掉了
HYBRID_CANDIDATES = 20


# 混合检索：向量检索和关键词检索各取一批候选，用 RRF 合并，返回前 top_k 段。
# 返回的 score 仍然是余弦相似度（不是 RRF 得分），所以调用方原来的最低分数线照样可以用
def hybrid_search_chunks(
    db: Session, user_id: int, query: str, query_vector: list[float], top_k: int, document_id: int | None = None
) -> list[RetrievedChunk]:
    candidates = max(top_k, HYBRID_CANDIDATES)
    return fuse(
        [
            search_chunks(db, user_id, query_vector, candidates, document_id),
            keyword_search_chunks(db, user_id, query, query_vector, candidates, document_id),
        ],
        top_k,
    )


# 按顺序读取某个文档的前 limit 段（"通读"一个文档，不是检索）
def list_chunks(db: Session, user_id: int, document_id: int, limit: int) -> list[RetrievedChunk]:
    stmt = (
        select(Chunk.id, Chunk.document_id, Document.filename, Chunk.content)
        .join(Document, Chunk.document_id == Document.id)
        .where(
            Chunk.document_id == document_id,
            Document.user_id == user_id,  # 权限过滤：只能读自己的文档
            Document.status == DocumentStatus.READY,  # 只读处理完成的文档
        )
        # 按段落在文档里的顺序
        .order_by(Chunk.chunk_index)
        .limit(limit)
    )
    # 不是检索出来的，没有相似度
    return [RetrievedChunk(chunk_id, doc_id, filename, content, None) for chunk_id, doc_id, filename, content in db.execute(stmt)]
