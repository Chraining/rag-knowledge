"""Data model. Comments explain the reasoning behind FKs and indexes.

数据模型：定义六张表。每个类对应数据库里的一张表，类里的每个字段对应一列。
这种"用 Python 类描述数据库表"的方式叫 ORM（对象关系映射）。
"""

# enum：Python 标准库，用来定义"只能取几个固定值"的类型（枚举）
import enum
# datetime：日期时间类型，这里只用来做类型标注
from datetime import datetime

# 从 SQLAlchemy 导入各种"列类型"和"约束"
from sqlalchemy import (
    BigInteger,  # 大整数（8 字节），范围比 Integer 大得多
    CheckConstraint,  # CHECK 约束：插入/更新时，数据库检查某个条件必须成立
    DateTime,  # 日期时间类型
    Enum,  # 枚举类型：限制一列只能取几个固定值
    ForeignKey,  # 外键：表示"这一列引用另一张表的主键"
    Index,  # 索引：加快查询的数据结构，类似书的目录
    Integer,  # 普通整数（4 字节）
    String,  # 有长度上限的字符串，对应 SQL 的 VARCHAR(n)
    Text,  # 不限长度的长文本
    UniqueConstraint,  # 唯一约束：某一列（或几列组合）的值不能重复
    func,  # 用来调用数据库里的函数，例如 func.now() 就是 SQL 的 now()
)
# Vector：pgvector 扩展提供的向量列类型，用来存 embedding（一串浮点数）
from pgvector.sqlalchemy import Vector
# JSONB：PostgreSQL 特有的 JSON 类型（二进制存储，可以建索引、可以查询内部字段）
from sqlalchemy.dialects.postgresql import JSONB
# DeclarativeBase：所有模型类的基类
# Mapped：类型标注，表示"这个属性映射到数据库的一列"
# mapped_column：定义一列的具体细节（类型、约束、默认值等）
# relationship：定义表与表之间的关系，方便在 Python 里通过属性互相访问
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# 导入配置，用来拿默认每日配额
from app.config import settings


# 向量的维度：每段文字被 embedding 模型转换成 1024 个浮点数。
# 必须和 embedding 模型输出的维度一致（text-embedding-v4 调用时指定 dimensions=1024）
EMBEDDING_DIM = 1024


# 所有模型的基类。继承它的类会被 SQLAlchemy 识别为一张表。
# Base.metadata 里会收集所有表的定义，Alembic 生成迁移时就是读它。
class Base(DeclarativeBase):
    pass  # 不需要额外内容，pass 表示"空的类体"


# 文档的处理状态（状态机）：pending(等待) -> processing(处理中) -> ready(完成) / failed(失败)
# StrEnum 的好处：每个成员本身就是字符串，DocumentStatus.READY == "ready" 成立
class DocumentStatus(enum.StrEnum):
    PENDING = "pending"  # 刚上传，等待处理
    PROCESSING = "processing"  # 后台正在解析、切分、向量化
    READY = "ready"  # 处理成功，可以用来问答
    FAILED = "failed"  # 处理失败，原因记录在 error_message


# 消息的角色：是用户说的，还是 AI 助手说的
class MessageRole(enum.StrEnum):
    USER = "user"  # 用户的提问
    ASSISTANT = "assistant"  # AI 的回答


# 使用记录的动作类型：用来统计配额
class UsageAction(enum.StrEnum):
    UPLOAD = "upload"  # 上传了一个文档
    QUERY = "query"  # 提了一个问题


# 辅助函数：把 Python 枚举转换成数据库列类型。前面的下划线表示"仅本文件内部使用"。
# 参数 enum_cls 是枚举类（如 DocumentStatus），name 是约束名
def _str_enum(enum_cls: type[enum.Enum], name: str) -> Enum:
    # VARCHAR + CHECK instead of a native PG enum: adding a value later is a
    # simple constraint change rather than ALTER TYPE.
    # 不用 PostgreSQL 原生的 ENUM 类型，而是用"普通字符串列 + CHECK 约束"。
    # 原因：以后要增加一个新状态时，改 CHECK 约束很简单；改原生 ENUM 要用 ALTER TYPE，比较麻烦。
    return Enum(
        enum_cls,  # 用哪个 Python 枚举
        name=name,  # 生成的 CHECK 约束叫什么名字
        native_enum=False,  # 不用数据库原生 ENUM，用 VARCHAR
        create_constraint=True,  # 自动加 CHECK 约束，防止存入非法值
        length=20,  # VARCHAR(20)，最长 20 个字符
        # 存进数据库的是枚举的"值"（如 "pending"），而不是"名字"（如 "PENDING"）
        # lambda e: [...] 是一个匿名函数：输入枚举类 e，返回它所有成员的 value 列表
        values_callable=lambda e: [m.value for m in e],
    )


# ============================ users 用户表 ============================
class User(Base):
    __tablename__ = "users"  # 数据库里的表名

    # 主键：每行的唯一编号，数据库自动递增生成（1, 2, 3, ...）
    # Mapped[int] 表示这一列在 Python 里是 int 类型，且不能为空（没写 | None）
    id: Mapped[int] = mapped_column(primary_key=True)
    # 用户名，最长 50 字符。unique=True 表示不能重复（数据库会自动为它建唯一索引）
    username: Mapped[str] = mapped_column(String(50), unique=True)
    # 邮箱，同样不能重复
    email: Mapped[str] = mapped_column(String(255), unique=True)
    # 密码的哈希值。绝不存明文密码：数据库泄露时，攻击者也拿不到原始密码
    password_hash: Mapped[str] = mapped_column(String(255))
    # 每日提问上限
    daily_quota: Mapped[int] = mapped_column(
        # default：用 Python 创建对象时的默认值
        # server_default：数据库层面的默认值（直接写 SQL 插入时也生效），必须是字符串
        Integer, default=settings.default_daily_quota, server_default=str(settings.default_daily_quota)
    )
    # 账号是否有效。用于"软删除"：注销用户时把它设成 False，而不是真的删掉这一行
    is_active: Mapped[bool] = mapped_column(default=True, server_default="true")
    # 创建时间。timezone=True 表示带时区（PostgreSQL 的 timestamptz 类型）
    # server_default=func.now()：插入时由数据库自动填当前时间
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # 下面两行不是数据库的列，而是 ORM 的"关系"，只存在于 Python 里
    # user.documents 可以拿到这个用户的所有文档（会自动执行 SQL 查询）
    # back_populates="user" 表示：它和 Document 类里的 user 属性是同一个关系的两端
    documents: Mapped[list["Document"]] = relationship(back_populates="user")
    # user.conversations 可以拿到这个用户的所有会话
    conversations: Mapped[list["Conversation"]] = relationship(back_populates="user")


# ============================ documents 文档表 ============================
class Document(Base):
    __tablename__ = "documents"
    # __table_args__ 用来定义"涉及多列"的约束和索引，必须是一个元组（tuple）
    __table_args__ = (
        # Same user uploading the same file twice -> reuse the existing row (idempotency).
        # 唯一约束：(user_id, content_hash) 这对组合不能重复
        # 意思是：同一个用户不能上传两份内容完全相同的文件（用于去重和幂等）
        # 但不同用户上传同一份文件是允许的
        UniqueConstraint("user_id", "content_hash", name="uq_documents_user_hash"),
        # Leading user_id covers "WHERE user_id = ?" and also serves ORDER BY created_at.
        # 复合索引 (user_id, created_at)：先按 user_id 排，同一用户内再按创建时间排
        # 能加速"查某个用户的文档，按时间倒序"这个最常用的查询，而且不需要额外排序
        Index("ix_documents_user_created", "user_id", "created_at"),
        # CHECK 约束：文件大小不能是负数
        CheckConstraint("file_size >= 0", name="ck_documents_file_size"),
    )

    # 主键
    id: Mapped[int] = mapped_column(primary_key=True)
    # RESTRICT: users are soft-deleted; a hard delete must clean up files on disk first,
    # which a DB cascade can't do.
    # 外键：指向 users 表的 id，表示"这个文档属于哪个用户"
    # ondelete="RESTRICT"：如果这个用户还有文档，数据库禁止删除该用户
    # 原因：用户是软删除的；真要物理删除时，必须先删掉磁盘上的文件，而数据库的级联删除做不到这一点
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"))
    # 用户上传时的原始文件名，例如 "我的笔记.md"
    filename: Mapped[str] = mapped_column(String(255))
    # 文件在服务器上实际保存的路径
    storage_path: Mapped[str] = mapped_column(String(500))
    # 文件大小（字节）。用 BigInteger 是为了能表示超过 2GB 的数字
    file_size: Mapped[int] = mapped_column(BigInteger)
    # 文件内容的 SHA-256 哈希，64 个十六进制字符。内容相同 -> 哈希一定相同
    content_hash: Mapped[str] = mapped_column(String(64))  # sha256 hex
    # 处理状态，只能是 DocumentStatus 里的四个值之一
    status: Mapped[DocumentStatus] = mapped_column(
        _str_enum(DocumentStatus, "document_status"),  # 列类型：VARCHAR(20) + CHECK 约束
        default=DocumentStatus.PENDING,  # Python 创建对象时默认是 pending
        server_default=DocumentStatus.PENDING.value,  # 数据库默认值也是 'pending'
    )
    # 文档被切成了多少段
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # 失败原因。str | None 表示"可以为空"（数据库里是 NULL）
    error_message: Mapped[str | None] = mapped_column(Text)
    # 上传时间
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # 处理完成的时间，还没处理完时为空
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # 关系：document.user 拿到文档所属的用户对象
    user: Mapped[User] = relationship(back_populates="documents")
    # 关系：document.chunks 拿到这个文档的所有段落
    chunks: Mapped[list["Chunk"]] = relationship(
        # cascade="all, delete-orphan"：在 Python 里删除文档时，它的 chunks 也一起删
        # passive_deletes=True：删除时不先把 chunks 查出来逐个删，而是交给数据库的 ON DELETE CASCADE，更快
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


# ============================ chunks 文本段表 ============================
# 一个文档会被切成很多小段（chunk），问答时检索的是这些小段
class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        # Re-processing a document can't create duplicate chunks; also serves
        # "WHERE document_id = ?" so no separate document_id index is needed.
        # 唯一约束 (document_id, chunk_index)：同一个文档里，"第几段"不能重复
        # 作用 1：文档被重复处理时，不会产生重复的段落（幂等）
        # 作用 2：唯一约束会自带一个索引，且以 document_id 开头，
        #        所以"查某文档的所有段"也能用上它，不需要再单独给 document_id 建索引
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
        # HNSW 向量索引：加速"找出和问题向量最相似的段落"。
        # 没有它，每次提问都要和所有段落逐个计算距离（全表扫描）；
        # 有了它，像查地图一样只看"附近"的点，代价是结果是近似的（偶尔漏掉一个真正最近的）。
        # vector_cosine_ops：按余弦距离建索引，必须和查询里用的距离函数（cosine_distance）一致，否则用不上索引
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    # 主键
    id: Mapped[int] = mapped_column(primary_key=True)
    # CASCADE: chunks are derived data with no meaning without their document.
    # 外键：指向 documents 表
    # ondelete="CASCADE"：删除文档时，数据库自动删掉它的所有段落
    # 原因：段落是从文档"派生"出来的数据，文档没了，段落也就没有意义了
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    # 这是文档的第几段（从 0 开始）
    chunk_index: Mapped[int] = mapped_column(Integer)
    # 这一段的文本内容
    content: Mapped[str] = mapped_column(Text)
    # 这一段文字的向量（embedding）。语义相近的文字，向量之间的距离也近。
    # 允许为空的原因：给一张已经有数据的表加新列时，旧的行没有值。
    # 正常流程里，worker 写入段落时一定会同时写入向量
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))

    # 关系：chunk.document 拿到所属文档
    document: Mapped[Document] = relationship(back_populates="chunks")


# ============================ conversations 会话表 ============================
# 一个会话 = 一次多轮对话，里面包含很多条消息
class Conversation(Base):
    __tablename__ = "conversations"
    # 复合索引 (user_id, updated_at)：加速"查某用户的会话列表，按最近更新时间排序"
    # 注意结尾的逗号：只有一个元素的元组必须写成 (x,)，否则 Python 会把括号当成普通括号
    __table_args__ = (Index("ix_conversations_user_updated", "user_id", "updated_at"),)

    # 主键
    id: Mapped[int] = mapped_column(primary_key=True)
    # 外键：会话属于哪个用户。同样用 RESTRICT，理由和 documents 一样
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"))
    # 会话标题（计划从第一个问题自动生成）
    title: Mapped[str] = mapped_column(String(200))
    # 创建时间
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    # 最后更新时间
    updated_at: Mapped[datetime] = mapped_column(
        # onupdate=func.now()：通过 ORM 更新这一行时，自动把它改成当前时间
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    # 关系：conversation.user 拿到所属用户
    user: Mapped[User] = relationship(back_populates="conversations")
    # 关系：conversation.messages 拿到这个会话的所有消息
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",  # 删会话时，消息一起删
        passive_deletes=True,  # 交给数据库的 ON DELETE CASCADE 去删
        order_by="Message.created_at",  # 取出来的消息按时间从早到晚排列
    )


# ============================ messages 消息表 ============================
class Message(Base):
    __tablename__ = "messages"
    # 复合索引 (conversation_id, created_at)：加速"按时间顺序取某个会话的消息"
    # 这是规范里要求的联合索引
    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)

    # 主键
    id: Mapped[int] = mapped_column(primary_key=True)
    # 外键：消息属于哪个会话。CASCADE：会话删了，消息跟着删
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"))
    # 角色：user 或 assistant
    role: Mapped[MessageRole] = mapped_column(_str_enum(MessageRole, "message_role"))
    # 消息内容
    content: Mapped[str] = mapped_column(Text)
    # [{"chunk_id": 1, "document_id": 2, "score": 0.83}, ...]
    # 引用来源：AI 回答时参考了哪些段落，用 JSON 数组存，格式如上一行所示
    # 用户的消息没有来源，所以可以为空
    sources: Mapped[list | None] = mapped_column(JSONB)
    # 这条消息消耗了多少 token（LLM 按 token 计费）
    tokens_used: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # clock_timestamp(), not now(): now() is fixed per transaction, so a question and
    # its answer saved together would tie and their order would be undefined.
    # 这里用 clock_timestamp() 而不是 now()：
    # now() 在同一个事务里永远返回同一个时间（事务开始的时间）。
    # 如果提问和回答在一个事务里一起保存，两条消息时间完全相同，就分不出先后了。
    # clock_timestamp() 返回的是真实的当前时间，每次调用都不一样。
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.clock_timestamp()
    )

    # 关系：message.conversation 拿到所属会话
    conversation: Mapped[Conversation] = relationship(back_populates="messages")


# ============================ usage_logs 使用记录表 ============================
# 每上传一次、每提问一次就记一行，用来统计配额和用量
class UsageLog(Base):
    __tablename__ = "usage_logs"
    __table_args__ = (
        # Matches the quota query exactly: WHERE user_id=? AND action=? AND created_at >= ?
        # 三列复合索引，正好对应配额查询的条件：
        #   WHERE user_id = ? AND action = ? AND created_at >= 今天零点
        # 查询只读索引就能数出结果，完全不用读表（EXPLAIN 里显示为 Index Only Scan）
        Index("ix_usage_logs_user_action_created", "user_id", "action", "created_at"),
    )

    # 主键。用 BigInteger：这张表增长最快，每次操作都记一行，可能超过 21 亿（Integer 上限）
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    # 外键：是哪个用户的操作
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"))
    # 动作类型：upload 或 query
    action: Mapped[UsageAction] = mapped_column(_str_enum(UsageAction, "usage_action"))
    # 这次操作用了多少 token
    tokens_used: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    # 操作时间
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
