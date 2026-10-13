# 测试 app/queries.py 里的查询函数：权限隔离、N+1、配额计数

# hashlib：计算哈希
import hashlib
# UTC：UTC 时区；datetime：日期时间；timedelta：时间间隔（如"1 分钟"）
from datetime import UTC, datetime, timedelta

# event：SQLAlchemy 的事件系统，可以"监听"数据库操作（这里用来数 SQL 执行了几次）
from sqlalchemy import event

# 导入被测试的模块
from app import queries
# 导入模型
from app.models import Conversation, Document, Message, MessageRole, UsageAction, UsageLog


# 辅助函数：创建一个文档
def make_doc(db, user, name="a.md") -> Document:
    doc = Document(
        user_id=user.id,
        filename=name,
        storage_path=f"storage/{user.id}/{name}",
        file_size=1,
        # 用"用户id/文件名"计算哈希，保证每个文档的哈希都不同，不会触发唯一约束
        # .encode() 把字符串转成字节，因为哈希函数只接受字节
        content_hash=hashlib.sha256(f"{user.id}/{name}".encode()).hexdigest(),
    )
    db.add(doc)
    db.flush()
    return doc


# 辅助函数：创建一个会话，并在里面放 n_messages 条消息
def make_conv(db, user, n_messages: int) -> Conversation:
    conv = Conversation(user_id=user.id, title="t")
    db.add(conv)
    db.flush()  # 拿到 conv.id
    # range(n) 产生 0, 1, ..., n-1
    for i in range(n_messages):
        # 三元表达式：i 是偶数时为 USER，奇数时为 ASSISTANT（模拟一问一答）
        # % 是取余数，i % 2 == 0 表示 i 是偶数
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        # 消息内容依次是 m0, m1, m2, ...
        db.add(Message(conversation_id=conv.id, role=role, content=f"m{i}"))
        # 每条消息单独 flush，让它们的创建时间（clock_timestamp）有先后之分
        db.flush()
    return conv


# 一个工具类：统计一段代码里执行了多少条 SQL
# 用法：with CountQueries(连接) as counter: ...  结束后看 counter.count
class CountQueries:
    # 构造函数：保存连接，计数从 0 开始
    def __init__(self, conn):
        self.conn = conn
        self.count = 0

    # 每执行一条 SQL，这个方法就会被调用一次，计数加 1
    # *args 表示接收任意数量的位置参数（事件会传很多参数进来，我们用不到）
    def _inc(self, *args):
        self.count += 1

    # 进入 with 块时调用：开始监听
    def __enter__(self):
        # "before_cursor_execute"：每条 SQL 发送到数据库之前触发这个事件
        event.listen(self.conn, "before_cursor_execute", self._inc)
        return self  # 返回值就是 "as counter" 里的 counter

    # 离开 with 块时调用：停止监听
    def __exit__(self, *exc):
        event.remove(self.conn, "before_cursor_execute", self._inc)


# --- isolation: user A must never see user B's data -------------------------
# --- 权限隔离：用户 A 永远不能看到用户 B 的数据 ---


# 测试：文档列表只返回自己的文档
def test_list_documents_only_returns_own(db, make_user):
    # 一次创建两个用户：alice 和 bob
    alice, bob = make_user(), make_user()
    make_doc(db, alice, "a1.md")  # alice 的两个文档
    make_doc(db, alice, "a2.md")
    make_doc(db, bob, "b1.md")  # bob 的一个文档

    docs = queries.list_documents(db, alice.id)
    # {d.filename for d in docs} 是"集合推导式"，得到文件名的集合（集合不关心顺序）
    # 结果应该只有 alice 的两个文档，没有 bob 的
    assert {d.filename for d in docs} == {"a1.md", "a2.md"}


# 测试：拿着别人的文档 id 也查不到
def test_get_document_of_other_user_returns_none(db, make_user):
    alice, bob = make_user(), make_user()
    bobs_doc = make_doc(db, bob)

    # alice 用 bob 的文档 id 去查 -> None（这就是越权测试）
    assert queries.get_document(db, alice.id, bobs_doc.id) is None
    # bob 自己查 -> 能查到
    assert queries.get_document(db, bob.id, bobs_doc.id).id == bobs_doc.id


# 测试：按哈希查文档也是按用户隔离的
def test_get_document_by_hash_is_per_user(db, make_user):
    alice, bob = make_user(), make_user()
    doc = make_doc(db, alice)

    # alice 能查到自己的
    assert queries.get_document_by_hash(db, alice.id, doc.content_hash).id == doc.id
    # bob 用同样的哈希查不到 alice 的文档
    assert queries.get_document_by_hash(db, bob.id, doc.content_hash) is None


# 测试：查别人会话的消息，结果是空的
def test_list_messages_of_other_users_conversation_is_empty(db, make_user):
    alice, bob = make_user(), make_user()
    conv = make_conv(db, bob, 3)  # bob 的会话，有 3 条消息

    # alice 查 bob 的会话 -> 空列表
    assert queries.list_messages(db, alice.id, conv.id) == []
    # bob 查自己的 -> 3 条消息，并且按时间顺序排列
    assert [m.content for m in queries.list_messages(db, bob.id, conv.id)] == ["m0", "m1", "m2"]


# 测试：会话列表只返回自己的会话
def test_list_conversations_only_returns_own(db, make_user):
    alice, bob = make_user(), make_user()
    make_conv(db, alice, 1)
    make_conv(db, bob, 1)

    result = queries.list_conversations_with_last_message(db, alice.id)
    # 结果里所有会话的 user_id 都应该是 alice，而且只有 1 个
    assert [s.conversation.user_id for s in result] == [alice.id]


# --- N+1 ---------------------------------------------------------------------
# --- N+1 问题 ---


# 测试：查会话列表（带最后一条消息）只执行 1 条 SQL，而且结果正确
def test_conversations_with_last_message_is_one_query(db, make_user):
    user = make_user()
    # 创建 4 个会话，分别有 3、1、0、5 条消息（包括"没有消息"这种边界情况）
    convs = [make_conv(db, user, n) for n in (3, 1, 0, 5)]

    # 在 with 块里统计执行了几条 SQL
    with CountQueries(db.connection()) as counter:
        result = queries.list_conversations_with_last_message(db, user.id)

    # 4 个会话，只执行了 1 条 SQL（如果有 N+1 问题，这里会是 5）
    assert counter.count == 1
    # 字典推导式：{会话id: 最后一条消息的内容}；没有消息的会话对应 None
    last = {s.conversation.id: (s.last_message.content if s.last_message else None) for s in result}
    # 3 条消息 -> 最后是 m2；1 条 -> m0；0 条 -> None；5 条 -> m4
    assert last == {convs[0].id: "m2", convs[1].id: "m0", convs[2].id: None, convs[3].id: "m4"}


# --- quota -------------------------------------------------------------------
# --- 配额计数 ---


# 测试：只统计"指定用户 + 指定操作 + 指定时间之后"的记录
def test_count_actions_since_counts_only_window_and_action(db, make_user):
    alice, bob = make_user(), make_user()
    # 今天零点（UTC）：把当前时间的时、分、秒、微秒都设成 0
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    db.add_all(
        [
            # 算：alice 今天的 query
            UsageLog(user_id=alice.id, action=UsageAction.QUERY, created_at=today + timedelta(minutes=1)),
            # 算：alice 今天的 query
            UsageLog(user_id=alice.id, action=UsageAction.QUERY, created_at=today + timedelta(minutes=2)),
            # 不算：昨天的（零点前 1 分钟）
            UsageLog(user_id=alice.id, action=UsageAction.QUERY, created_at=today - timedelta(minutes=1)),
            # 不算：是 upload 不是 query
            UsageLog(user_id=alice.id, action=UsageAction.UPLOAD, created_at=today + timedelta(minutes=1)),
            # 不算：是 bob 的
            UsageLog(user_id=bob.id, action=UsageAction.QUERY, created_at=today + timedelta(minutes=1)),
        ]
    )
    db.flush()

    # 只有前两条符合条件
    assert queries.count_actions_since(db, alice.id, UsageAction.QUERY, today) == 2
