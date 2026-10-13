"""Constraint behaviour: these are enforced by the database, not by app code.

测试数据库约束的行为：这些规则是由数据库强制执行的，而不是靠应用代码检查。
好处是：即使以后有人写错代码、或直接用 SQL 改数据，数据库也会拦住非法数据。
"""

# hashlib：Python 标准库，计算哈希值（这里用 SHA-256）
import hashlib

# pytest：测试框架
import pytest
# delete：构造 DELETE 语句；func：SQL 函数；select：构造 SELECT；text：原生 SQL
from sqlalchemy import delete, func, select, text
# IntegrityError：违反数据库约束（唯一、外键、CHECK 等）时抛出的异常
from sqlalchemy.exc import IntegrityError

# 导入要测试的模型
from app.models import Chunk, Conversation, Document, DocumentStatus, Message, MessageRole, User


# 辅助函数：创建一个文档。不是测试（名字不以 test_ 开头，pytest 不会把它当测试运行）
# content=b"hello"：b 开头表示字节串（bytes），文件内容本来就是字节
def make_doc(db, user, name="a.md", content=b"hello") -> Document:
    doc = Document(
        user_id=user.id,  # 属于哪个用户
        filename=name,  # 文件名
        storage_path=f"storage/{user.id}/{name}",  # f-string：花括号里的变量会被替换成实际值
        file_size=len(content),  # 字节数
        # 计算内容的 SHA-256 哈希，hexdigest() 转成 64 位十六进制字符串
        content_hash=hashlib.sha256(content).hexdigest(),
    )
    db.add(doc)  # 加入会话
    db.flush()  # 发送 INSERT 到数据库，拿到 id
    return doc


# 测试：新文档的默认状态应该是 pending，段数是 0
def test_new_document_defaults_to_pending(db, make_user):
    doc = make_doc(db, make_user())
    # refresh：从数据库重新读取这一行，确保拿到的是数据库里的真实值
    db.refresh(doc)
    # assert：断言。条件不成立时测试失败
    assert doc.status == DocumentStatus.PENDING
    assert doc.chunk_count == 0


# 测试：非法的状态值会被数据库拒绝（CHECK 约束在起作用）
def test_invalid_status_is_rejected(db, make_user):
    doc = make_doc(db, make_user())
    # pytest.raises：断言 with 块里的代码"一定会"抛出 IntegrityError；没抛出就算测试失败
    with pytest.raises(IntegrityError):
        # 故意用原生 SQL 绕过 Python 的枚举检查，直接测数据库的 CHECK 约束
        # :id 是参数占位符，后面的字典提供它的值（防止 SQL 注入的正确写法）
        db.execute(text("UPDATE documents SET status = 'weird' WHERE id = :id"), {"id": doc.id})


# 测试：同一个用户上传相同内容的文件，会被唯一约束拒绝
def test_same_user_same_content_is_rejected(db, make_user):
    user = make_user()
    make_doc(db, user, "a.md", b"same")
    with pytest.raises(IntegrityError):
        # 文件名不同但内容相同 -> 哈希相同 -> 违反 (user_id, content_hash) 唯一约束
        make_doc(db, user, "renamed.md", b"same")


# 测试：不同用户上传相同内容是允许的
def test_different_users_may_upload_same_content(db, make_user):
    make_doc(db, make_user(), "a.md", b"same")  # 用户 1 上传
    make_doc(db, make_user(), "a.md", b"same")  # no error  用户 2 上传同样内容，不报错


# 测试：删除文档时，它的段落会被数据库自动删除（ON DELETE CASCADE）
def test_deleting_document_deletes_its_chunks(db, make_user):
    doc = make_doc(db, make_user())
    # 列表推导式：一次创建 3 个段落（chunk_index 为 0、1、2）
    # add_all：一次加入多个对象
    db.add_all([Chunk(document_id=doc.id, chunk_index=i, content="x") for i in range(3)])
    db.flush()

    # Core DELETE bypasses ORM cascades, so this proves the DB-level ON DELETE CASCADE.
    # 这里直接执行 DELETE 语句，绕过了 ORM 的级联删除逻辑。
    # 所以如果段落被删掉了，就证明是数据库层面的 ON DELETE CASCADE 在起作用
    db.execute(delete(Document).where(Document.id == doc.id))
    # SELECT count(*) FROM chunks：段落数应该是 0
    assert db.scalar(select(func.count()).select_from(Chunk)) == 0


# 测试：同一个文档里不能有两个相同序号的段落
def test_duplicate_chunk_index_is_rejected(db, make_user):
    doc = make_doc(db, make_user())
    db.add(Chunk(document_id=doc.id, chunk_index=0, content="x"))  # 第 0 段
    db.flush()
    db.add(Chunk(document_id=doc.id, chunk_index=0, content="y"))  # 又一个第 0 段
    with pytest.raises(IntegrityError):
        db.flush()  # 写入数据库时才会触发约束检查，所以异常在 flush 时抛出


# 测试：用户还有文档时，不能物理删除该用户（ON DELETE RESTRICT）
def test_user_with_documents_cannot_be_hard_deleted(db, make_user):
    user = make_user()
    make_doc(db, user)
    with pytest.raises(IntegrityError):
        # DELETE FROM users WHERE id = ?  -> 被外键约束拦住
        db.execute(delete(User).where(User.id == user.id))


# 测试：删除会话时，它的消息会被自动删除
def test_deleting_conversation_deletes_messages(db, make_user):
    user = make_user()
    conv = Conversation(user_id=user.id, title="t")  # 创建一个会话
    db.add(conv)
    db.flush()
    # 在会话里加一条消息
    db.add(Message(conversation_id=conv.id, role=MessageRole.USER, content="hi"))
    db.flush()

    # 删除会话
    db.execute(delete(Conversation).where(Conversation.id == conv.id))
    # 消息应该也没了
    assert db.scalar(select(func.count()).select_from(Message)) == 0
