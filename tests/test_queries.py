import hashlib
from datetime import UTC, datetime, timedelta

from sqlalchemy import event

from app import queries
from app.models import Conversation, Document, Message, MessageRole, UsageAction, UsageLog


def make_doc(db, user, name="a.md") -> Document:
    doc = Document(
        user_id=user.id,
        filename=name,
        storage_path=f"storage/{user.id}/{name}",
        file_size=1,
        content_hash=hashlib.sha256(f"{user.id}/{name}".encode()).hexdigest(),
    )
    db.add(doc)
    db.flush()
    return doc


def make_conv(db, user, n_messages: int) -> Conversation:
    conv = Conversation(user_id=user.id, title="t")
    db.add(conv)
    db.flush()
    for i in range(n_messages):
        role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        db.add(Message(conversation_id=conv.id, role=role, content=f"m{i}"))
        db.flush()
    return conv


class CountQueries:
    def __init__(self, conn):
        self.conn = conn
        self.count = 0

    def _inc(self, *args):
        self.count += 1

    def __enter__(self):
        event.listen(self.conn, "before_cursor_execute", self._inc)
        return self

    def __exit__(self, *exc):
        event.remove(self.conn, "before_cursor_execute", self._inc)


# --- isolation: user A must never see user B's data -------------------------


def test_list_documents_only_returns_own(db, make_user):
    alice, bob = make_user(), make_user()
    make_doc(db, alice, "a1.md")
    make_doc(db, alice, "a2.md")
    make_doc(db, bob, "b1.md")

    docs = queries.list_documents(db, alice.id)
    assert {d.filename for d in docs} == {"a1.md", "a2.md"}


def test_get_document_of_other_user_returns_none(db, make_user):
    alice, bob = make_user(), make_user()
    bobs_doc = make_doc(db, bob)

    assert queries.get_document(db, alice.id, bobs_doc.id) is None
    assert queries.get_document(db, bob.id, bobs_doc.id).id == bobs_doc.id


def test_get_document_by_hash_is_per_user(db, make_user):
    alice, bob = make_user(), make_user()
    doc = make_doc(db, alice)

    assert queries.get_document_by_hash(db, alice.id, doc.content_hash).id == doc.id
    assert queries.get_document_by_hash(db, bob.id, doc.content_hash) is None


def test_list_messages_of_other_users_conversation_is_empty(db, make_user):
    alice, bob = make_user(), make_user()
    conv = make_conv(db, bob, 3)

    assert queries.list_messages(db, alice.id, conv.id) == []
    assert [m.content for m in queries.list_messages(db, bob.id, conv.id)] == ["m0", "m1", "m2"]


def test_list_conversations_only_returns_own(db, make_user):
    alice, bob = make_user(), make_user()
    make_conv(db, alice, 1)
    make_conv(db, bob, 1)

    result = queries.list_conversations_with_last_message(db, alice.id)
    assert [s.conversation.user_id for s in result] == [alice.id]


# --- N+1 ---------------------------------------------------------------------


def test_conversations_with_last_message_is_one_query(db, make_user):
    user = make_user()
    convs = [make_conv(db, user, n) for n in (3, 1, 0, 5)]

    with CountQueries(db.connection()) as counter:
        result = queries.list_conversations_with_last_message(db, user.id)

    assert counter.count == 1
    last = {s.conversation.id: (s.last_message.content if s.last_message else None) for s in result}
    assert last == {convs[0].id: "m2", convs[1].id: "m0", convs[2].id: None, convs[3].id: "m4"}


# --- quota -------------------------------------------------------------------


def test_count_actions_since_counts_only_window_and_action(db, make_user):
    alice, bob = make_user(), make_user()
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    db.add_all(
        [
            UsageLog(user_id=alice.id, action=UsageAction.QUERY, created_at=today + timedelta(minutes=1)),
            UsageLog(user_id=alice.id, action=UsageAction.QUERY, created_at=today + timedelta(minutes=2)),
            UsageLog(user_id=alice.id, action=UsageAction.QUERY, created_at=today - timedelta(minutes=1)),
            UsageLog(user_id=alice.id, action=UsageAction.UPLOAD, created_at=today + timedelta(minutes=1)),
            UsageLog(user_id=bob.id, action=UsageAction.QUERY, created_at=today + timedelta(minutes=1)),
        ]
    )
    db.flush()

    assert queries.count_actions_since(db, alice.id, UsageAction.QUERY, today) == 2
