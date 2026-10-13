"""Constraint behaviour: these are enforced by the database, not by app code."""

import hashlib

import pytest
from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError

from app.models import Chunk, Conversation, Document, DocumentStatus, Message, MessageRole, User


def make_doc(db, user, name="a.md", content=b"hello") -> Document:
    doc = Document(
        user_id=user.id,
        filename=name,
        storage_path=f"storage/{user.id}/{name}",
        file_size=len(content),
        content_hash=hashlib.sha256(content).hexdigest(),
    )
    db.add(doc)
    db.flush()
    return doc


def test_new_document_defaults_to_pending(db, make_user):
    doc = make_doc(db, make_user())
    db.refresh(doc)
    assert doc.status == DocumentStatus.PENDING
    assert doc.chunk_count == 0


def test_invalid_status_is_rejected(db, make_user):
    doc = make_doc(db, make_user())
    with pytest.raises(IntegrityError):
        db.execute(text("UPDATE documents SET status = 'weird' WHERE id = :id"), {"id": doc.id})


def test_same_user_same_content_is_rejected(db, make_user):
    user = make_user()
    make_doc(db, user, "a.md", b"same")
    with pytest.raises(IntegrityError):
        make_doc(db, user, "renamed.md", b"same")


def test_different_users_may_upload_same_content(db, make_user):
    make_doc(db, make_user(), "a.md", b"same")
    make_doc(db, make_user(), "a.md", b"same")  # no error


def test_deleting_document_deletes_its_chunks(db, make_user):
    doc = make_doc(db, make_user())
    db.add_all([Chunk(document_id=doc.id, chunk_index=i, content="x") for i in range(3)])
    db.flush()

    # Core DELETE bypasses ORM cascades, so this proves the DB-level ON DELETE CASCADE.
    db.execute(delete(Document).where(Document.id == doc.id))
    assert db.scalar(select(func.count()).select_from(Chunk)) == 0


def test_duplicate_chunk_index_is_rejected(db, make_user):
    doc = make_doc(db, make_user())
    db.add(Chunk(document_id=doc.id, chunk_index=0, content="x"))
    db.flush()
    db.add(Chunk(document_id=doc.id, chunk_index=0, content="y"))
    with pytest.raises(IntegrityError):
        db.flush()


def test_user_with_documents_cannot_be_hard_deleted(db, make_user):
    user = make_user()
    make_doc(db, user)
    with pytest.raises(IntegrityError):
        db.execute(delete(User).where(User.id == user.id))


def test_deleting_conversation_deletes_messages(db, make_user):
    user = make_user()
    conv = Conversation(user_id=user.id, title="t")
    db.add(conv)
    db.flush()
    db.add(Message(conversation_id=conv.id, role=MessageRole.USER, content="hi"))
    db.flush()

    db.execute(delete(Conversation).where(Conversation.id == conv.id))
    assert db.scalar(select(func.count()).select_from(Message)) == 0

