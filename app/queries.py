"""Query functions.

Rule: every function that reads user-owned data takes `user_id` and filters on it
in SQL. Never load a row by id and check ownership afterwards in Python.
"""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select, true
from sqlalchemy.orm import Session, aliased

from app.models import Conversation, Document, Message, UsageAction, UsageLog


def list_documents(db: Session, user_id: int, limit: int = 20, offset: int = 0) -> list[Document]:
    stmt = (
        select(Document)
        .where(Document.user_id == user_id)
        .order_by(Document.created_at.desc(), Document.id.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(db.scalars(stmt))


def get_document(db: Session, user_id: int, document_id: int) -> Document | None:
    """Returns None both for "doesn't exist" and "belongs to someone else"."""
    stmt = select(Document).where(Document.id == document_id, Document.user_id == user_id)
    return db.scalar(stmt)


def get_document_by_hash(db: Session, user_id: int, content_hash: str) -> Document | None:
    stmt = select(Document).where(Document.user_id == user_id, Document.content_hash == content_hash)
    return db.scalar(stmt)


def list_messages(db: Session, user_id: int, conversation_id: int) -> list[Message]:
    """Messages of a conversation, oldest first. Empty if the conversation isn't the user's."""
    stmt = (
        select(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .order_by(Message.created_at, Message.id)
    )
    return list(db.scalars(stmt))


@dataclass
class ConversationSummary:
    conversation: Conversation
    last_message: Message | None


def list_conversations_with_last_message(
    db: Session, user_id: int, limit: int = 20
) -> list[ConversationSummary]:
    """One SQL query, not 1 + N.

    The naive version loops over conversations and runs one "latest message" query
    per conversation. Here a LATERAL subquery picks the latest message per
    conversation, using the (conversation_id, created_at) index for each lookup.
    """
    last_msg = (
        select(Message)
        .where(Message.conversation_id == Conversation.id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(1)
        .lateral("last_msg")
    )
    last_msg_entity = aliased(Message, last_msg)
    stmt = (
        select(Conversation, last_msg_entity)
        .outerjoin(last_msg_entity, true())
        .where(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc(), Conversation.id.desc())
        .limit(limit)
    )
    return [ConversationSummary(conv, msg) for conv, msg in db.execute(stmt)]


def count_actions_since(db: Session, user_id: int, action: UsageAction, since: datetime) -> int:
    """Used for daily quota. Stage 4 replaces the hot path with a Redis counter."""
    stmt = select(func.count()).where(
        UsageLog.user_id == user_id,
        UsageLog.action == action,
        UsageLog.created_at >= since,
    )
    return db.scalar(stmt) or 0
