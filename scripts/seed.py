"""Fill the dev database with fake data.

    uv run python -m scripts.seed --users 200 --docs-per-user 50

Big enough that the planner actually prefers indexes (on a tiny table a seq scan
is cheaper, so EXPLAIN would show nothing interesting).
"""

import argparse
import hashlib
import random
from datetime import UTC, datetime, timedelta

from sqlalchemy import insert, text

from app.db import SessionLocal
from app.models import (
    Chunk,
    Conversation,
    Document,
    DocumentStatus,
    Message,
    MessageRole,
    UsageAction,
    UsageLog,
    User,
)

STATUSES = list(DocumentStatus)


def main(n_users: int, docs_per_user: int, convs_per_user: int, msgs_per_conv: int) -> None:
    rnd = random.Random(42)
    now = datetime.now(UTC)

    with SessionLocal.begin() as db:
        db.execute(
            text("TRUNCATE usage_logs, messages, conversations, chunks, documents, users RESTART IDENTITY")
        )

        user_ids = db.scalars(
            insert(User).returning(User.id),
            [
                {
                    "username": f"user{i}",
                    "email": f"user{i}@example.com",
                    "password_hash": "not-a-real-hash",
                }
                for i in range(n_users)
            ],
        ).all()

        docs = []
        for uid in user_ids:
            for j in range(docs_per_user):
                status = rnd.choice(STATUSES)
                created = now - timedelta(minutes=rnd.randint(0, 60 * 24 * 90))
                docs.append(
                    {
                        "user_id": uid,
                        "filename": f"doc_{uid}_{j}.md",
                        "storage_path": f"storage/{uid}/{j}.md",
                        "file_size": rnd.randint(1_000, 10_000_000),
                        "content_hash": hashlib.sha256(f"{uid}-{j}".encode()).hexdigest(),
                        "status": status,
                        "chunk_count": 3 if status == DocumentStatus.READY else 0,
                        "error_message": "parse error" if status == DocumentStatus.FAILED else None,
                        "created_at": created,
                        "processed_at": created + timedelta(seconds=30)
                        if status in (DocumentStatus.READY, DocumentStatus.FAILED)
                        else None,
                    }
                )
        doc_rows = db.execute(insert(Document).returning(Document.id, Document.status), docs).all()

        chunks = [
            {"document_id": doc_id, "chunk_index": k, "content": f"chunk {k} of document {doc_id}"}
            for doc_id, status in doc_rows
            if status == DocumentStatus.READY
            for k in range(3)
        ]
        if chunks:
            db.execute(insert(Chunk), chunks)

        convs = [
            {"user_id": uid, "title": f"Conversation {c} of user {uid}"}
            for uid in user_ids
            for c in range(convs_per_user)
        ]
        conv_ids = db.scalars(insert(Conversation).returning(Conversation.id), convs).all()

        msgs = []
        for cid in conv_ids:
            start = now - timedelta(days=rnd.randint(0, 30))
            for m in range(msgs_per_conv):
                role = MessageRole.USER if m % 2 == 0 else MessageRole.ASSISTANT
                msgs.append(
                    {
                        "conversation_id": cid,
                        "role": role,
                        "content": f"message {m} in conversation {cid}",
                        "sources": [] if role == MessageRole.ASSISTANT else None,
                        "created_at": start + timedelta(seconds=m * 10),
                    }
                )
        db.execute(insert(Message), msgs)

        logs = [
            {
                "user_id": uid,
                "action": rnd.choice(list(UsageAction)),
                "tokens_used": rnd.randint(0, 2000),
                "created_at": now - timedelta(minutes=rnd.randint(0, 60 * 24 * 30)),
            }
            for uid in user_ids
            for _ in range(100)
        ]
        db.execute(insert(UsageLog), logs)

        # Fresh planner statistics, otherwise EXPLAIN guesses row counts badly.
        db.execute(text("ANALYZE"))

    print(
        f"seeded {len(user_ids)} users, {len(doc_rows)} documents, {len(chunks)} chunks, "
        f"{len(conv_ids)} conversations, {len(msgs)} messages, {len(logs)} usage logs"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--docs-per-user", type=int, default=50)
    p.add_argument("--convs-per-user", type=int, default=10)
    p.add_argument("--msgs-per-conv", type=int, default=20)
    a = p.parse_args()
    main(a.users, a.docs_per_user, a.convs_per_user, a.msgs_per_conv)
