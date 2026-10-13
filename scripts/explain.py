"""Compare query plans with and without an index.

    uv run python -m scripts.explain

Runs EXPLAIN ANALYZE, drops the index inside a transaction, runs it again, then
rolls back. PostgreSQL DDL is transactional, so the index is never really lost.
Run scripts.seed first.
"""

from sqlalchemy import text

from app.db import engine

CASES = [
    (
        "list a user's documents",
        "ix_documents_user_created",
        "SELECT * FROM documents WHERE user_id = 7 ORDER BY created_at DESC LIMIT 20",
    ),
    (
        "messages of one conversation",
        "ix_messages_conversation_created",
        "SELECT * FROM messages WHERE conversation_id = 123 ORDER BY created_at",
    ),
    (
        "today's query count (quota)",
        "ix_usage_logs_user_action_created",
        "SELECT count(*) FROM usage_logs "
        "WHERE user_id = 7 AND action = 'query' AND created_at >= date_trunc('day', now())",
    ),
]


def explain(conn, sql: str) -> str:
    rows = conn.execute(text(f"EXPLAIN (ANALYZE, BUFFERS) {sql}")).scalars()
    return "\n".join(f"    {r}" for r in rows)


def main() -> None:
    with engine.connect() as conn:
        for title, index, sql in CASES:
            print(f"=== {title} ===\n{sql}\n")
            trans = conn.begin()
            try:
                print(f"--- with {index}")
                print(explain(conn, sql))
                conn.execute(text(f"DROP INDEX {index}"))
                print(f"--- without {index}")
                print(explain(conn, sql))
            finally:
                trans.rollback()
            print()


if __name__ == "__main__":
    main()
