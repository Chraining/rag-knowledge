"""Compare query plans with and without an index.

    uv run python -m scripts.explain

Runs EXPLAIN ANALYZE, drops the index inside a transaction, runs it again, then
rolls back. PostgreSQL DDL is transactional, so the index is never really lost.
Run scripts.seed first.

对比"有索引"和"没有索引"时的查询计划。用法见上面的命令。
做法：先执行一次 EXPLAIN ANALYZE；然后在事务里删掉索引，再执行一次；最后回滚事务。
PostgreSQL 的 DDL（建表、删索引这类语句）也支持事务，所以回滚后索引会恢复，不会真的丢失。
运行前要先执行 scripts.seed 填充数据。
"""

# text：写原生 SQL
from sqlalchemy import text

# 数据库引擎（连接开发数据库）
from app.db import engine

# 要对比的查询列表。每一项是一个元组：(标题, 索引名, SQL 语句)
CASES = [
    (
        "list a user's documents",  # 查某用户的文档列表
        "ix_documents_user_created",
        "SELECT * FROM documents WHERE user_id = 7 ORDER BY created_at DESC LIMIT 20",
    ),
    (
        "messages of one conversation",  # 查一个会话的消息
        "ix_messages_conversation_created",
        "SELECT * FROM messages WHERE conversation_id = 123 ORDER BY created_at",
    ),
    (
        "today's query count (quota)",  # 统计今日提问次数（配额）
        "ix_usage_logs_user_action_created",
        # 两个相邻的字符串会被 Python 自动拼接成一个
        # date_trunc('day', now())：把当前时间截断到"天"，也就是今天零点
        "SELECT count(*) FROM usage_logs "
        "WHERE user_id = 7 AND action = 'query' AND created_at >= date_trunc('day', now())",
    ),
]


# 执行 EXPLAIN 并把结果格式化成字符串
def explain(conn, sql: str) -> str:
    # EXPLAIN：显示数据库打算怎么执行这条 SQL（查询计划）
    # ANALYZE：真的执行一遍，显示实际耗时和行数
    # BUFFERS：显示读了多少数据页（越少越好）
    # .scalars()：结果每行只有一列（计划的一行文字），直接取出来
    rows = conn.execute(text(f"EXPLAIN (ANALYZE, BUFFERS) {sql}")).scalars()
    # 每行前面加 4 个空格缩进，再用换行符连接成一个字符串
    return "\n".join(f"    {r}" for r in rows)


def main() -> None:
    # 拿一个连接，with 块结束时自动归还
    with engine.connect() as conn:
        # 遍历每个查询，同时把元组拆成三个变量
        for title, index, sql in CASES:
            # \n 是换行符
            print(f"=== {title} ===\n{sql}\n")
            # 手动开启事务
            trans = conn.begin()
            # try ... finally：无论中间是否出错，finally 里的代码都一定会执行
            try:
                print(f"--- with {index}")  # 有索引时
                print(explain(conn, sql))
                # 删掉索引（在事务里，还没有生效到别的连接）
                conn.execute(text(f"DROP INDEX {index}"))
                print(f"--- without {index}")  # 没有索引时
                print(explain(conn, sql))
            finally:
                # 回滚：撤销 DROP INDEX，索引恢复原样
                trans.rollback()
            print()  # 打印一个空行，分隔不同的查询


# 直接运行这个文件时才执行 main()
if __name__ == "__main__":
    main()
