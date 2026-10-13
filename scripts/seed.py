"""Fill the dev database with fake data.

    uv run python -m scripts.seed --users 200 --docs-per-user 50

Big enough that the planner actually prefers indexes (on a tiny table a seq scan
is cheaper, so EXPLAIN would show nothing interesting).

往开发数据库里填充假数据。用法见上面的命令。
数据量要足够大，查询优化器才会真正选择用索引。
（表很小时，直接全表扫描反而更快，EXPLAIN 就看不出索引的作用了。）
"""

# argparse：解析命令行参数（如 --users 200）
import argparse
# hashlib：计算哈希
import hashlib
# random：生成随机数
import random
# UTC 时区、日期时间、时间间隔
from datetime import UTC, datetime, timedelta

# insert：构造 INSERT 语句；text：原生 SQL
from sqlalchemy import insert, text

# 会话工厂（连接开发数据库）
from app.db import SessionLocal
# 导入所有模型和枚举
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

# 把枚举转成列表：[PENDING, PROCESSING, READY, FAILED]，方便随机挑一个
STATUSES = list(DocumentStatus)


# 主函数。参数：用户数、每个用户的文档数、每个用户的会话数、每个会话的消息数
# -> None 表示这个函数没有返回值
def main(n_users: int, docs_per_user: int, convs_per_user: int, msgs_per_conv: int) -> None:
    # 创建随机数生成器，种子固定为 42：每次运行生成的"随机"数据都一样，结果可以复现
    rnd = random.Random(42)
    # 当前时间（UTC）
    now = datetime.now(UTC)

    # SessionLocal.begin()：创建会话并开启事务
    # with 块正常结束时自动提交；中途出错时自动回滚——要么全部成功，要么什么都不写
    with SessionLocal.begin() as db:
        # TRUNCATE：快速清空这几张表
        # RESTART IDENTITY：把自增 id 重置回 1
        db.execute(
            text("TRUNCATE usage_logs, messages, conversations, chunks, documents, users RESTART IDENTITY")
        )

        # 批量插入用户
        # insert(User).returning(User.id)：INSERT ... RETURNING id，插入后把生成的 id 返回
        # 第二个参数是一个字典列表，每个字典是一行数据——这样一次就插入多行（批量插入，比逐条快得多）
        user_ids = db.scalars(
            insert(User).returning(User.id),
            [
                {
                    "username": f"user{i}",  # user0, user1, ...
                    "email": f"user{i}@example.com",
                    "password_hash": "not-a-real-hash",  # 假数据，不是真的哈希
                }
                for i in range(n_users)  # 列表推导式：生成 n_users 个字典
            ],
        ).all()  # .all() 把结果转成列表，得到所有新用户的 id

        # 准备文档数据
        docs = []  # 空列表，下面逐个往里加
        for uid in user_ids:  # 遍历每个用户
            for j in range(docs_per_user):  # 给每个用户造 docs_per_user 个文档
                status = rnd.choice(STATUSES)  # 随机选一个状态
                # 创建时间：过去 90 天内的随机时刻（60 * 24 * 90 是 90 天的分钟数）
                created = now - timedelta(minutes=rnd.randint(0, 60 * 24 * 90))
                docs.append(
                    {
                        "user_id": uid,
                        "filename": f"doc_{uid}_{j}.md",
                        "storage_path": f"storage/{uid}/{j}.md",
                        # 1KB 到 10MB 之间的随机大小。数字里的下划线只是为了好读，1_000 就是 1000
                        "file_size": rnd.randint(1_000, 10_000_000),
                        # 用"用户id-序号"算哈希，保证同一用户下不重复
                        "content_hash": hashlib.sha256(f"{uid}-{j}".encode()).hexdigest(),
                        "status": status,
                        # 只有 ready 的文档有段落（3 段），其他为 0
                        "chunk_count": 3 if status == DocumentStatus.READY else 0,
                        # 只有 failed 的文档有错误信息
                        "error_message": "parse error" if status == DocumentStatus.FAILED else None,
                        "created_at": created,
                        # 处理完成（ready 或 failed）的文档，处理时间是创建后 30 秒；其他为空
                        "processed_at": created + timedelta(seconds=30)
                        if status in (DocumentStatus.READY, DocumentStatus.FAILED)
                        else None,
                    }
                )
        # 批量插入文档，返回每个文档的 (id, status)，下面造段落时要用
        doc_rows = db.execute(insert(Document).returning(Document.id, Document.status), docs).all()

        # 只给 ready 状态的文档造段落，每个 3 段
        # 这是一个"双层循环"的列表推导式，等价于：
        #   for doc_id, status in doc_rows:
        #       if status == READY:
        #           for k in range(3): 生成一行
        chunks = [
            {"document_id": doc_id, "chunk_index": k, "content": f"chunk {k} of document {doc_id}"}
            for doc_id, status in doc_rows
            if status == DocumentStatus.READY
            for k in range(3)
        ]
        # 列表非空才插入（空列表在 Python 里等于 False）
        if chunks:
            db.execute(insert(Chunk), chunks)

        # 每个用户造 convs_per_user 个会话
        convs = [
            {"user_id": uid, "title": f"Conversation {c} of user {uid}"}
            for uid in user_ids
            for c in range(convs_per_user)
        ]
        # 批量插入会话，拿到所有会话 id
        conv_ids = db.scalars(insert(Conversation).returning(Conversation.id), convs).all()

        # 准备消息数据
        msgs = []
        for cid in conv_ids:  # 遍历每个会话
            # 会话开始时间：过去 30 天内的某一天
            start = now - timedelta(days=rnd.randint(0, 30))
            for m in range(msgs_per_conv):  # 每个会话 msgs_per_conv 条消息
                # 偶数条是用户提问，奇数条是 AI 回答
                role = MessageRole.USER if m % 2 == 0 else MessageRole.ASSISTANT
                msgs.append(
                    {
                        "conversation_id": cid,
                        "role": role,
                        "content": f"message {m} in conversation {cid}",
                        # AI 回答有来源（这里是空数组），用户提问没有来源（None）
                        "sources": [] if role == MessageRole.ASSISTANT else None,
                        # 每条消息间隔 10 秒，保证顺序明确
                        "created_at": start + timedelta(seconds=m * 10),
                    }
                )
        # 批量插入消息（不需要返回值，所以不用 returning）
        db.execute(insert(Message), msgs)

        # 每个用户造 100 条使用记录
        logs = [
            {
                "user_id": uid,
                "action": rnd.choice(list(UsageAction)),  # 随机 upload 或 query
                "tokens_used": rnd.randint(0, 2000),  # 0 到 2000 的随机 token 数
                # 过去 30 天内的随机时刻
                "created_at": now - timedelta(minutes=rnd.randint(0, 60 * 24 * 30)),
            }
            for uid in user_ids
            for _ in range(100)  # 下划线 _ 表示"这个循环变量我用不到"
        ]
        db.execute(insert(UsageLog), logs)

        # Fresh planner statistics, otherwise EXPLAIN guesses row counts badly.
        # ANALYZE：让 PostgreSQL 重新统计每张表的数据分布。
        # 查询优化器根据这些统计信息决定"用索引还是全表扫描"；不更新的话，它对行数的估计会很离谱
        db.execute(text("ANALYZE"))

    # 打印汇总信息。两个 f-string 挨着写，Python 会自动把它们拼成一个字符串
    print(
        f"seeded {len(user_ids)} users, {len(doc_rows)} documents, {len(chunks)} chunks, "
        f"{len(conv_ids)} conversations, {len(msgs)} messages, {len(logs)} usage logs"
    )


# 这个判断的意思是：只有"直接运行这个文件"时才执行下面的代码；
# 如果是被别的文件 import，下面的代码不会执行
if __name__ == "__main__":
    # 创建命令行参数解析器
    p = argparse.ArgumentParser()
    # 定义四个可选参数，都是整数，都有默认值
    p.add_argument("--users", type=int, default=200)
    p.add_argument("--docs-per-user", type=int, default=50)
    p.add_argument("--convs-per-user", type=int, default=10)
    p.add_argument("--msgs-per-conv", type=int, default=20)
    # 解析实际传入的参数
    a = p.parse_args()
    # 注意：参数名里的减号会变成下划线，--docs-per-user 对应 a.docs_per_user
    main(a.users, a.docs_per_user, a.convs_per_user, a.msgs_per_conv)
