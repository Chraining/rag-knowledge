"""Re-enqueue documents that got stuck before reaching ready/failed.

    uv run python -m scripts.requeue_pending

把卡住的文档重新投递到任务队列。什么时候会卡住：
  - 上传时 Redis 挂了，投递失败，文档停在 pending
  - worker 执行到一半，整台机器断电，消息也丢了，文档停在 processing
任务是幂等的（见 app/processing.py），所以就算某个文档其实还在正常排队，
重复投递也只是白做一次，不会产生重复数据。
生产环境里可以用 Celery Beat（定时任务）每隔几分钟自动跑一次这个逻辑。
"""

# datetime、timedelta、UTC：计算"10 分钟前"这个时间点
from datetime import UTC, datetime, timedelta

# select：构造查询
from sqlalchemy import select

# 会话工厂
from app.db import SessionLocal
# 模型
from app.models import Document, DocumentStatus
# 要投递的任务
from app.tasks import process_document

# 上传超过多少分钟还没处理完，才算"卡住"。太短的话，正常排队中的文档也会被重复投递
STUCK_AFTER = timedelta(minutes=10)


# 主函数
def main() -> None:
    # 截止时间：现在减去 10 分钟
    cutoff = datetime.now(UTC) - STUCK_AFTER
    with SessionLocal() as db:
        # 查出所有卡住的文档 id
        # in_(...)：SQL 的 IN，状态是 pending 或 processing 之一
        ids = db.scalars(
            select(Document.id).where(
                Document.status.in_([DocumentStatus.PENDING, DocumentStatus.PROCESSING]),
                Document.created_at < cutoff,
            )
        ).all()
    # 逐个重新投递
    for document_id in ids:
        process_document.delay(document_id)
    print(f"re-enqueued {len(ids)} document(s)")


# 只有直接运行这个文件时才执行 main()；被别的文件 import 时不执行
if __name__ == "__main__":
    main()
