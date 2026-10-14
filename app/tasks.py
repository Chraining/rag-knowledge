# Celery 任务：在 worker 进程里执行的函数。
# 这一层只负责"怎么跑"（开数据库会话、失败重试、最终失败时记录原因），
# 具体"做什么"在 processing.py 里。

# get_task_logger：Celery 提供的日志工具，日志里会自动带上任务名和任务 id
from celery.utils.log import get_task_logger

# 处理逻辑
from app import processing
# Celery 应用
from app.celery_app import celery_app
# 会话工厂：worker 是独立的进程，不经过 FastAPI，所以要自己创建数据库会话
from app.db import SessionLocal

# 日志对象
logger = get_task_logger(__name__)

# 最多重试几次（不算第一次执行）。第一次加上 3 次重试，一共最多执行 4 次
MAX_RETRIES = 3


# 把文档标记为失败。单独开一个新的会话，因为出错时原来的会话可能已经不能用了（比如数据库连接断了）
def _mark_failed(document_id: int, message: str) -> None:
    with SessionLocal() as db:
        processing.mark_failed(db, document_id, message)


# @celery_app.task：把普通函数注册成 Celery 任务。注册之后：
#   process_document(5)        -> 在当前进程里直接执行（和普通函数一样）
#   process_document.delay(5)  -> 把消息放进队列，由 worker 在后台执行
# bind=True：函数的第一个参数 self 是任务本身，可以通过它读取"已经重试了几次"、调用 self.retry()
# max_retries：最多重试几次
@celery_app.task(bind=True, max_retries=MAX_RETRIES)
def process_document(self, document_id: int) -> None:
    # 任务参数只传 document_id（一个整数），不传文档对象或文件内容。原因：
    #   1. 消息要经过 Redis，越小越好，10MB 的文件不该塞进消息里
    #   2. 消息可能排队很久，执行时应该读数据库里的最新状态，而不是投递时的旧数据
    try:
        # with 块结束时自动关闭会话
        with SessionLocal() as db:
            processing.process_document(db, document_id)
        logger.info("document %s processed", document_id)
    except processing.PermanentError as e:
        # 永久性错误：重试没用，直接标记失败。正常结束任务（不抛异常），消息就会被确认、从队列删除
        logger.warning("document %s failed: %s", document_id, e)
        _mark_failed(document_id, str(e))
    except Exception as e:
        # 其他错误（数据库断开、磁盘读写出错等）可能只是暂时的，值得重试。
        # self.request.retries：已经重试了几次（第一次执行时是 0）
        if self.request.retries >= MAX_RETRIES:
            # 重试次数用完了，标记失败，让用户知道
            logger.error("document %s failed after %s retries: %r", document_id, MAX_RETRIES, e)
            # !r：用 repr 格式显示错误，会带上错误类型，例如 OperationalError('...')
            _mark_failed(document_id, f"Failed after {MAX_RETRIES} retries: {e!r}")
            # 把异常继续抛出去，Celery 会把这次任务记为失败
            raise
        # 计算等多久再重试：第 1 次等 1 秒，第 2 次 2 秒，第 3 次 4 秒（"指数退避"）。
        # 越等越久，是为了给出问题的数据库或服务留出恢复的时间，而不是一直猛敲它
        countdown = 2**self.request.retries
        logger.warning("document %s error, retry in %ss: %r", document_id, countdown, e)
        # self.retry 会重新投递这个任务（countdown 秒后执行），然后抛出一个特殊的 Retry 异常结束本次执行。
        # 前面加 raise 是 Celery 推荐的写法，让代码读起来更清楚"这里结束了"
        raise self.retry(exc=e, countdown=countdown)
