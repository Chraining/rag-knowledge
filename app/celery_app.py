# Celery 应用：后台任务队列的配置。
# 工作方式：
#   1. 上传接口调用 process_document.delay(文档id)，把一条消息放进 Redis（broker，"消息中转站"）
#   2. 另一个独立的进程（worker）不断从 Redis 取消息，取到一条就执行一次任务
#   3. 上传接口不用等任务做完，立刻返回给用户
# 启动 worker（Windows 上必须加 --pool=solo，因为 Celery 默认的多进程模式不支持 Windows）：
#   uv run celery -A app.celery_app worker --pool=solo --loglevel=info

# Celery：任务队列框架
from celery import Celery

# 读取配置（broker 地址）
from app.config import settings

# 创建 Celery 应用
#   "rag"：应用名字，会出现在日志里
#   broker：消息队列的地址
#   include：worker 启动时要导入的模块，任务函数定义在这些模块里
celery_app = Celery("rag", broker=settings.celery_broker_url, include=["app.tasks"])

# 修改 Celery 的配置
celery_app.conf.update(
    # acks_late：任务"执行完"才向 broker 确认（ack），而不是"一取到"就确认。
    #   默认是一取到就确认：如果 worker 执行到一半崩溃了，这条消息已经被删掉，文档永远卡在 processing。
    #   改成执行完再确认：worker 崩溃时消息还在，会被重新投递给别的 worker。
    #   代价：同一个任务可能被执行不止一次，所以任务必须是"幂等"的（见 processing.py）
    task_acks_late=True,
    # worker 进程被杀死（比如内存不够被系统杀掉）时，把消息退回队列，让它被重新执行
    task_reject_on_worker_lost=True,
    # 每个 worker 一次只预取 1 条消息。
    #   默认会预取好几条，如果当前任务很慢，预取的消息就一直排在它后面，别的空闲 worker 拿不到
    worker_prefetch_multiplier=1,
    # 不保存任务的返回值。处理结果（状态、段数、错误信息）已经写进了 documents 表，
    # 用户查数据库就知道进度，不需要再多一个地方存结果
    task_ignore_result=True,
    # worker 启动时如果连不上 broker，自动重试，而不是直接退出
    broker_connection_retry_on_startup=True,
)
