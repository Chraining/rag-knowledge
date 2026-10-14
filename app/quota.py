# 每日提问配额：每个用户每天最多提问 daily_quota 次（LLM 按 token 收费，必须限制）。
#
# 为什么用 Redis 计数，而不是每次都 SELECT COUNT(*) FROM usage_logs：
#   每次提问都要检查配额，是高频操作。COUNT 要扫描索引里今天的所有记录，用户越活跃越慢；
#   Redis 的 INCR 是内存操作，不管计数是多少，耗时都一样（不到 1 毫秒）。
#
# 为什么用 INCR（先加 1，再判断），而不是"先读出来，判断没超，再加 1"：
#   后一种写法有并发问题：用户同时发 10 个请求，它们可能都读到"已用 49 次"，都判断"没超"，
#   结果一共用了 59 次。INCR 是原子操作，10 个请求一定拿到 50、51、...、59 这 10 个不同的值。
#
# 降级：Redis 挂了怎么办？改用数据库的 usage_logs 来数（慢，但结果正确），而不是直接拒绝所有提问

# datetime、timedelta、timezone：计算北京时间的"今天"
from datetime import datetime, timedelta, timezone

# RedisError：redis 库所有错误的基类（连不上、超时等）
from redis.exceptions import RedisError
# Session：数据库会话的类型
from sqlalchemy.orm import Session

# queries：Redis 不可用时，用数据库计数；redis_client：Redis 客户端
from app import queries, redis_client
# 模型
from app.models import UsageAction, User

# 北京时间 = UTC + 8 小时。中国没有夏令时，所以用固定的时差就够了，
# 不需要 zoneinfo（它在 Windows 上还要额外安装 tzdata 包）
BEIJING = timezone(timedelta(hours=8))


# 配额用完了
class QuotaExceeded(Exception):
    pass


# 北京时间今天零点（带时区）
def _today_start() -> datetime:
    # 先取北京时间的"现在"，再把时、分、秒、微秒都设成 0
    return datetime.now(BEIJING).replace(hour=0, minute=0, second=0, microsecond=0)


# 这个用户今天的计数器在 Redis 里的键名，例如 "quota:7:2026-10-14"。
# 日期写进键名里：第二天自动换成一个新的键，计数从 0 开始，不需要定时任务去清零
def _key(user_id: int) -> str:
    return f"quota:{user_id}:{_today_start():%Y-%m-%d}"


# 用掉一次配额。超了就抛出 QuotaExceeded
def consume(db: Session, user: User) -> None:
    key = _key(user.id)
    try:
        # pipeline：把两条命令打包，一次网络往返发给 Redis
        pipe = redis_client.client.pipeline()
        pipe.incr(key)  # 计数加 1，返回加完之后的值；键不存在时从 0 开始
        # 2 天后自动删除这个键，避免 Redis 里堆满旧日期的计数器。
        # 设成 2 天而不是 1 天，是为了留余量：只要能撑过"今天"就够了
        pipe.expire(key, timedelta(days=2))
        # execute() 返回每条命令的结果，第一条（incr）的结果就是新的计数
        count = pipe.execute()[0]
    except RedisError:
        # Redis 不可用：改用数据库数今天已经成功提问了几次，再加上这一次
        count = queries.count_actions_since(db, user.id, UsageAction.QUERY, _today_start()) + 1
    if count > user.daily_quota:
        raise QuotaExceeded


# 退还一次配额：提问失败（比如 LLM 挂了）时调用，不能让用户为没得到的回答买单
def refund(user_id: int) -> None:
    try:
        redis_client.client.decr(_key(user_id))
    except RedisError:
        # Redis 不可用时，consume 用的是数据库计数，而失败的提问不会写 usage_logs，
        # 所以本来就没有算进去，这里什么都不用做
        pass
