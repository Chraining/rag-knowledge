# Redis 客户端：配额计数和缓存共用这一个。
# 测试时会被替换成 fakeredis（一个在内存里模拟 Redis 的库），所以测试不需要真的 Redis。
# 使用方式：from app import redis_client，然后 redis_client.client.incr(...)
#   不要写 from app.redis_client import client：那样会在导入时把对象复制一份引用，测试就替换不掉了

# redis：Redis 的官方 Python 客户端
import redis

# 读取配置
from app.config import settings

# from_url：根据地址创建客户端。它内部有连接池，可以在多个请求之间共用
client = redis.Redis.from_url(
    settings.redis_url,
    # 连接和读写的超时都设成 1 秒。Redis 正常时响应在 1 毫秒以内；
    # 如果 1 秒都没响应，说明 Redis 出问题了，快速失败，让调用方走降级逻辑，而不是让用户一直等
    socket_connect_timeout=1,
    socket_timeout=1,
)
