# RAG（检索增强生成）的辅助步骤：把问题转成向量（带缓存）。
# 检索工具在 agent.py 里调用它；整个问答流程的串联在 agent.py 和 routers/chat.py 里。

# hashlib：给问题计算哈希，作为缓存的键
import hashlib
# json：向量是一串浮点数，存进 Redis 之前要转成字符串
import json

# RedisError：Redis 出错
from redis.exceptions import RedisError

# llm：调用 embedding；redis_client：缓存
from app import llm, redis_client
# 读取配置（embedding 模型名）
from app.config import settings

# 问题向量在缓存里保存多久：7 天
EMBEDDING_CACHE_SECONDS = 7 * 24 * 3600


# 把问题转换成向量。先查缓存，缓存里没有才调用 embedding 服务
#
# 为什么缓存"问题的向量"，而不是缓存"检索结果"：
#   检索结果依赖用户当前有哪些文档，用户一上传或删除文档，缓存就过期了，还得想办法清理（缓存失效问题）；
#   而同一段文字的向量永远不变，缓存永远不会过期出错。
#   而且两步里真正慢、真正花钱的是 embedding（调外部 API），向量检索只是一条数据库查询
def embed_query(question: str) -> list[float]:
    # 键里带上模型名：换了模型，向量就不一样了，不能再用旧缓存
    digest = hashlib.sha256(question.encode()).hexdigest()
    key = f"emb:{settings.embedding_model}:{digest}"
    try:
        cached = redis_client.client.get(key)
    except RedisError:
        # 缓存只是用来加速的，Redis 出问题时就当作"没缓存"，照常调用 embedding 服务
        cached = None
    if cached is not None:
        return json.loads(cached)
    # to_list()：Vector 转回普通的 float 列表，才能用 json 存进 Redis
    vector = llm.embed_texts([question])[0].to_list()
    try:
        # ex：过期时间（秒）
        redis_client.client.set(key, json.dumps(vector), ex=EMBEDDING_CACHE_SECONDS)
    except RedisError:
        pass  # 写缓存失败不影响本次回答
    return vector

