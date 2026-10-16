# 健康检查接口：GET /health。给 Docker、监控工具、负载均衡器用，判断应用"是不是真的能干活"。
# 只看进程在不在是不够的：进程还在，但连不上数据库，所有请求照样失败。
#
# 判断规则：
#   数据库连不上 → 503（不健康）：没有数据库，几乎所有接口都没法用
#   Redis 连不上 → 仍然 200，但标记 "redis": "down"：
#     应用能降级运行（配额改用数据库计数、缓存跳过），只是上传的文档暂时进不了队列。
#     如果这时报"不健康"，监控会以为应用本身坏了，而重启应用并不能修好 Redis

# logging：记录日志
import logging

# APIRouter：路由分组；Response：用来修改响应的状态码
from fastapi import APIRouter, Response, status
# text：写原生 SQL
from sqlalchemy import text

# Redis 客户端（这样写，测试里替换成 fakeredis 才会生效，见 redis_client.py）
from app import redis_client
# 依赖：数据库会话
from app.deps import DbSession

# 日志对象
logger = logging.getLogger(__name__)

# 路由分组，Swagger 里归到 "health" 组
router = APIRouter(tags=["health"])


# GET /health：不需要登录（监控工具没有 token）
@router.get("/health")
def health(response: Response, db: DbSession) -> dict[str, str]:
    # 检查数据库：执行一条最简单的 SQL
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except Exception:
        # 什么错误都算"连不上"。详细原因记进日志，不返回给调用方（可能包含内部地址等信息）
        logger.exception("health check: database unreachable")
        database = "down"

    # 检查 Redis：发一个 PING，正常会回 PONG
    try:
        redis_client.client.ping()
        redis = "ok"
    except Exception:
        logger.warning("health check: redis unreachable")
        redis = "down"

    # 只有数据库挂了才算不健康
    if database == "down":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if database == "ok" else "down", "database": database, "redis": redis}
