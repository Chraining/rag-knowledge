# 测试健康检查接口

# get_db：要替换掉的依赖
from app.db import get_db
# FastAPI 应用（用来临时替换依赖）
from app.main import app


# 一个"坏掉的"数据库会话：执行任何 SQL 都报错，模拟数据库连不上
class BrokenSession:
    def execute(self, *args, **kwargs):
        raise ConnectionError("database is down")


# 测试：一切正常时返回 200，不需要登录
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "database": "ok", "redis": "ok"}


# 测试：Redis 挂了，仍然返回 200（应用能降级运行），但标出 redis 是 down
def test_health_redis_down_is_degraded_not_failed(client, fake_redis):
    fake_redis.connected = False
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["redis"] == "down"


# 测试：数据库挂了，返回 503
def test_health_database_down_returns_503(client):
    # 临时把数据库依赖换成坏掉的会话。client fixture 本身也替换了 get_db，所以测试结束要恢复它
    original = app.dependency_overrides[get_db]
    app.dependency_overrides[get_db] = lambda: BrokenSession()
    try:
        r = client.get("/health")
    finally:
        app.dependency_overrides[get_db] = original
    assert r.status_code == 503
    assert r.json()["database"] == "down"
