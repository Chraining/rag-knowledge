# conftest.py 是 pytest 的特殊文件：这里定义的 fixture 会自动提供给同目录下的所有测试。
# fixture（测试夹具）= 测试运行前的"准备工作"，比如连好数据库、造好测试用户。
# 测试函数只要在参数里写上 fixture 的名字（如 db），pytest 就会自动把准备好的东西传进来。

# pytest：测试框架
import pytest
# command：Alembic 的命令接口，可以在代码里执行"alembic upgrade head"
from alembic import command
# Config：Alembic 的配置对象，对应 alembic.ini 文件
from alembic.config import Config
# TestClient：FastAPI 的测试客户端，不用真的启动服务器就能发 HTTP 请求
from fastapi.testclient import TestClient
# create_engine：创建数据库引擎；text：写原生 SQL
from sqlalchemy import create_engine, text
# Session：数据库会话
from sqlalchemy.orm import Session

# 读取配置（拿测试数据库地址）
from app.config import settings
# get_db：要在测试里替换掉的依赖
from app.db import get_db
# FastAPI 应用
from app.main import app
# User 模型，造测试用户时用
from app.models import User


# scope="session"：整个测试过程只执行一次（而不是每个测试都执行一次），因为建表比较慢
@pytest.fixture(scope="session")
def engine():
    # Build the schema through the real migrations, so they are tested too.
    # 用真正的迁移文件来建表，这样迁移文件本身也被顺带测试了
    # 连接测试数据库（rag_test），不碰开发数据库（rag）
    eng = create_engine(settings.test_database_url)
    # eng.begin() 开启一个事务，with 块结束时自动提交
    with eng.begin() as conn:
        # 删掉整个 public schema（里面所有表都会被删），再新建一个空的
        # 保证每次跑测试都从完全干净的数据库开始
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    # 读取 alembic.ini
    cfg = Config("alembic.ini")
    # 把数据库地址改成测试库
    cfg.set_main_option("sqlalchemy.url", settings.test_database_url)
    # 相当于在命令行执行 alembic upgrade head：把所有迁移执行到最新版本，建出所有表
    command.upgrade(cfg, "head")
    # yield 把引擎交给测试使用；所有测试结束后，才会继续执行 yield 后面的代码
    yield eng
    # 清理：关闭连接池里的所有连接
    eng.dispose()


# 没写 scope，默认是 "function"：每个测试函数都会重新执行一次
# 参数 engine 表示这个 fixture 依赖上面的 engine fixture
@pytest.fixture
def db(engine):
    """Each test runs inside a transaction that is rolled back afterwards.

    Code under test may call commit(); with create_savepoint those commits only
    release a SAVEPOINT, and the outer transaction still gets rolled back.

    每个测试都在一个事务里运行，测试结束后回滚（撤销所有改动）。
    这样每个测试之间互不影响，而且不用手动清理数据。
    被测试的代码可能会调用 commit()；用了 create_savepoint 模式后，
    这些 commit 只会释放一个"保存点"，外层事务最后仍然会被回滚。
    """
    # 从连接池拿一个连接
    conn = engine.connect()
    # 在这个连接上开启外层事务
    trans = conn.begin()
    # 创建一个绑定到这个连接上的 Session
    # join_transaction_mode="create_savepoint"：Session 不自己开新事务，而是在外层事务里建保存点
    session = Session(bind=conn, join_transaction_mode="create_savepoint", expire_on_commit=False)
    # 把 session 交给测试函数
    yield session
    # 测试结束后的清理：
    session.close()  # 关闭会话
    trans.rollback()  # 回滚外层事务，测试中写入的所有数据都消失
    conn.close()  # 把连接还给连接池


# 一个"工厂"fixture：返回的是一个函数，测试里可以多次调用它来创建多个用户
@pytest.fixture
def make_user(db):
    # 计数器，保证每个用户的用户名和邮箱都不同（因为它们有唯一约束）
    counter = 0

    # 内部函数。**kw 表示接收任意数量的关键字参数，例如 make_user(daily_quota=5)
    def _make(**kw) -> User:
        # nonlocal：声明 counter 是外层函数的变量，这样才能修改它
        nonlocal counter
        counter += 1  # 每创建一个用户，计数加 1
        user = User(
            # kw.pop("username", 默认值)：如果调用时传了 username 就用它，否则用 "u1"、"u2"……
            username=kw.pop("username", f"u{counter}"),
            email=kw.pop("email", f"u{counter}@example.com"),
            password_hash="x",  # 测试不关心密码，随便填
            **kw,  # 把剩下的参数原样传给 User（如 daily_quota=5）
        )
        db.add(user)  # 把对象加入会话（此时还没写数据库）
        # flush：把待写入的改动发送给数据库执行（INSERT），但不提交事务
        # 执行后数据库会生成 id，user.id 就有值了
        db.flush()
        return user

    # 返回这个函数本身（注意没有括号，不是调用它）
    return _make


# autouse=True：每个测试都会自动使用这个 fixture，不用写在参数里。
# 作用：把上传接口里的"投递后台任务"换成"记到一个列表里"。
#   1. 测试不需要真的启动 Redis
#   2. 测试可以检查"有没有投递、投递的是哪个文档"
# 测试里写参数 enqueued 就能拿到这个列表
@pytest.fixture(autouse=True)
def enqueued(monkeypatch):
    # 记录被投递的文档 id
    calls: list[int] = []

    # 一个假的任务对象，只有 delay 方法。
    # staticmethod：这个方法不需要 self 参数
    class FakeTask:
        delay = staticmethod(calls.append)

    # 替换 documents 路由模块里的 process_document（它在 import 时就绑定到了模块的变量上）
    monkeypatch.setattr("app.routers.documents.process_document", FakeTask)
    return calls


# 测试客户端：用它给应用发请求，就像真实的前端一样
# monkeypatch：pytest 自带的 fixture，用来临时修改属性，测试结束后自动恢复
# tmp_path：pytest 自带的 fixture，每个测试一个全新的临时目录，测试结束后自动清理
@pytest.fixture
def client(db, monkeypatch, tmp_path):
    # 把上传文件的存储目录改成临时目录，测试不会往项目的 storage/ 里写文件
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    # 替换掉 get_db 依赖：让接口使用测试的 db 会话（在会被回滚的事务里），而不是连开发数据库
    def override_get_db():
        yield db

    # dependency_overrides：FastAPI 提供的机制，测试时把某个依赖换成另一个函数
    app.dependency_overrides[get_db] = override_get_db
    # with 块：启动和关闭测试客户端
    with TestClient(app) as c:
        yield c
    # 测试结束后清除替换，避免影响别的测试
    app.dependency_overrides.clear()


# 工厂 fixture：通过真实的接口注册并登录一个用户，返回带 token 的请求头
# 用法：headers = register_and_login("alice")，之后 client.get(..., headers=headers)
@pytest.fixture
def register_and_login(client):
    def _register_and_login(username: str, password: str = "password123") -> dict[str, str]:
        # 调用注册接口
        r = client.post(
            "/auth/register",
            json={"username": username, "email": f"{username}@example.com", "password": password},
        )
        # 注册必须成功，否则后面的测试没意义。assert 后面的 r.text 是失败时显示的信息
        assert r.status_code == 201, r.text
        # 调用登录接口。登录用的是表单格式（data=），不是 JSON（json=）
        r = client.post("/auth/login", data={"username": username, "password": password})
        assert r.status_code == 200, r.text
        # 返回请求头：Authorization: Bearer <token>
        return {"Authorization": f"Bearer {r.json()['access_token']}"}

    return _register_and_login
