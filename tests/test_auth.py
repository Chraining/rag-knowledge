# 测试认证接口：注册、登录、token 校验

# sqlalchemy 的 select：直接查数据库，验证存进去的数据
from sqlalchemy import select

# 读取配置（测试过期 token 时临时修改有效期）
from app.config import settings
# 用户模型
from app.models import User
# 直接签发 token，用来构造特殊的测试用 token
from app.security import create_access_token


# --- 注册 --------------------------------------------------------------------


# 测试：注册成功，响应里没有密码，数据库里存的是哈希而不是明文
def test_register_returns_user_without_password(client, db):
    r = client.post(
        "/auth/register",
        json={"username": "alice", "email": "alice@example.com", "password": "password123"},
    )
    assert r.status_code == 201
    body = r.json()  # 把响应的 JSON 解析成字典
    assert body["username"] == "alice"
    # 响应里不能出现任何和密码有关的字段
    assert "password" not in body and "password_hash" not in body

    # 直接查数据库，确认存的不是明文
    user = db.scalar(select(User).where(User.username == "alice"))
    assert user.password_hash != "password123"
    # argon2 哈希的固定开头
    assert user.password_hash.startswith("$argon2")


# 测试：用户名和邮箱会被转成小写
def test_register_lowercases_username_and_email(client):
    r = client.post(
        "/auth/register",
        json={"username": "Alice", "email": "Alice@Example.com", "password": "password123"},
    )
    assert r.json()["username"] == "alice"
    assert r.json()["email"] == "alice@example.com"


# 测试：用户名只是大小写不同，也算重复，返回 409
def test_register_duplicate_username_case_insensitive(client, register_and_login):
    register_and_login("alice")
    r = client.post(
        "/auth/register",
        json={"username": "ALICE", "email": "other@example.com", "password": "password123"},
    )
    assert r.status_code == 409


# 测试：邮箱重复返回 409
def test_register_duplicate_email(client, register_and_login):
    register_and_login("alice")  # 邮箱是 alice@example.com
    r = client.post(
        "/auth/register",
        json={"username": "bob", "email": "alice@example.com", "password": "password123"},
    )
    assert r.status_code == 409


# 测试：不合法的输入返回 422（由 Pydantic 自动校验，接口函数不会执行）
def test_register_rejects_invalid_input(client):
    # 密码太短
    r = client.post("/auth/register", json={"username": "alice", "email": "a@example.com", "password": "short"})
    assert r.status_code == 422
    # 用户名有非法字符
    r = client.post("/auth/register", json={"username": "a b!", "email": "a@example.com", "password": "password123"})
    assert r.status_code == 422
    # 邮箱格式不对
    r = client.post("/auth/register", json={"username": "alice", "email": "not-an-email", "password": "password123"})
    assert r.status_code == 422


# --- 登录 --------------------------------------------------------------------


# 测试：登录成功拿到 token，用 token 能访问 /auth/me
def test_login_and_me(client, register_and_login):
    headers = register_and_login("alice")
    r = client.get("/auth/me", headers=headers)
    assert r.status_code == 200
    assert r.json()["username"] == "alice"


# 测试：登录时用户名大小写不敏感
def test_login_username_case_insensitive(client, register_and_login):
    register_and_login("alice")
    r = client.post("/auth/login", data={"username": "ALICE", "password": "password123"})
    assert r.status_code == 200


# 测试：密码错误和用户不存在，返回完全相同的错误（防止探测用户名是否存在）
def test_login_failures_are_indistinguishable(client, register_and_login):
    register_and_login("alice")
    wrong_password = client.post("/auth/login", data={"username": "alice", "password": "wrong-password"})
    no_such_user = client.post("/auth/login", data={"username": "nobody", "password": "password123"})
    assert wrong_password.status_code == no_such_user.status_code == 401
    # 错误信息也一样
    assert wrong_password.json() == no_such_user.json()


# 测试：被停用的账号不能登录
def test_inactive_user_cannot_login(client, db, register_and_login):
    register_and_login("alice")
    # 直接改数据库，模拟管理员停用了这个账号
    db.scalar(select(User).where(User.username == "alice")).is_active = False
    db.flush()
    r = client.post("/auth/login", data={"username": "alice", "password": "password123"})
    assert r.status_code == 401


# --- token 校验 --------------------------------------------------------------


# 测试：没有 token 访问需要登录的接口，返回 401
def test_me_without_token(client):
    assert client.get("/auth/me").status_code == 401


# 测试：伪造或乱写的 token，返回 401
def test_me_with_garbage_token(client):
    r = client.get("/auth/me", headers={"Authorization": "Bearer not.a.real.token"})
    assert r.status_code == 401


# 测试：用别的密钥签名的 token（模拟攻击者伪造），返回 401
def test_me_with_token_signed_by_other_key(client, register_and_login, monkeypatch):
    register_and_login("alice")
    # 临时把密钥换掉，签发一个 token
    monkeypatch.setattr(settings, "jwt_secret_key", "attacker-key-that-is-at-least-32-bytes-long")
    forged = create_access_token(1)
    # 换回真正的密钥（monkeypatch.undo 恢复所有被临时修改的属性）
    monkeypatch.undo()
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401


# 测试：过期的 token，返回 401
def test_me_with_expired_token(client, db, register_and_login, monkeypatch):
    register_and_login("alice")
    user = db.scalar(select(User).where(User.username == "alice"))
    # 有效期设成 -1 分钟，签发出来的 token 一出生就已经过期
    monkeypatch.setattr(settings, "access_token_expire_minutes", -1)
    expired = create_access_token(user.id)
    r = client.get("/auth/me", headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401


# 测试：账号被停用后，之前签发的 token 立即失效
def test_token_rejected_after_user_deactivated(client, db, register_and_login):
    headers = register_and_login("alice")
    # 停用前可以访问
    assert client.get("/auth/me", headers=headers).status_code == 200
    db.scalar(select(User).where(User.username == "alice")).is_active = False
    db.flush()
    # 停用后，同一个 token 被拒绝
    assert client.get("/auth/me", headers=headers).status_code == 401
