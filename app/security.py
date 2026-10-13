# 安全相关的工具函数：密码哈希、JWT 的签发和解析。
# 这里只放"纯计算"的函数，不访问数据库，所以很容易单独测试。

# datetime：日期时间；timedelta：时间间隔；UTC：UTC 时区
from datetime import UTC, datetime, timedelta

# PyJWT 库：生成和验证 JWT
import jwt
# pwdlib：密码哈希库
from pwdlib import PasswordHash

# 读取配置（JWT 密钥、算法、有效期）
from app.config import settings

# 创建密码哈希器。recommended() 使用当前推荐的算法 argon2。
# argon2 是 2015 年密码哈希竞赛的冠军，特点是：
#   1. 慢：故意让每次计算需要几十毫秒，攻击者暴力猜密码的成本就会很高
#   2. 吃内存：让攻击者很难用显卡或专用芯片大规模并行破解
#   3. 自动加盐：每次哈希都会生成随机的"盐"，同一个密码两次哈希结果也不同，
#      这样攻击者没法用提前算好的"密码 -> 哈希"对照表（彩虹表）来反查
_password_hash = PasswordHash.recommended()

# 一个"假的"哈希值，在用户不存在时拿来做一次验证（见 verify_password_or_dummy）。
# 模块加载时计算一次即可。
_DUMMY_HASH = _password_hash.hash("dummy-password-for-timing")


# 把明文密码变成哈希值，注册时调用。返回的字符串里包含了算法、参数、盐和哈希结果
def hash_password(password: str) -> str:
    return _password_hash.hash(password)


# 登录时验证密码。password_hash 为 None 表示"用户不存在"
def verify_password_or_dummy(password: str, password_hash: str | None) -> bool:
    # 用户不存在时，也要用假哈希做一次完整的验证计算，再返回 False。
    # 原因（防"时序攻击"）：如果用户不存在就立刻返回，响应会比"用户存在但密码错"快几十毫秒。
    # 攻击者通过测量响应时间，就能判断出某个用户名是否已注册。
    if password_hash is None:
        _password_hash.verify(password, _DUMMY_HASH)
        return False
    # 用户存在：正常验证。verify 会从哈希字符串里取出盐，重新计算并比较
    return _password_hash.verify(password, password_hash)


# 生成访问令牌（JWT）。登录成功后调用，返回 token 字符串
def create_access_token(user_id: int) -> str:
    # 当前时间
    now = datetime.now(UTC)
    # JWT 的"载荷"（payload），也就是 token 里携带的信息。
    # 注意：JWT 只是签名，不是加密！任何人都能解码看到载荷内容，所以绝不能放密码等敏感信息。
    # 签名的作用是：防止被篡改。改了内容，签名就对不上了。
    payload = {
        # sub（subject）：这个 token 代表谁。JWT 标准要求它是字符串，所以要 str()
        "sub": str(user_id),
        # iat（issued at）：签发时间
        "iat": now,
        # exp（expiration）：过期时间。PyJWT 解码时会自动检查，过期就抛异常
        "exp": now + timedelta(minutes=settings.access_token_expire_minutes),
    }
    # 用密钥和算法对载荷签名，生成 token
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


# 解析 token，返回其中的用户 id。token 无效（签名不对、已过期、格式错误）时返回 None
def decode_access_token(token: str) -> int | None:
    # try ... except：尝试执行，出错时进入 except 分支，而不是让程序崩溃
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            # algorithms 必须明确指定允许的算法列表。
            # 如果不指定，攻击者可以把 token 头部的算法改成 "none"（不签名），绕过验证
            algorithms=[settings.jwt_algorithm],
            # 要求 token 里必须有 exp 和 sub，缺了就当作无效
            options={"require": ["exp", "sub"]},
        )
        # 把 sub 从字符串转回整数
        return int(payload["sub"])
    # jwt.InvalidTokenError：签名错误、过期、格式错误等所有 JWT 异常的父类
    # ValueError：sub 不是数字时 int() 会抛出
    except (jwt.InvalidTokenError, ValueError):
        return None
