# 接口的公共"依赖"。
# FastAPI 的依赖注入：接口函数在参数里声明"我需要什么"，FastAPI 负责在调用前准备好。
# 例如 user: CurrentUser 表示"我需要当前登录的用户"，FastAPI 会先执行 get_current_user，
# 把结果传进来；如果 get_current_user 抛出异常，接口函数根本不会执行。
# 好处：鉴权逻辑只写一次，每个接口加一个参数就能用，不会有接口忘了鉴权。

# Annotated：给类型附加额外信息，这里用来把"类型"和"依赖"绑在一起
from typing import Annotated

# Depends：声明依赖；HTTPException：抛出后 FastAPI 返回对应的 HTTP 错误；status：状态码常量
from fastapi import Depends, HTTPException, status
# OAuth2PasswordBearer：从请求头 "Authorization: Bearer <token>" 里取出 token
from fastapi.security import OAuth2PasswordBearer
# Session：数据库会话类型
from sqlalchemy.orm import Session

# 每个请求一个数据库会话
from app.db import get_db
# 用户模型
from app.models import User
# 解析 token
from app.security import decode_access_token

# 创建一个"取 token"的依赖。
# tokenUrl 告诉 Swagger 页面去哪个接口登录，这样 /docs 页面右上角的 "Authorize" 按钮就能用了。
# 如果请求头里没有 token，它会自动返回 401。
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# 类型别名：DbSession 等价于"一个 Session，由 get_db 提供"
# 以后接口函数参数写 db: DbSession 就行，不用每次都写 Depends(get_db)
DbSession = Annotated[Session, Depends(get_db)]


# 根据请求里的 token 找出当前用户。任何一步失败都返回 401
def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], db: DbSession) -> User:
    # 先准备好要抛出的异常。所有失败情况都返回同样的错误，不告诉调用方具体是哪里错了
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,  # 401：未认证（没登录或登录已失效）
        detail="Could not validate credentials",
        # HTTP 标准要求 401 响应带上这个头，说明应该用什么方式认证
        headers={"WWW-Authenticate": "Bearer"},
    )
    # 解析 token，拿到用户 id；无效则为 None
    user_id = decode_access_token(token)
    if user_id is None:
        raise unauthorized
    # db.get(模型, 主键)：按主键查一行，查不到返回 None
    user = db.get(User, user_id)
    # 用户不存在（比如被物理删除了），或者已被停用（软删除），token 都不再有效。
    # 这就是为什么每个请求都要查一次数据库，而不是只信任 token 里的内容：
    # token 签发后，在过期前都是有效的，没法"撤回"；查库可以让停用立即生效。
    if user is None or not user.is_active:
        raise unauthorized
    return user


# 类型别名：接口函数参数写 user: CurrentUser，就能拿到当前登录的用户
CurrentUser = Annotated[User, Depends(get_current_user)]
