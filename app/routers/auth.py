# 认证相关的接口：注册、登录、查看当前用户。

# Annotated：给类型附加依赖信息
from typing import Annotated

# APIRouter：路由分组；Depends：依赖；HTTPException：返回 HTTP 错误；status：状态码
from fastapi import APIRouter, Depends, HTTPException, status
# OAuth2PasswordRequestForm：OAuth2 标准的登录表单，包含 username 和 password 两个字段
from fastapi.security import OAuth2PasswordRequestForm
# IntegrityError：违反数据库约束时抛出的异常
from sqlalchemy.exc import IntegrityError

# 查询函数
from app import queries
# 数据库会话依赖、当前用户依赖
from app.deps import CurrentUser, DbSession
# 用户模型
from app.models import User
# 请求和响应的数据结构
from app.schemas import Token, UserCreate, UserOut
# 密码哈希和 token 签发
from app.security import create_access_token, hash_password, verify_password_or_dummy

# 创建路由分组：
#   prefix="/auth"：这个文件里所有接口的路径都以 /auth 开头
#   tags=["auth"]：在 Swagger 页面里归到 "auth" 这一组
router = APIRouter(prefix="/auth", tags=["auth"])


# POST /auth/register：注册
# response_model=UserOut：返回值会按 UserOut 的字段过滤并转成 JSON（不会带出 password_hash）
# status_code=201：创建成功返回 201 Created，而不是默认的 200
@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def register(body: UserCreate, db: DbSession) -> User:
    # body 已经被 Pydantic 校验过了：用户名格式、邮箱格式、密码长度都合法，用户名和邮箱已转小写
    # 409 Conflict：请求和服务器现有数据冲突（这里是用户名或邮箱已被占用）
    conflict = HTTPException(status.HTTP_409_CONFLICT, "Username or email already registered")

    # 第一道检查：先查一下有没有被占用。能处理绝大多数情况，并返回友好的错误
    if queries.username_or_email_taken(db, body.username, body.email):
        raise conflict

    # 创建用户对象。注意存的是密码的哈希，不是明文
    user = User(username=body.username, email=body.email, password_hash=hash_password(body.password))
    db.add(user)
    try:
        # 提交到数据库
        db.commit()
    except IntegrityError:
        # 第二道检查（并发兜底）：两个人同时注册同一个用户名时，
        # 两个请求都可能通过上面的检查，但数据库的唯一约束只允许一个插入成功。
        # 失败的那个会走到这里：先回滚，再返回 409
        db.rollback()
        raise conflict
    # 返回数据库模型对象，FastAPI 会按 response_model 把它转成 UserOut
    return user


# POST /auth/login：登录，成功返回 token
# 用表单格式（而不是 JSON）接收用户名和密码，这是 OAuth2 标准的要求，
# 这样 Swagger 页面的 "Authorize" 按钮才能直接用这个接口登录
@router.post("/login", response_model=Token)
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()], db: DbSession) -> Token:
    # 注册时用户名被转成了小写，所以登录时也要转小写再查
    user = queries.get_user_by_username(db, form.username.lower())
    # 验证密码。用户不存在时传 None，函数内部仍会做一次假的验证，防止通过响应时间判断用户名是否存在
    password_ok = verify_password_or_dummy(form.password, user.password_hash if user else None)
    # 用户不存在、密码错误、账号已停用：统一返回同一个错误。
    # 如果分别提示"用户不存在"和"密码错误"，攻击者就能借此探测哪些用户名已注册
    if user is None or not password_ok or not user.is_active:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # 验证通过，签发 token
    return Token(access_token=create_access_token(user.id))


# GET /auth/me：查看当前登录的用户
# 参数 user: CurrentUser 会触发鉴权：没有 token 或 token 无效时，直接返回 401，不会执行到函数体
@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> User:
    return user
