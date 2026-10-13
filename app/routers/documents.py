# 文档相关的接口：上传、列表、详情、删除。
# 所有接口都需要登录，并且只能操作自己的文档。

# hashlib：计算文件内容的 SHA-256 哈希
import hashlib
# PurePath：只做路径字符串处理、不访问磁盘的路径类，用来取文件名和扩展名
from pathlib import PurePath
# Annotated：给类型附加额外信息（这里用来给查询参数加校验规则）
from typing import Annotated

# APIRouter：路由分组；HTTPException：返回错误；Query：查询参数的校验规则
# Response：用来动态修改响应状态码；UploadFile：上传的文件；status：状态码常量
from fastapi import APIRouter, HTTPException, Query, Response, UploadFile, status
# IntegrityError：违反数据库约束
from sqlalchemy.exc import IntegrityError

# 查询函数
from app import queries
# 读取配置（文件大小上限）
from app.config import settings
# 依赖：当前用户、数据库会话
from app.deps import CurrentUser, DbSession
# 模型
from app.models import Document, UsageAction, UsageLog
# 响应数据结构
from app.schemas import DocumentOut
# 文件存取
from app.storage import delete_file, save_file

# 所有路径以 /documents 开头，Swagger 里归到 "documents" 组
router = APIRouter(prefix="/documents", tags=["documents"])

# 允许上传的文件扩展名。用集合（set）是因为判断"是否在集合里"非常快
ALLOWED_EXTENSIONS = {".md", ".txt", ".pdf"}


# 读取上传的文件内容，超过大小上限就返回 413
def _read_limited(file: UploadFile) -> bytes:
    # 最多读"上限 + 1"个字节：如果真的读到了这么多，说明文件超过了上限。
    # 不能相信请求头里的 Content-Length，因为客户端可以随便填。
    # 注意：FastAPI 在调用接口函数之前，已经把整个上传内容接收到了临时文件里，
    # 所以这里的检查挡不住"超大文件占用带宽和磁盘"。
    # 真正在入口处拦截要靠 nginx 的 client_max_body_size 配置（阶段 5）。
    content = file.file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        # 413 Content Too Large：请求体太大
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, "File exceeds 10MB limit")
    return content


# 检查文件类型，返回扩展名（小写）。不支持的类型返回 415
def _check_file_type(filename: str, content: bytes) -> str:
    # 415 Unsupported Media Type：不支持的文件类型
    unsupported = HTTPException(
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "Only .md, .txt and .pdf files are supported"
    )
    # 取扩展名并转小写，例如 "笔记.MD" -> ".md"
    extension = PurePath(filename).suffix.lower()
    # 第一步：检查扩展名是否在白名单里
    if extension not in ALLOWED_EXTENSIONS:
        raise unsupported
    # 第二步：检查文件内容是否和扩展名相符。
    # 只看扩展名不可靠，任何文件都可以被改名成 .pdf。
    if extension == ".pdf":
        # PDF 文件的前 5 个字节一定是 "%PDF-"（叫"魔数"，magic number）
        if not content.startswith(b"%PDF-"):
            raise unsupported
    else:
        # .md 和 .txt 应该是 UTF-8 编码的文本。解码失败说明是二进制文件，或者是 GBK 等其他编码
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            raise unsupported
    return extension


# 清理用户上传的文件名，只用于显示，不用于拼接磁盘路径
def _clean_filename(raw: str | None) -> str:
    # 上传时可能没有文件名
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Missing filename")
    # 有些浏览器会把完整路径发过来，例如 "C:\Users\me\笔记.md"。
    # 先把反斜杠统一换成斜杠，再用 PurePath(...).name 只取最后的文件名部分
    name = PurePath(raw.replace("\\", "/")).name
    # 截断到 255 个字符，和数据库字段长度一致，否则插入时会报错
    return name[:255]


# POST /documents：上传文档
# 返回 201 表示新建了文档；如果同一个用户已上传过相同内容，返回 200 和已有的文档（幂等）
@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def upload_document(file: UploadFile, response: Response, user: CurrentUser, db: DbSession) -> Document:
    # ---------- 第 1 步：校验（全部在内存里完成，还没有任何副作用） ----------
    filename = _clean_filename(file.filename)
    content = _read_limited(file)
    # 空文件没有意义，直接拒绝
    if not content:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "File is empty")
    extension = _check_file_type(filename, content)
    # 计算内容哈希。内容相同，哈希一定相同
    content_hash = hashlib.sha256(content).hexdigest()

    # ---------- 第 2 步：查重（幂等） ----------
    # 同一用户已经传过相同内容：直接返回已有文档，不再存文件、不再写数据库
    existing = queries.get_document_by_hash(db, user.id, content_hash)
    if existing is not None:
        # 修改响应状态码为 200（默认是装饰器里写的 201）
        response.status_code = status.HTTP_200_OK
        return existing

    # ---------- 第 3 步：先存文件 ----------
    # 为什么先存文件再写数据库：
    #   如果反过来，数据库里有了记录但文件存失败，后续处理时会找不到文件（错误数据）；
    #   先存文件的话，最坏情况是磁盘上多一个没有记录的文件（垃圾数据），危害小得多，而且下面会清理
    storage_path = save_file(user.id, extension, content)

    # ---------- 第 4 步：写数据库（一个事务） ----------
    doc = Document(
        user_id=user.id,
        filename=filename,
        storage_path=storage_path,
        file_size=len(content),
        content_hash=content_hash,
        # status 不用写，默认就是 pending
    )
    db.add(doc)
    # 同时记一条使用记录。它和文档记录在同一个事务里：要么都写入，要么都不写入
    db.add(UsageLog(user_id=user.id, action=UsageAction.UPLOAD))
    try:
        db.commit()
    except IntegrityError:
        # 并发情况：两个相同的上传请求同时到达，都通过了第 2 步的查重，
        # 但唯一约束 (user_id, content_hash) 只允许一个插入成功。失败的那个走到这里。
        db.rollback()
        # 删掉自己刚存的文件。因为文件名是随机的，不会误删另一个请求的文件
        delete_file(storage_path)
        # 查出另一个请求创建的文档，返回给用户，效果和第 2 步一样
        existing = queries.get_document_by_hash(db, user.id, content_hash)
        if existing is None:
            # 理论上不会发生（唯一约束冲突说明一定存在），保险起见把原始异常抛出去
            raise
        response.status_code = status.HTTP_200_OK
        return existing
    except Exception:
        # 其他任何数据库错误（比如数据库断开）：回滚，并删掉已经存好的文件，不留垃圾
        db.rollback()
        delete_file(storage_path)
        # 不带参数的 raise：把原来的异常继续往外抛，FastAPI 会返回 500
        raise

    # ---------- 第 5 步（阶段 3）：投递后台解析任务 ----------
    # 必须放在 commit 之后：否则 worker 可能在事务提交前就去查这个文档，查不到
    return doc


# GET /documents：当前用户的文档列表（分页）
# limit 和 offset 是查询参数，例如 /documents?limit=10&offset=20
# Query(ge=1, le=100)：ge 是"大于等于"，le 是"小于等于"。限制 limit 最大 100，防止一次查太多
@router.get("", response_model=list[DocumentOut])
def list_documents(
    user: CurrentUser,
    db: DbSession,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[Document]:
    # 查询函数内部强制按 user.id 过滤，只会返回自己的文档
    return queries.list_documents(db, user.id, limit=limit, offset=offset)


# GET /documents/{document_id}：文档详情
# 路径里的 {document_id} 会自动转成 int 传给同名参数；不是数字时 FastAPI 返回 422
@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: int, user: CurrentUser, db: DbSession) -> Document:
    doc = queries.get_document(db, user.id, document_id)
    # 文档不存在、或者是别人的，都返回 404。
    # 不返回 403 的原因：403 等于告诉对方"这个 id 存在，只是不是你的"，攻击者可以借此探测
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return doc


# DELETE /documents/{document_id}：删除文档
# 204 No Content：删除成功，响应没有内容
@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: int, user: CurrentUser, db: DbSession) -> None:
    # 先按"当前用户"查，查不到（不存在或不是自己的）就返回 404，保证不能删别人的
    doc = queries.get_document(db, user.id, document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    # 先记下文件路径，删除记录后就拿不到了
    storage_path = doc.storage_path
    # 删除数据库记录。它的 chunks 会被数据库的 ON DELETE CASCADE 自动删除
    db.delete(doc)
    db.commit()
    # 提交成功后再删文件。顺序很重要：
    #   如果先删文件、后提交失败，数据库里就有一条指向不存在文件的记录（错误数据）；
    #   现在的顺序下，最坏情况是文件删除失败，磁盘上留一个没人引用的文件（垃圾数据），危害小
    delete_file(storage_path)
