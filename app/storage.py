# 文件存储：把上传的文件保存到磁盘、从磁盘删除。
# 单独放一个模块，以后要换成阿里云 OSS 等对象存储时，只需要改这一个文件。

# uuid：生成全局唯一的随机 id
import uuid
# Path：面向对象的文件路径操作，自动处理 Windows（\）和 Linux（/）路径分隔符的差异
from pathlib import Path

# 读取配置（存储目录）
from app.config import settings


# 把相对路径（数据库里存的）转换成磁盘上的完整路径
def _full_path(relative_path: str) -> Path:
    # Path 对象可以用 / 运算符拼接路径，例如 Path("storage") / "1/abc.md" -> storage/1/abc.md
    return Path(settings.storage_dir) / relative_path


# 保存文件，返回相对路径（存进数据库的 storage_path 字段）
def save_file(user_id: int, extension: str, content: bytes) -> str:
    # 文件名用随机的 uuid，而不是用户上传时的原始文件名。原因：
    #   1. 安全：用户可以把文件名设成 "../../app/main.py"，如果直接拿来拼路径，
    #      就能把文件写到存储目录以外的任意位置（"路径穿越"攻击）
    #   2. 不冲突：两个用户都上传 "笔记.md" 不会互相覆盖
    #   3. 避免奇怪字符：原始文件名里可能有空格、中文、特殊符号，在某些系统上会出问题
    # uuid4().hex 生成 32 位随机十六进制字符串
    relative_path = f"{user_id}/{uuid.uuid4().hex}{extension}"
    path = _full_path(relative_path)
    # 创建用户的目录。parents=True：上级目录不存在时一起创建；exist_ok=True：已存在不报错
    path.parent.mkdir(parents=True, exist_ok=True)
    # 以二进制方式写入文件内容
    path.write_bytes(content)
    return relative_path


# 删除文件。文件不存在时什么也不做（不报错）
def delete_file(relative_path: str) -> None:
    # missing_ok=True：文件不存在时不抛异常。
    # 删除操作因此是"幂等"的：删一次和删多次结果一样，重试时不会出错
    _full_path(relative_path).unlink(missing_ok=True)
