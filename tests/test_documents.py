# 测试文档接口：上传、校验、幂等、权限隔离、删除

# Path：检查测试临时目录里的文件
from pathlib import Path

# select、func：直接查数据库做验证
from sqlalchemy import func, select

# 读取配置（临时调小文件大小上限）
from app.config import settings
# 模型
from app.models import Document, UsageAction, UsageLog


# 辅助函数：上传一个文件。files= 参数会以 multipart/form-data 格式发送，和浏览器上传文件一样
# 元组的三个元素：(文件名, 文件内容, 内容类型)
def upload(client, headers, filename="note.md", content=b"# hello", content_type="text/markdown"):
    return client.post("/documents", headers=headers, files={"file": (filename, content, content_type)})


# 辅助函数：统计临时存储目录里一共有多少个文件
# rglob("*") 递归列出所有文件和子目录，is_file() 只保留文件
def stored_files(tmp_path: Path) -> list[Path]:
    return [p for p in tmp_path.rglob("*") if p.is_file()]


# --- 上传 --------------------------------------------------------------------


# 测试：上传成功，返回 201，状态是 pending，文件存到了磁盘上
def test_upload_markdown(client, register_and_login, tmp_path):
    headers = register_and_login("alice")
    r = upload(client, headers)
    assert r.status_code == 201
    body = r.json()
    assert body["filename"] == "note.md"
    assert body["status"] == "pending"
    assert body["file_size"] == len(b"# hello")
    # 响应里不能暴露服务器上的存储路径
    assert "storage_path" not in body
    # 磁盘上有且只有一个文件，内容和上传的一致
    files = stored_files(tmp_path)
    assert len(files) == 1
    assert files[0].read_bytes() == b"# hello"


# 测试：PDF 和 TXT 也能上传
def test_upload_pdf_and_txt(client, register_and_login):
    headers = register_and_login("alice")
    assert upload(client, headers, "a.pdf", b"%PDF-1.7 fake body", "application/pdf").status_code == 201
    assert upload(client, headers, "b.txt", "中文内容".encode(), "text/plain").status_code == 201


# 测试：上传成功会记一条 upload 使用记录
def test_upload_writes_usage_log(client, db, register_and_login):
    headers = register_and_login("alice")
    upload(client, headers)
    count = db.scalar(select(func.count()).select_from(UsageLog).where(UsageLog.action == UsageAction.UPLOAD))
    assert count == 1


# 测试：同一用户重复上传相同内容，返回 200 和同一个文档；数据库和磁盘上都只有一份
def test_upload_same_content_is_idempotent(client, db, register_and_login, tmp_path):
    headers = register_and_login("alice")
    first = upload(client, headers, "a.md", b"same content")
    # 换个文件名，内容不变
    second = upload(client, headers, "renamed.md", b"same content")
    assert first.status_code == 201
    assert second.status_code == 200
    # 返回的是同一个文档
    assert second.json()["id"] == first.json()["id"]
    # 数据库里只有一条记录
    assert db.scalar(select(func.count()).select_from(Document)) == 1
    # 磁盘上只有一个文件
    assert len(stored_files(tmp_path)) == 1


# 测试：不同用户上传相同内容，各自得到自己的文档
def test_same_content_different_users(client, register_and_login):
    alice, bob = register_and_login("alice"), register_and_login("bob")
    a = upload(client, alice, content=b"shared")
    b = upload(client, bob, content=b"shared")
    assert a.status_code == b.status_code == 201
    assert a.json()["id"] != b.json()["id"]


# --- 上传校验 ----------------------------------------------------------------


# 测试：不支持的扩展名，返回 415
def test_upload_rejects_unsupported_extension(client, register_and_login, tmp_path):
    headers = register_and_login("alice")
    r = upload(client, headers, "virus.exe", b"MZ...", "application/octet-stream")
    assert r.status_code == 415
    # 被拒绝的文件不能留在磁盘上
    assert stored_files(tmp_path) == []


# 测试：扩展名是 .pdf，但内容不是 PDF（改了后缀的文件），返回 415
def test_upload_rejects_fake_pdf(client, register_and_login):
    headers = register_and_login("alice")
    r = upload(client, headers, "fake.pdf", b"this is not a pdf", "application/pdf")
    assert r.status_code == 415


# 测试：GBK 编码的中文 .txt 也能上传（老的中文 Windows 程序常用这种编码）
def test_upload_accepts_gbk_text(client, register_and_login):
    headers = register_and_login("alice")
    r = upload(client, headers, "gbk.txt", "中文".encode("gbk"), "text/plain")
    assert r.status_code == 201


# 测试：改了后缀的二进制文件（这里是 PNG 图片的开头），返回 415
def test_upload_rejects_binary_as_text(client, register_and_login):
    headers = register_and_login("alice")
    r = upload(client, headers, "image.txt", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "text/plain")
    assert r.status_code == 415


# 测试：空文件，返回 400
def test_upload_rejects_empty_file(client, register_and_login):
    headers = register_and_login("alice")
    assert upload(client, headers, "empty.md", b"").status_code == 400


# 测试：超过大小上限，返回 413
def test_upload_rejects_too_large(client, register_and_login, monkeypatch, tmp_path):
    headers = register_and_login("alice")
    # 把上限临时调成 10 字节，就不用在测试里真的造一个 10MB 的文件
    monkeypatch.setattr(settings, "max_upload_bytes", 10)
    assert upload(client, headers, "ok.md", b"0123456789").status_code == 201  # 正好 10 字节：允许
    assert upload(client, headers, "big.md", b"0123456789X").status_code == 413  # 11 字节：拒绝
    # 只有允许的那个文件存到了磁盘上
    assert len(stored_files(tmp_path)) == 1


# 测试：文件名里带路径（路径穿越攻击），只保留文件名部分，文件仍然存在存储目录里面
def test_upload_path_traversal_filename(client, register_and_login, tmp_path):
    headers = register_and_login("alice")
    r = upload(client, headers, "../../../etc/evil.md", b"# evil")
    assert r.status_code == 201
    # 显示的文件名只剩最后一部分
    assert r.json()["filename"] == "evil.md"
    # 文件确实存在临时存储目录里面，没有跑到外面去
    files = stored_files(tmp_path)
    assert len(files) == 1
    # is_relative_to：判断路径是否在某个目录之下
    assert files[0].resolve().is_relative_to(tmp_path.resolve())


# --- 权限隔离 ----------------------------------------------------------------


# 测试：没登录不能上传、不能看列表
def test_documents_require_login(client):
    assert client.get("/documents").status_code == 401
    assert client.post("/documents", files={"file": ("a.md", b"x", "text/markdown")}).status_code == 401


# 测试：列表只返回自己的文档
def test_list_only_own_documents(client, register_and_login):
    alice, bob = register_and_login("alice"), register_and_login("bob")
    upload(client, alice, "a1.md", b"a1")
    upload(client, alice, "a2.md", b"a2")
    upload(client, bob, "b1.md", b"b1")
    r = client.get("/documents", headers=alice)
    assert r.status_code == 200
    # 集合推导式取出所有文件名
    assert {d["filename"] for d in r.json()} == {"a1.md", "a2.md"}


# 测试：用 A 的 token 查 B 的文档，返回 404（规范里写的是 403，这里选 404 的原因见 routers/documents.py）
def test_cannot_get_other_users_document(client, register_and_login):
    alice, bob = register_and_login("alice"), register_and_login("bob")
    bobs_doc_id = upload(client, bob).json()["id"]
    assert client.get(f"/documents/{bobs_doc_id}", headers=alice).status_code == 404
    # bob 自己可以查到
    assert client.get(f"/documents/{bobs_doc_id}", headers=bob).status_code == 200


# 测试：用 A 的 token 删 B 的文档，返回 404，并且 B 的文档和文件都还在
def test_cannot_delete_other_users_document(client, register_and_login, tmp_path):
    alice, bob = register_and_login("alice"), register_and_login("bob")
    bobs_doc_id = upload(client, bob).json()["id"]
    assert client.delete(f"/documents/{bobs_doc_id}", headers=alice).status_code == 404
    # 文档还在
    assert client.get(f"/documents/{bobs_doc_id}", headers=bob).status_code == 200
    # 文件还在
    assert len(stored_files(tmp_path)) == 1


# --- 删除与分页 ---------------------------------------------------------------


# 测试：删除自己的文档，返回 204，记录和文件都被删掉，再查返回 404
def test_delete_own_document(client, register_and_login, tmp_path):
    headers = register_and_login("alice")
    doc_id = upload(client, headers).json()["id"]
    r = client.delete(f"/documents/{doc_id}", headers=headers)
    assert r.status_code == 204
    assert client.get(f"/documents/{doc_id}", headers=headers).status_code == 404
    assert stored_files(tmp_path) == []


# 测试：删除后可以重新上传相同内容（因为旧记录没了，唯一约束不再冲突）
def test_reupload_after_delete(client, register_and_login):
    headers = register_and_login("alice")
    doc_id = upload(client, headers, content=b"again").json()["id"]
    client.delete(f"/documents/{doc_id}", headers=headers)
    assert upload(client, headers, content=b"again").status_code == 201


# 测试：分页参数生效，并且非法参数返回 422
def test_list_pagination(client, register_and_login):
    headers = register_and_login("alice")
    # 上传 3 个不同内容的文件
    for i in range(3):
        upload(client, headers, f"{i}.md", f"doc {i}".encode())
    # 每页 2 个：第一页 2 个，第二页 1 个
    assert len(client.get("/documents?limit=2", headers=headers).json()) == 2
    assert len(client.get("/documents?limit=2&offset=2", headers=headers).json()) == 1
    # limit 超过 100 或小于 1，都被拒绝
    assert client.get("/documents?limit=101", headers=headers).status_code == 422
    assert client.get("/documents?limit=0", headers=headers).status_code == 422
