# 测试前端页面的挂载


# 测试：打开根路径 / 返回前端页面
def test_root_serves_frontend(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "RAG 知识库" in r.text


# 测试：前端挂在 / 上，但不会抢走接口的请求（接口仍然要求登录）
def test_api_routes_still_take_priority(client):
    assert client.get("/documents").status_code == 401
