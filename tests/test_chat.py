# 测试问答链路：检索、权限隔离、会话与多轮对话、配额、缓存、外部服务失败时的降级

# json：构造工具参数
import json
# Decimal：费用是精确的十进制小数
from decimal import Decimal

# pytest：approx 比较浮点数
import pytest
# select、func：直接查数据库做验证
from sqlalchemy import func, select

# 被测试的模块
from app import llm, processing, queries, rag
# 模型
from app.models import Conversation, Message, MessageRole, UsageAction, UsageLog, User
# 假 embedding：测试里直接算问题的向量
from tests.conftest import fake_embedding


# ============================ 辅助函数 ============================


# 上传一个文档并立刻在当前进程里处理完（模拟 worker），返回文档 id
def add_ready_doc(client, db, headers, enqueued, filename: str, content: str) -> int:
    r = client.post("/documents", headers=headers, files={"file": (filename, content.encode(), "text/markdown")})
    doc_id = r.json()["id"]
    # enqueued[-1] 是刚刚投递的文档 id，就是这个文档
    processing.process_document(db, enqueued[-1])
    return doc_id


# 提问
def ask(client, headers, question: str, conversation_id: int | None = None):
    return client.post("/chat", headers=headers, json={"question": question, "conversation_id": conversation_id})


# 把某个用户的每日配额改成 n
def set_quota(db, username: str, n: int) -> None:
    db.scalar(select(User).where(User.username == username)).daily_quota = n
    db.commit()


# ============================ 检索 ============================


# 测试：处理文档时，每个段落都存了 1024 维的向量
def test_processing_stores_embeddings(client, db, register_and_login, enqueued):
    headers = register_and_login("alice")
    doc_id = add_ready_doc(client, db, headers, enqueued, "a.md", "apples are red")
    chunk = db.scalar(select(processing.Chunk).where(processing.Chunk.document_id == doc_id))
    assert len(chunk.embedding) == 1024


# 测试：检索结果按相似度排序，最相关的在前
def test_search_returns_most_similar_first(client, db, register_and_login, enqueued):
    headers = register_and_login("alice")
    add_ready_doc(client, db, headers, enqueued, "fruit.md", "apples and bananas are fruit")
    add_ready_doc(client, db, headers, enqueued, "cars.md", "engines wheels and brakes")
    user_id = db.scalar(select(User.id).where(User.username == "alice"))
    results = queries.search_chunks(db, user_id, fake_embedding("tell me about engines and brakes"), top_k=2)
    assert [r.filename for r in results] == ["cars.md", "fruit.md"]
    # 相似度从大到小
    assert results[0].score > results[1].score


# 测试（最重要）：检索只会搜到自己的文档，哪怕别人的文档和问题完全一样
def test_search_never_returns_other_users_chunks(client, db, register_and_login, enqueued):
    alice, bob = register_and_login("alice"), register_and_login("bob")
    add_ready_doc(client, db, bob, enqueued, "secret.md", "bob salary is 50000")
    add_ready_doc(client, db, alice, enqueued, "notes.md", "salary notes")
    # 这个问题和 bob 的文档也有一定相似度（高于最低分数线），如果没有权限过滤，bob 的文档会出现在来源里
    r = ask(client, alice, "salary notes")
    assert [s["filename"] for s in r.json()["sources"]] == ["notes.md"]


# 测试：还没处理完（不是 ready）的文档不会被检索到
def test_search_skips_documents_that_are_not_ready(client, db, register_and_login):
    headers = register_and_login("alice")
    # 只上传，不处理：状态是 pending，也没有段落
    client.post("/documents", headers=headers, files={"file": ("a.md", b"pending doc", "text/markdown")})
    user_id = db.scalar(select(User.id).where(User.username == "alice"))
    assert queries.search_chunks(db, user_id, fake_embedding("pending doc"), top_k=5) == []


# ============================ 问答与会话 ============================


# 测试：第一次提问会新建会话，保存提问和回答，回答带来源，记一条 query 使用记录
def test_first_question_creates_conversation(client, db, register_and_login, enqueued, fake_llm):
    headers = register_and_login("alice")
    doc_id = add_ready_doc(client, db, headers, enqueued, "tea.md", "green tea comes from china")
    r = ask(client, headers, "where does green tea come from?")
    assert r.status_code == 200
    body = r.json()
    # 假的模型引用了检索到的第 1 段
    assert body["answer"] == "fake answer [1]"
    assert body["sources"][0]["document_id"] == doc_id
    assert body["sources"][0]["snippet"] == "green tea comes from china"
    # 第 1 轮：问题是最后一条消息；第 2 轮：最后一条是检索工具的结果，里面有带编号的资料
    assert fake_llm.chat_calls[0][-1]["content"] == "where does green tea come from?"
    tool_result = fake_llm.chat_calls[1][-1]
    assert tool_result["role"] == "tool" and "[1] （来自 tea.md" in tool_result["content"]
    # 数据库：会话标题取自问题，里面有一问一答两条消息
    conversation = db.get(Conversation, body["conversation_id"])
    assert conversation.title == "where does green tea come from?"
    roles = db.scalars(select(Message.role).where(Message.conversation_id == conversation.id)).all()
    assert [r.value for r in roles] == ["user", "assistant"]
    assert db.scalar(select(func.count()).select_from(UsageLog).where(UsageLog.action == UsageAction.QUERY)) == 1


# 测试：追问时，之前的问答作为历史发给 LLM（多轮对话）
def test_follow_up_sends_history(client, db, register_and_login, enqueued, fake_llm):
    headers = register_and_login("alice")
    add_ready_doc(client, db, headers, enqueued, "tea.md", "green tea comes from china")
    conversation_id = ask(client, headers, "where does green tea come from?").json()["conversation_id"]
    # 追问要和文档有足够多相同的词，否则相似度低于分数线，不会调用 LLM
    ask(client, headers, "where does black tea come from?", conversation_id)
    # 每次提问调用 2 次模型（检索、回答），所以第二个问题的第一次调用是 chat_calls[2]
    second_call = fake_llm.chat_calls[2]
    # 消息顺序：系统提示词、上一轮的提问、上一轮的回答、这次的问题。历史里不含上一轮的工具调用过程
    assert [m["role"] for m in second_call] == ["system", "user", "assistant", "user"]
    assert second_call[1]["content"] == "where does green tea come from?"
    # 同一个会话里现在有 4 条消息
    assert len(client.get(f"/conversations/{conversation_id}/messages", headers=headers).json()) == 4


# 测试：每次问答返回所有轮次加起来的 token 数和费用，并记录在回答消息和使用记录里
def test_answer_reports_and_records_usage(client, db, register_and_login, enqueued, fake_llm):
    headers = register_and_login("alice")
    add_ready_doc(client, db, headers, enqueued, "tea.md", "green tea comes from china")
    # 两轮：先检索，再回答。参数依次是：内容、工具调用、总 token 数、输入 token 数、费用
    search = llm.ToolCall("call-1", "search_documents", json.dumps({"query": "green tea"}))
    fake_llm.script = [
        llm.ChatReply("", [search], 100, 90, Decimal("0.001")),
        llm.ChatReply("green tea [1]", [], 50, 30, Decimal("0.002")),
    ]
    body = ask(client, headers, "where does green tea come from?").json()
    assert body["tokens"] == 150
    assert body["cost"] == pytest.approx(0.003)
    answer = db.scalar(select(Message).where(Message.role == MessageRole.ASSISTANT))
    assert (answer.tokens_used, answer.cost) == (150, Decimal("0.003"))
    assert db.scalar(select(UsageLog.cost).where(UsageLog.action == UsageAction.QUERY)) == Decimal("0.003")
    # 历史消息里也能看到：用户的提问没有费用，回答有
    messages = client.get(f"/conversations/{body['conversation_id']}/messages", headers=headers).json()
    assert [(m["tokens_used"], m["cost"]) for m in messages] == [(0, None), (150, pytest.approx(0.003))]


# 测试：不能往别人的会话里提问，也不能看、不能删别人的会话
def test_cannot_use_other_users_conversation(client, db, register_and_login, enqueued):
    alice, bob = register_and_login("alice"), register_and_login("bob")
    add_ready_doc(client, db, bob, enqueued, "b.md", "bob notes")
    bobs_conversation = ask(client, bob, "bob notes").json()["conversation_id"]
    assert ask(client, alice, "hi", bobs_conversation).status_code == 404
    assert client.get(f"/conversations/{bobs_conversation}/messages", headers=alice).status_code == 404
    assert client.delete(f"/conversations/{bobs_conversation}", headers=alice).status_code == 404


# 测试：相似度低于分数线的段落不算来源；全都不相关时，检索工具告诉模型"没有找到"
def test_irrelevant_chunks_are_dropped(client, db, register_and_login, enqueued, fake_llm):
    headers = register_and_login("alice")
    add_ready_doc(client, db, headers, enqueued, "tea.md", "green tea comes from china")
    add_ready_doc(client, db, headers, enqueued, "car.md", "change engine oil every 5000 km")
    r = ask(client, headers, "where does green tea come from")
    assert [s["filename"] for s in r.json()["sources"]] == ["tea.md"]
    r = ask(client, headers, "what is the python gil")
    assert r.json()["sources"] == []
    assert fake_llm.chat_calls[-1][-1]["content"] == "没有找到相关段落。"


# 测试：没有可检索的文档时，回答没有来源
def test_no_documents_gives_no_sources(client, register_and_login, fake_llm):
    headers = register_and_login("alice")
    r = ask(client, headers, "anything?")
    assert r.json() == {
        "conversation_id": r.json()["conversation_id"],
        "answer": "fake answer",
        "sources": [],
        # 假的模型：检索一轮 10 个 token、回答一轮 42 个 token，没有设置费用
        "tokens": 52,
        "cost": 0.0,
    }


# 测试：会话列表按最近活跃排序；往旧会话里追问后，它会排到最前面（updated_at 被更新）
def test_conversation_list_orders_by_last_activity(client, db, register_and_login, enqueued):
    headers = register_and_login("alice")
    add_ready_doc(client, db, headers, enqueued, "a.md", "some notes")
    first = ask(client, headers, "first question").json()["conversation_id"]
    second = ask(client, headers, "second question").json()["conversation_id"]
    assert [c["id"] for c in client.get("/conversations", headers=headers).json()] == [second, first]
    ask(client, headers, "follow up", first)
    listed = client.get("/conversations", headers=headers).json()
    assert [c["id"] for c in listed] == [first, second]
    # 每个会话都带着最后一条消息（AI 的回答）
    assert listed[0]["last_message"]["role"] == "assistant"


# 测试：删除会话后，它的消息也一起被删除
def test_delete_conversation(client, db, register_and_login, enqueued):
    headers = register_and_login("alice")
    add_ready_doc(client, db, headers, enqueued, "a.md", "some notes")
    conversation_id = ask(client, headers, "question").json()["conversation_id"]
    assert client.delete(f"/conversations/{conversation_id}", headers=headers).status_code == 204
    assert db.scalar(select(func.count()).select_from(Message).where(Message.conversation_id == conversation_id)) == 0


# ============================ 配额 ============================


# 测试：配额用完后返回 429，而且不会再调用 LLM
def test_quota_exceeded_returns_429(client, db, register_and_login, enqueued, fake_llm):
    headers = register_and_login("alice")
    add_ready_doc(client, db, headers, enqueued, "a.md", "some notes")
    set_quota(db, "alice", 2)
    assert ask(client, headers, "notes 1").status_code == 200
    assert ask(client, headers, "notes 2").status_code == 200
    assert ask(client, headers, "notes 3").status_code == 429
    # 前两个问题各调用 2 次模型，第三个问题一次都没有调用
    assert len(fake_llm.chat_calls) == 4


# 测试：配额是按用户分开算的，alice 用完了不影响 bob
def test_quota_is_per_user(client, db, register_and_login):
    alice, bob = register_and_login("alice"), register_and_login("bob")
    set_quota(db, "alice", 1)
    ask(client, alice, "q")
    assert ask(client, alice, "q").status_code == 429
    assert ask(client, bob, "q").status_code == 200


# 测试：Redis 挂了时，改用数据库的 usage_logs 计数，配额照样生效（降级而不是全部拒绝）
def test_quota_falls_back_to_database_when_redis_is_down(client, db, register_and_login, fake_redis):
    headers = register_and_login("alice")
    set_quota(db, "alice", 2)
    # 模拟 Redis 断开
    fake_redis.connected = False
    assert ask(client, headers, "q1").status_code == 200
    assert ask(client, headers, "q2").status_code == 200
    assert ask(client, headers, "q3").status_code == 429


# ============================ 外部服务失败时的降级 ============================


# 测试：LLM 挂了，返回 503 友好提示（不是 500），退还配额，什么都不保存
def test_llm_outage_returns_503_and_refunds_quota(client, db, register_and_login, enqueued, fake_llm):
    headers = register_and_login("alice")
    add_ready_doc(client, db, headers, enqueued, "a.md", "some notes")
    set_quota(db, "alice", 1)
    fake_llm.fail = True
    r = ask(client, headers, "some notes")
    assert r.status_code == 503
    assert "temporarily unavailable" in r.json()["detail"]
    # 没有保存任何会话和消息
    assert db.scalar(select(func.count()).select_from(Conversation)) == 0
    # 服务恢复后，唯一的一次配额还能用（之前失败的那次被退还了）
    fake_llm.fail = False
    assert ask(client, headers, "some notes").status_code == 200


# ============================ 缓存 ============================


# 测试：同一个问题第二次提问时，问题的向量从 Redis 缓存里取，不再调用 embedding 服务
def test_question_embedding_is_cached(fake_llm):
    rag.embed_query("what is rag?")
    rag.embed_query("what is rag?")
    assert fake_llm.embedded.count("what is rag?") == 1


# 测试：Redis 挂了时，缓存失效但问答照常进行
def test_embedding_cache_survives_redis_outage(fake_llm, fake_redis):
    fake_redis.connected = False
    assert len(rag.embed_query("what is rag?")) == 1024
