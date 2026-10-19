# 测试 Agent（app/agent.py）：工具的权限隔离、参数出错时的处理、循环一定会停、引用编号

# json：构造工具参数
import json

# pytest：参数化测试
import pytest
# select：查用户 id
from sqlalchemy import select

# 被测试的模块
from app import agent, llm
# 模型
from app.models import User
# 复用问答测试里的辅助函数：上传并处理文档、提问
from tests.test_chat import add_ready_doc, ask


# 按用户名查用户 id
def user_id(db, username: str) -> int:
    return db.scalar(select(User.id).where(User.username == username))


# 构造一条"模型要调用工具"的回复
def tool_call(name: str, **args) -> llm.ChatReply:
    return llm.ChatReply("", [llm.ToolCall(f"call-{name}", name, json.dumps(args))], 10)


# 构造一条"模型直接回答"的回复
def answer(text: str) -> llm.ChatReply:
    return llm.ChatReply(text, [], 10)


# ============================ 工具 ============================


# 测试（最重要）：三个工具都只能看到自己的文档，传入别人的文档 id 也查不到
def test_tools_only_see_own_documents(client, db, register_and_login, enqueued):
    alice, bob = register_and_login("alice"), register_and_login("bob")
    add_ready_doc(client, db, alice, enqueued, "notes.md", "alice notes")
    bob_doc = add_ready_doc(client, db, bob, enqueued, "secret.md", "bob secret salary")
    tools = agent.Agent(db, user_id(db, "alice"))
    listed = tools.call_tool("list_documents", "{}")
    assert "notes.md" in listed and "secret.md" not in listed
    assert tools.call_tool("read_document", json.dumps({"document_id": bob_doc})) == "找不到这个文档，或者它还没有处理完成。"
    search = json.dumps({"query": "bob secret salary", "document_id": bob_doc})
    assert tools.call_tool("search_documents", search) == "没有找到相关段落。"


# 测试：search_documents 指定 document_id 时只在这个文档里检索
def test_search_can_be_limited_to_one_document(client, db, register_and_login, enqueued):
    headers = register_and_login("alice")
    first = add_ready_doc(client, db, headers, enqueued, "a.md", "tea notes from china")
    add_ready_doc(client, db, headers, enqueued, "b.md", "tea notes from japan")
    tools = agent.Agent(db, user_id(db, "alice"))
    result = tools.call_tool("search_documents", json.dumps({"query": "tea notes", "document_id": first}))
    assert "a.md" in result and "b.md" not in result


# 测试：read_document 按顺序读取；超长的文档只读前 READ_MAX_CHUNKS 段，并告诉模型后面还有内容
def test_read_document_is_capped(client, db, register_and_login, enqueued, monkeypatch):
    monkeypatch.setattr(agent, "READ_MAX_CHUNKS", 2)
    headers = register_and_login("alice")
    # 5000 个字符，会被切成 10 段以上
    doc_id = add_ready_doc(client, db, headers, enqueued, "long.md", "word " * 1000)
    result = agent.Agent(db, user_id(db, "alice")).call_tool("read_document", json.dumps({"document_id": doc_id}))
    assert result.count("（来自 long.md") == 2
    assert "只读取了前 2 段" in result


# 测试：参数格式不对、缺参数、工具名不存在时，把错误告诉模型让它改正，而不是让接口报 500
@pytest.mark.parametrize(
    "name, arguments",
    [
        ("search_documents", "not json"),  # 不是 JSON
        ("search_documents", "{}"),  # 缺少必填的 query
        ("read_document", '{"document_id": "abc"}'),  # id 不是数字
        ("read_document", "[1]"),  # 参数不是对象
        ("delete_everything", "{}"),  # 没有这个工具
    ],
    ids=["not-json", "missing", "not-int", "not-object", "unknown"],
)
def test_bad_tool_calls_are_reported_to_the_model(db, make_user, name, arguments):
    result = agent.Agent(db, make_user().id).call_tool(name, arguments)
    assert result.startswith(("参数错误", "没有名为"))


# ============================ 循环 ============================


# 测试：打招呼时模型不调用工具，直接回答，只调用一次模型，没有来源
def test_answer_without_tools(client, register_and_login, fake_llm):
    headers = register_and_login("alice")
    fake_llm.script = [answer("你好！")]
    r = ask(client, headers, "你好")
    assert r.json()["answer"] == "你好！"
    assert r.json()["sources"] == []
    assert len(fake_llm.chat_calls) == 1


# 测试："我有几个文件"这类问题：模型调用 list_documents，拿到的是文件列表，而不是检索出来的段落
def test_list_documents_answers_file_questions(client, db, register_and_login, enqueued, fake_llm):
    headers = register_and_login("alice")
    add_ready_doc(client, db, headers, enqueued, "a.md", "first")
    add_ready_doc(client, db, headers, enqueued, "b.md", "second")
    fake_llm.script = [tool_call("list_documents"), answer("你有 2 个文件")]
    assert ask(client, headers, "我有几个文件？").json()["answer"] == "你有 2 个文件"
    tool_result = fake_llm.chat_calls[1][-1]["content"]
    assert "共 2 个文档" in tool_result and "a.md" in tool_result and "b.md" in tool_result


# 测试：模型一直要调用工具也会停：最多 MAX_ROUNDS 轮工具，最后一次不许再调用工具，必须回答
def test_loop_stops_after_max_rounds(client, register_and_login, monkeypatch):
    headers = register_and_login("alice")
    # 记录每次调用时是否允许调用工具
    allowed: list[bool] = []

    # 一个"停不下来"的模型：只要允许，就一直调用工具
    def endless(messages, tools=None, allow_tools=True):
        allowed.append(allow_tools)
        return tool_call("list_documents") if allow_tools else answer("final")

    monkeypatch.setattr(llm, "chat", endless)
    assert ask(client, headers, "loop").json()["answer"] == "final"
    assert allowed == [True] * agent.MAX_ROUNDS + [False]


# ============================ 引用 ============================


# 测试：多轮检索的编号连续，同一段再次被检索到时编号不变；来源只包含回答里实际引用了的段落
def test_citations_across_rounds(client, db, register_and_login, enqueued, fake_llm):
    headers = register_and_login("alice")
    add_ready_doc(client, db, headers, enqueued, "tea.md", "green tea comes from china")
    add_ready_doc(client, db, headers, enqueued, "car.md", "change engine oil every 5000 km")
    fake_llm.script = [
        tool_call("search_documents", query="green tea china"),  # 检索到茶的段落 → [1]
        tool_call("search_documents", query="engine oil"),  # 检索到机油的段落 → [2]
        tool_call("search_documents", query="green tea china"),  # 又检索到茶的段落 → 仍然是 [1]
        answer("change the oil [2]. see also [9]"),  # [9] 不存在，应该被忽略
    ]
    r = ask(client, headers, "question")
    assert fake_llm.chat_calls[3][-1]["content"].startswith("[1] （来自 tea.md")
    assert [s["filename"] for s in r.json()["sources"]] == ["car.md"]


# 测试：通读得到的段落作为来源时没有相似度（score 为 null）
def test_read_document_sources_have_no_score(client, db, register_and_login, enqueued, fake_llm):
    headers = register_and_login("alice")
    doc_id = add_ready_doc(client, db, headers, enqueued, "a.md", "short note")
    fake_llm.script = [tool_call("read_document", document_id=doc_id), answer("summary [1]")]
    source = ask(client, headers, "summarize a.md").json()["sources"][0]
    assert source["filename"] == "a.md" and source["score"] is None
