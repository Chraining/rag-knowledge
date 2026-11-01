# 测试对话摘要（app/summary.py，以及 app/tasks.py 的 summarize_conversation）：
# 什么时候摘要、压缩哪些消息、摘要怎么发给模型、后台任务怎么更新会话

# contextlib.nullcontext：把测试的 db 会话包装成 SessionLocal() 的样子
import contextlib
# Decimal：费用是精确的十进制小数
from decimal import Decimal

# pytest：fixture
import pytest
# select、func、update：直接查、改数据库
from sqlalchemy import func, select, update

# 被测试的模块
from app import agent, llm, summary, tasks
# 模型
from app.models import Conversation, Message, MessageRole, UsageAction, UsageLog
# 复用的辅助函数：构造历史、构造模型的回答、提问
from tests.test_agent import answer, make_history
from tests.test_chat import ask


# 让 tasks.py 使用测试的 db 会话，而不是连开发数据库（和 test_processing.py 一样）
@pytest.fixture
def task_db(db, monkeypatch):
    monkeypatch.setattr(tasks, "SessionLocal", lambda: contextlib.nullcontext(db))
    return db


# 一个有两组问答的会话，历史预算调得很小，所以第一组问答需要被压缩。返回会话 id
@pytest.fixture
def long_conversation(client, register_and_login, monkeypatch) -> int:
    monkeypatch.setattr(agent, "HISTORY_TOKEN_BUDGET", 5)
    headers = register_and_login("alice")
    conversation_id = ask(client, headers, "first question").json()["conversation_id"]
    ask(client, headers, "second question", conversation_id)
    return conversation_id


# ============================ 压缩哪些消息 ============================


# 测试：没超出预算不摘要；超出后压缩最早的消息，只留下最近一半预算以内的
def test_fold_oldest_messages_beyond_budget(monkeypatch):
    monkeypatch.setattr(agent, "HISTORY_TOKEN_BUDGET", 100)
    # 每条 50 字 ≈ 31 token，3 条 ≈ 93，没超出
    assert summary.messages_to_fold(make_history(50, 50, 50)) == 0
    # 100 字 ≈ 62 token，20 字 ≈ 12 token，一共 ≈ 161，超出了。
    # 留下 50 token 以内：最近 3 条 ≈ 37，再加一条就超出，所以压缩最早的 2 条
    assert summary.messages_to_fold(make_history(100, 100, 20, 20, 20)) == 2
    # 最后一组问答超出了一半预算也要保留（和 recent_history 一样）
    assert summary.messages_to_fold(make_history(100, 100, 100, 100)) == 2


# 测试：摘要时把旧摘要和要压缩的对话一起发给模型（增量合并，不从头重新摘要）
def test_summarize_merges_old_summary(fake_llm):
    fake_llm.script = [answer("新摘要")]
    messages = [Message(role=MessageRole.USER, content="只用中文回答"), Message(role=MessageRole.ASSISTANT, content="好的")]
    assert summary.summarize("旧摘要", messages).content == "新摘要"
    system, user = fake_llm.chat_calls[0]
    assert system["content"] == summary.SUMMARY_PROMPT
    assert "旧摘要" in user["content"]
    assert "用户：只用中文回答" in user["content"] and "助手：好的" in user["content"]


# ============================ 问答接口 ============================


# 测试：有摘要时，摘要紧跟在系统提示词后面，历史只包括还没被摘要的消息
def test_summary_replaces_summarized_messages(client, db, register_and_login, fake_llm):
    headers = register_and_login("alice")
    conversation_id = ask(client, headers, "first question").json()["conversation_id"]
    ask(client, headers, "second question", conversation_id)
    # 假设第一组问答已经被摘要了
    db.execute(update(Conversation).where(Conversation.id == conversation_id).values(summary="用户要求只用中文回答", summarized_count=2))
    fake_llm.chat_calls.clear()
    ask(client, headers, "third question", conversation_id)
    first_call = fake_llm.chat_calls[0]
    assert [m["role"] for m in first_call] == ["system", "system", "user", "assistant", "user"]
    assert first_call[1]["content"] == agent.SUMMARY_INTRO + "用户要求只用中文回答"
    assert first_call[2]["content"] == "second question"


# 测试：加上这一问一答之后，历史超出预算并且有可以压缩的消息，才投递摘要任务
def test_chat_enqueues_summary_when_over_budget(client, register_and_login, summaries_enqueued, monkeypatch):
    monkeypatch.setattr(agent, "HISTORY_TOKEN_BUDGET", 5)
    headers = register_and_login("alice")
    conversation_id = ask(client, headers, "first question").json()["conversation_id"]
    # 只有一组问答：超出了预算，但最后一组问答一定保留，没有可以压缩的
    assert summaries_enqueued == []
    ask(client, headers, "second question", conversation_id)
    assert summaries_enqueued == [conversation_id]


# ============================ 后台任务 ============================


# 测试：后台任务把最早的消息合并进摘要，更新 summarized_count，并记录这次调用的用量
def test_task_folds_messages_into_summary(task_db, long_conversation, fake_llm):
    fake_llm.script = [llm.ChatReply("摘要", [], 30, 20, Decimal("0.001"))]
    tasks.summarize_conversation(long_conversation)
    conversation = task_db.get(Conversation, long_conversation)
    assert (conversation.summary, conversation.summarized_count) == ("摘要", 2)
    log = task_db.scalar(select(UsageLog).where(UsageLog.action == UsageAction.SUMMARY))
    assert (log.tokens_used, log.cost) == (30, Decimal("0.001"))
    # 再运行一次：剩下的一组问答不需要压缩，不再调用模型
    calls = len(fake_llm.chat_calls)
    tasks.summarize_conversation(long_conversation)
    assert len(fake_llm.chat_calls) == calls


# 测试："比较后再写入"：调用模型期间，另一个任务已经写入了摘要，这个任务就放弃，不覆盖
def test_task_skips_when_another_task_finished_first(task_db, long_conversation, monkeypatch):
    def concurrent(messages, tools=None, tool_choice="auto"):
        # 模拟：这个任务在调用大模型的时候，另一个任务抢先写入了摘要
        task_db.execute(
            update(Conversation).where(Conversation.id == long_conversation).values(summary="先写入的摘要", summarized_count=2)
        )
        return llm.ChatReply("过时的摘要", [], 30, 20, Decimal("0.001"))

    monkeypatch.setattr(llm, "chat", concurrent)
    tasks.summarize_conversation(long_conversation)
    assert task_db.get(Conversation, long_conversation).summary == "先写入的摘要"
    # 摘要没用上，但钱已经花了，仍然记录
    count = select(func.count()).select_from(UsageLog).where(UsageLog.action == UsageAction.SUMMARY)
    assert task_db.scalar(count) == 1


# 测试：大模型接口失败时，任务直接结束，会话不变，下次提问时会再投递
def test_task_gives_up_when_llm_is_down(task_db, long_conversation, fake_llm):
    fake_llm.fail = True
    tasks.summarize_conversation(long_conversation)
    conversation = task_db.get(Conversation, long_conversation)
    assert (conversation.summary, conversation.summarized_count) == (None, 0)
