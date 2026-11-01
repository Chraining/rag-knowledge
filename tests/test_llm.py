# 测试费用计算（app/llm.py 的 chat_cost、is_peak）：三种价格、高峰和低谷时段、时区换算

# datetime：构造调用时间
from datetime import datetime, timezone
# Decimal：费用是精确的十进制小数，可以直接用 == 比较
from decimal import Decimal

# pytest：参数化测试
import pytest

# 被测试的模块
from app import llm
# 北京时间
from app.quota import BEIJING

# 2026-11-02 是周一，2026-11-07 是周六
MONDAY_10AM = datetime(2026, 11, 2, 10, 0, tzinfo=BEIJING)


# 测试：高峰时段按配置的价格算，命中缓存的输入、没命中的输入、输出各是一种价格
def test_cost_uses_three_prices_at_peak():
    million = 1_000_000
    assert llm.chat_cost(million, 0, 0, MONDAY_10AM) == Decimal("0.04")
    assert llm.chat_cost(0, million, 0, MONDAY_10AM) == Decimal("2")
    assert llm.chat_cost(0, 0, million, MONDAY_10AM) == Decimal("8")
    # 实际的一次调用：2000 个输入 token（其中 1500 个命中缓存），300 个输出 token
    assert llm.chat_cost(1500, 500, 300, MONDAY_10AM) == Decimal("0.00346")


# 测试：低谷时段半价
def test_cost_is_half_off_peak():
    saturday = datetime(2026, 11, 7, 10, 0, tzinfo=BEIJING)
    assert llm.chat_cost(0, 1_000_000, 0, saturday) == Decimal("1")


# 测试：高峰时段的边界。开始的那一刻算高峰，结束的那一刻不算
@pytest.mark.parametrize(
    "hour, minute, peak",
    [(8, 59, False), (9, 0, True), (11, 59, True), (12, 0, False), (14, 0, True), (17, 59, True), (18, 0, False)],
)
def test_peak_hours_on_weekdays(hour, minute, peak):
    assert llm.is_peak(datetime(2026, 11, 2, hour, minute, tzinfo=BEIJING)) is peak


# 测试：按北京时间判断。UTC 的周一 01:00 就是北京时间的周一 09:00，算高峰
def test_peak_uses_beijing_time():
    assert llm.is_peak(datetime(2026, 11, 2, 1, 0, tzinfo=timezone.utc))
    # UTC 的周日 23:00 是北京时间的周一 07:00，不是高峰
    assert not llm.is_peak(datetime(2026, 11, 1, 23, 0, tzinfo=timezone.utc))
