"""add conversation summary

Revision ID: 9d2e4b7a1c08
Revises: 7f3a9c2e5b41
Create Date: 2026-11-01 18:00:00.000000

给 conversations 表加 summary（更早的消息的摘要）和 summarized_count（摘要包含了最早的多少条消息），
并允许 usage_logs.action 取 'summary'（后台摘要也调用大模型，要记录用量和费用）。手写的迁移。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9d2e4b7a1c08'
down_revision: Union[str, Sequence[str], None] = '7f3a9c2e5b41'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("conversations", sa.Column("summary", sa.Text(), nullable=True))
    # 已有的会话都还没有摘要，从 0 开始
    op.add_column("conversations", sa.Column("summarized_count", sa.Integer(), server_default="0", nullable=False))
    # action 列是"字符串 + CHECK 约束"（见 models._str_enum），加一个新值就是换一个约束
    op.drop_constraint("usage_action", "usage_logs", type_="check")
    op.create_check_constraint("usage_action", "usage_logs", "action IN ('upload', 'query', 'summary')")


def downgrade() -> None:
    """Downgrade schema."""
    # 先删掉摘要的使用记录，否则换回旧约束时这些行不符合条件，会报错
    op.execute("DELETE FROM usage_logs WHERE action = 'summary'")
    op.drop_constraint("usage_action", "usage_logs", type_="check")
    op.create_check_constraint("usage_action", "usage_logs", "action IN ('upload', 'query')")
    op.drop_column("conversations", "summarized_count")
    op.drop_column("conversations", "summary")
