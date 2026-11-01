"""add message cost

Revision ID: 7f3a9c2e5b41
Revises: 25d1daaa1806
Create Date: 2026-11-01 12:00:00.000000

给 messages 和 usage_logs 加 cost 列：每次问答的大模型费用（元）。手写的迁移。
已有的数据不补：当时没有记录输入、输出、缓存各用了多少 token，算不出准确的费用，所以保持为空，表示"没有记录"
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7f3a9c2e5b41'
down_revision: Union[str, Sequence[str], None] = '25d1daaa1806'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("messages", sa.Column("cost", sa.Numeric(12, 8), nullable=True))
    op.add_column("usage_logs", sa.Column("cost", sa.Numeric(12, 8), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("usage_logs", "cost")
    op.drop_column("messages", "cost")
