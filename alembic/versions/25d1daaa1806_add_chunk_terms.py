"""add chunk terms

Revision ID: 25d1daaa1806
Revises: b0c19d1dc682
Create Date: 2026-10-31 15:00:00.000000

给 chunks 表加关键词检索用的 terms 列（tsvector）和 GIN 索引，并给已有的段落补上分词结果。手写的迁移。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
# TSVECTOR：全文检索类型
from sqlalchemy.dialects.postgresql import TSVECTOR

# 分词函数。一般迁移不引用应用代码（代码以后会变，迁移记录的应该是"当时"的状态），
# 这里是例外：补数据必须和线上写入新段落时的切法完全一样，否则旧段落的词和问题的词对不上
from app.segment import segment


# revision identifiers, used by Alembic.
revision: str = '25d1daaa1806'
down_revision: Union[str, Sequence[str], None] = 'b0c19d1dc682'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# 补数据时每批处理多少段
BATCH_SIZE = 500


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("chunks", sa.Column("terms", TSVECTOR(), nullable=True))
    op.create_index("ix_chunks_terms_gin", "chunks", ["terms"], postgresql_using="gin")
    # 给已有的段落补上 terms。只需要分词（本地计算，不花钱），不用重新切分、重新生成向量
    conn = op.get_bind()
    update = sa.text("UPDATE chunks SET terms = to_tsvector('simple', :words) WHERE id = :id")
    last_id = 0
    while True:
        # 按 id 一批一批往后取，不一次性把所有段落读进内存
        rows = conn.execute(
            sa.text("SELECT id, content FROM chunks WHERE id > :last_id ORDER BY id LIMIT :n"),
            {"last_id": last_id, "n": BATCH_SIZE},
        ).all()
        if not rows:
            break
        conn.execute(update, [{"id": row.id, "words": " ".join(segment(row.content))} for row in rows])
        last_id = rows[-1].id


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_chunks_terms_gin", table_name="chunks")
    op.drop_column("chunks", "terms")
