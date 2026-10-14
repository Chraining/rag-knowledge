"""add chunk embeddings

Revision ID: b0c19d1dc682
Revises: c70aef97018b
Create Date: 2026-10-14 16:27:58.947225

给 chunks 表加向量列和 HNSW 向量索引（阶段 4）。手写的迁移，不是自动生成的。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
# pgvector 提供的向量列类型
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'b0c19d1dc682'
down_revision: Union[str, Sequence[str], None] = 'c70aef97018b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # 启用 pgvector 扩展。IF NOT EXISTS：已经启用过就跳过，不报错。
    # 需要数据库超级用户权限；Docker 镜像里的 rag 用户就是超级用户
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # 加一列 vector(1024)。这里直接写 1024，不引用 models.EMBEDDING_DIM：
    # 迁移文件记录的是"当时"的表结构，以后代码里的常量变了，旧迁移也不应该跟着变
    op.add_column("chunks", sa.Column("embedding", Vector(1024), nullable=True))
    # HNSW 索引，按余弦距离
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    """Downgrade schema."""
    # 按相反的顺序撤销：先删索引，再删列。扩展保留，别的表以后可能也会用到
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_column("chunks", "embedding")
