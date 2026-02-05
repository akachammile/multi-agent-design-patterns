"""
知识库数据库模型 - 基于 pgvector 的向量存储

用于 RAG 检索增强生成场景
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, DateTime, JSON, Index
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

from .base import Base


class KnowledgeChunk(Base):
    """知识块模型 - 用于存储文档片段和向量嵌入"""

    __tablename__ = "knowledge_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source = Column(String(512), nullable=False, index=True)  # 来源文档路径/URL
    content = Column(Text, nullable=False)  # 文本内容
    embedding = Column(Vector(1536))  # 向量嵌入 (OpenAI ada-002 维度)
    metadata = Column(JSON, default=dict)  # 额外元数据
    created_at = Column(DateTime, default=datetime.now)

    __table_args__ = (
        Index(
            "ix_knowledge_chunks_embedding",
            embedding,
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
