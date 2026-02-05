"""
业务数据库模型 - 用于存储对话消息等业务数据
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Column, String, Text, DateTime, JSON
from sqlalchemy.dialects.postgresql import UUID

from .base import Base


class Message(Base):
    """对话消息模型"""

    __tablename__ = "messages"

    id = Column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="消息唯一标识"
    )
    thread_id = Column(String(64), nullable=False, index=True, comment="对话线程ID")
    user_id = Column(String(64), nullable=False, index=True, comment="用户ID")
    agent_id = Column(String(64), nullable=False, index=True, comment="Agent类型标识")
    role = Column(
        String(32), nullable=False, comment="消息角色: user/assistant/system/tool"
    )
    content = Column(Text, nullable=False, comment="消息内容")
    metadata = Column(JSON, default=dict, comment="额外元数据")
    created_at = Column(DateTime, default=datetime.now, comment="创建时间")
