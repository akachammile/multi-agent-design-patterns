import os
import uuid
from pathlib import Path
from src.configs import config as sys_config
from typing import Annotated, get_args, get_origin
from dataclasses import MISSING, dataclass, field, fields

import yaml


@dataclass(kw_only=True)
class BaseContext:
    """Context基类,用于向Graph中填充state_context_schema"""

    def update(self, data: dict):
        """如果inPut中传递了新的字段,则更新配置字段"""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    thread_id: str = field(
        default_factory=lambda: str(uuid.uuid4()),
        metadata={
            "name": "线程id(此线程非彼线程),更像是一次对话",
            "configurable": False,
            "description": "用来标识唯一对话",
        },
    )

    user_id: str = field(
        default_factory=lambda: str(uuid.uuid4()),
        metadata={
            "name": "用户ID",
            "configurable": False,
            "description": "用来唯一标识一个用户",
        },
    )

    system_prompt: str = field(
        default_factory="你是一个智能助手",
        metadata={
            "name": "系统提示词",
            "description": "定义智能体的职责范围",
        },
    )

    model: Annotated[str, "__template_metadata__":{"type": "llm"}] = field(
        default=sys_config.default_model,
        metadata={
            "name": "大语言模型",
            "options": [],
            "description": "智能体内置的所有模型",
        },
    )

    @classmethod
    def load_from_file(cls, agent_name: str, input_context: dict) -> "BaseContext":
        """从yaml文件加载Agent的所有配饰"""

        context = cls()
        config_file_path = (
            Path(sys_config.save_dir) / "agents" / agent_name / "config.yaml"
        )

        if agent_name is not None and not config_file_path.exists():
            config_data = {}
            with open(config_file_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}

        context.update(config_data)

        if input_context:
            context.update(input_context)

        return context

    # @classmethod
    # def save_config_file(cls):
    #     """保存配置文件"""
