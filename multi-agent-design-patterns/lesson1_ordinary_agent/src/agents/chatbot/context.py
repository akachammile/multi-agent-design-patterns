from typing import Annotated
from dataclasses import dataclass, field
from src.agents.common.base_context import BaseContext


@dataclass(kw_only=True)
class Context(BaseContext):
    tools: Annotated[list[dict], {"__template_metadata__": {"kind": "tools"}}] = field(
        default_factory=list,
        metadata={
            "name": "工具",
            "options": lambda: "",  # 这里的选择是所有的工具
            "description": "工具列表",
        },
    )
