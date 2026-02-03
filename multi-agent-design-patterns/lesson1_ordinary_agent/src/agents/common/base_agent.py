import importlib.util
from pathlib import Path
from __future__ import annotations
from abc import ABC, abstractmethod
from src import config as sys_config
from .base_context import BaseContext


class BaseAgent:
    """baseagent,定义agent的源方法和元数据"""

    name: str = "base_agent"
    description: str = "agent基类,继承用"
    context: type[BaseContext] = BaseContext

    def __init__(self, **kwargs):
        self.graph = None
        self.checkpoint = None
        self.datadir = Path(sys_config.save_dir) / "agents" / self.module_name
        self.datadir.mkdir(parents=True, exist_ok=True)
        self._cache = None  # 数据缓存

    @property
    def module_name(self) -> str:
        """获取当前agent的所在模块名称"""
        return self.__class__.__module__.split(".")[-2]

    @property
    def id(self) -> str:
        """获取类名."""
        return self.__class__.__name__

    def save(self, path: Path | str | None = None) -> None:
        """保存agent"""
        if path is None:
            path = self.datadir / f"{self.name}.yaml"

    def load_metadata(self) -> dict:
        """加载缓存
        Returns:
            dict
        """

        if self._cache is not None:
            return self._cache

        agent = self.__class__.__module__

        agent_spec = importlib.util.find_spec(agent)
        if agent_spec and agent_spec.origin:
            agent_file = Path(agent_spec.origin)
            agent_dir = agent_file.parent
        else:
            module_path = agent.replace(".", "/")
            agent_file = Path(f"src/{module_path}.py")
            agent_dir = agent_file.parent

        metadata_file = agent_dir / "metadata.toml"

        if metadata_file.exists():
            with open(metadata_file, "rb") as f:
                metadata = tomli.load(f)
                self._metadata_cache = metadata
                logger.debug(f"Loaded metadata from {metadata_file}")
                return metadata
        else:
            logger.debug(
                f"No metadata.toml found for {self.module_name} at {metadata_file}"
            )
            self._metadata_cache = {}
            return {}
