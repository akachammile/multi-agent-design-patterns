import os
from sqlalchemy import text
from contextlib import asynccontextmanager
from .business import Base as BusinessBase
from .knowledge import Base as KnowledgeBase
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .base import Base


class PostgreService:
    """
    PostGre 服务类
    """

    POSTGRESQL_URL_ENV = "POSTGRESQL_URL"

    def __init__(self):
        self._initialized = False
        self.async_engine = None
        self.async_session_maker = None

    def _initialize(self):
        if self._initialized:
            return

        postgresql_url = os.getenv(self.POSTGRESQL_URL_ENV)
        if not postgresql_url:
            raise ValueError(
                f"Environment variable {self.POSTGRESQL_URL_ENV} is not set"
            )

        self.async_engine = create_async_engine(
            postgresql_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        self.async_session_maker = async_sessionmaker(
            bind=self.async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        self._initialized = True

    async def get_session(self) -> AsyncSession:
        """获取数据库会话"""
        if not self._initialized:
            self._initialize()
        return self.async_session_maker()

    @asynccontextmanager
    async def get_async_session_context(self):
        """获得异步session 的上下文"""

        session = self.get_session()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def close(self):
        """关闭引擎"""
        if self.async_engine:
            await self.async_engine.dispose()

    async def create_tables(self):
        async with self.async_engine.begin() as conn:
            await conn.run_sync(BusinessBase.metadata.create_all)
            await conn.run_sync(KnowledgeBase.metadata.create_all)
