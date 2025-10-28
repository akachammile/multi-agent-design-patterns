import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),  # 默认 .env，可通过 ENV_FILE 切换
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        validate_default=False,
    )
    
    MODEL_NAME: str | None = None
    MODEL_PROVIDER: str | None = None
    MODEL_BASE_URL: str | None = None
    MODEL_API_KEY: str | None = None
    MODEL_TEMPERATURE: float | None = None
    MODEL_MAX_TOKENS: int | None = None
    MODEL_TOP_P: float | None = None
    MODEL_TOP_K: int | None = None


settings = Settings()