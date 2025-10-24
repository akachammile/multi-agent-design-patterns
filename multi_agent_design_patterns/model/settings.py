
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
load_dotenv()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(os.getenv("ENV", ".env"))

    DB_HOST: str
    DB_PORT: str
    DB_USER: str
    DB_PASS: str
    DB_NAME: str

    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    # 大模型相关配置
    LLM_MODEL_NAME: str = "gpt-3.5-turbo"  # 默认大模型名称
    LLM_TEMPERATURE: float = 0.5          # 默认温度参数
    LLM_MAX_TOKENS: int = 1024            # 最大生成token数

settings = Settings()