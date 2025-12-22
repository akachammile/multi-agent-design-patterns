from typing import Dict, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LLM_API_KEY: str
    LLM_MODEL_NAME: str
    LLM_MODEL_BASE_URL: str
    LLM_MODEL_PROVIDER: str


settings = Settings()


class LLM:
    # create single instance dict
    _instances: Dict[str, "LLM"] = {}

    # single pattern
    def __new__(
        cls, config_name: str = "default", llm_config: Optional[Settings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[Settings] = None
    ):
        # Use global settings if no config provided
        if llm_config is None:
            llm_config = settings

        if not hasattr(self, "client"):
            self.model = llm_config.LLM_MODEL_NAME
            self.api_key = llm_config.LLM_API_KEY
            self.base_url = llm_config.LLM_MODEL_BASE_URL

        self.client = ChatOpenAI(
            model=self.model, api_key=self.api_key, base_url=self.base_url
        )
