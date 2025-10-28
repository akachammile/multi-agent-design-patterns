from settings import settings
from typing import Dict
from langchain.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama


class LLM:
    """设置LLM属性以及方法"""
    _instances: Dict[str, "LLM"] = {}


    # 单例模式
    def __new__(cls, config_name: str = "default"):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__()
            cls._instances[config_name] = instance
        return cls._instances[config_name]
    
    def __init__(self):
        if not hasattr(self, "client"):  # 没有llm实例则初始化一个即可
            self.model_name = settings.MODEL_NAME
            self.model_provider = settings.MODEL_PROVIDER
            self.model_base_url = settings.MODEL_BASE_URL
            self.max_tokens = settings.MODEL_MAX_TOKENS
            self.temperature = settings.MODEL_TEMPERATURE
            self.model_api_key = settings.MODEL_API_KEY
    
        if self.model_provider == "ollama":
            self.client = ChatOllama(
                model=self.model_name
                ,base_url=self.model_base_url
                ,temperature=self.temperature)
        elif self.model_provider == "anthropic":
            self.client = ChatAnthropic(
                name=self.model_name
                ,anthropic_api_key=self.model_api_key
                ,temperature=self.temperature
                ,max_tokens=self.max_tokens)
        elif self.model_provider == "openai":
            self.client = ChatOpenAI(
                model=self.model_name
                ,base_url=self.model_base_url
                ,api_key=self.model_api_key
                ,temperature=self.temperature
                ,max_tokens=self.max_tokens)