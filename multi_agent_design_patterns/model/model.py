from langchain_openai import ChatOpenAI
from settings import settings

class LLM:
    _instance = None  # 单例模式的静态变量

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LLM, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name="gpt-3.5-turbo", temperature=0.5):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    def get_response(self, prompt):
        return self.llm.predict(prompt)