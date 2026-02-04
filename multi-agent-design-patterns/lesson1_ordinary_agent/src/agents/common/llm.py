import os
from dotenv import load_dotenv
from src.configs import config
from langchain.chat_models import init_chat_model, BaseChatModel


def load_chat_model(full_model_name: str, **kwargs) -> BaseChatModel:

    provider, model = full_model_name.split("/", maxsplit=1)
    model_info = config.model_names.get(provider)
    if not model_info:
        raise ValueError(f"Unknown model provider: {provider}")
    api_key = os.getenv(model_info.env, [])

    if provider in ["openai", "deepseek"]:
        model_spec = f"{provider}:{model}"
        logger.debug(f"[offical] Loading model {model_spec} with kwargs {kwargs}")
        return init_chat_model(model_spec, **kwargs)

    elif provider in ["dashscope"]:
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=model,
            api_key=SecretStr(api_key),
            base_url=base_url,
            api_base=base_url,
            stream_usage=True,
        )

    elif provider in ["gemini"]:
        from langchain_google_genai import GoogleGenerativeAI

        return GoogleGenerativeAI(
            model=model_spec,
            api_key=SecretStr(api_key),
        )
