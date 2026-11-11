"""Build LangChain's local Ollama chat-model adapter."""

from langchain_ollama import ChatOllama

from .config import Settings


def build_model(settings: Settings) -> ChatOllama:
    return ChatOllama(
        model=settings.model,
        base_url=settings.base_url,
        temperature=settings.temperature,
        num_predict=settings.max_output_tokens,
        reasoning=False,
        keep_alive=settings.keep_alive,
        client_kwargs={"timeout": settings.timeout_seconds},
    )
