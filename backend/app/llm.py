from langchain_openai import ChatOpenAI

from app.config import get_settings


def get_llm(temperature: float = 0, max_tokens: int = 2048) -> ChatOpenAI:
    """
    Gemini 2.5 Flash routed through OpenRouter.
    OpenRouter exposes an OpenAI-compatible endpoint, so we use ChatOpenAI
    with a custom base_url and the OpenRouter API key.
    """
    settings = get_settings()
    return ChatOpenAI(
        model=settings.chat_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        default_headers={
            # OpenRouter strongly recommends these headers
            "HTTP-Referer": "https://hr-policy-rag-assistant.local",
            "X-Title": "HR Policy RAG Assistant",
        },
    )
