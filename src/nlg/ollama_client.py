import os
from typing import Optional
from config.settings import NLGConfig

# load .env if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from ollama import Client
except ImportError:
    Client = None  # type: ignore

_client = None

def get_ollama_client(config: Optional[NLGConfig] = None):
    global _client

    if _client is not None:
        return _client

    if Client is None:
        raise ImportError(
            "The 'ollama' Python package is not installed. "
            "Install it with: pip install ollama"
        )

    # LOCAL by default
    host = os.getenv("OLLAMA_HOST_URL", "http://localhost:11434")

    api_key = None
    if config is not None and getattr(config, "api_key", None):
        api_key = config.api_key
    else:
        api_key = os.getenv("OLLAMA_API_KEY")

    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}
        _client = Client(host=host, headers=headers)
    else:
        _client = Client(host=host)

    return _client


def ollama_llm_call(prompt: str, config: NLGConfig) -> str:
    client = get_ollama_client(config)

    response = client.generate(
        model=config.model_name,
        prompt=prompt,
        options={
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        },
    )

    # response can be dict or object with .response
    if isinstance(response, dict):
        return str(response.get("response", "")).strip()

    text = getattr(response, "response", "")
    return str(text).strip()
