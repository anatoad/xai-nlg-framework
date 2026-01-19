"""
LLM Client for XAI-NLG Framework.
Supports both local Ollama and ReaderBench (RoGemma) APIs.
"""
import os
from typing import Optional
from config.settings import NLGConfig

# load .env if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ================= CONFIGURATION =================
# ReaderBench settings (default)
DEFAULT_HOST_URL = "https://chat.readerbench.com/ollama"
DEFAULT_MODEL = "llama4:16x17b"
DEFAULT_API_KEY = "sk-56a239006a004929b080fd644a1f89ee"

# For local Ollama, set OLLAMA_HOST_URL=http://localhost:11434
# ================================================


_client = None


def get_ollama_client(config: Optional[NLGConfig] = None):
    """
    Get or create Ollama client.
    
    Supports:
    - ReaderBench: https://chat.readerbench.com/ollama (default)
    - Local Ollama: http://localhost:11434
    """
    global _client

    if _client is not None:
        return _client

    try:
        from ollama import Client
    except ImportError:
        raise ImportError(
            "The 'ollama' Python package is not installed. "
            "Install it with: pip install ollama"
        )

    # Get host URL (env var overrides default)
    host = os.getenv("OLLAMA_HOST_URL", DEFAULT_HOST_URL)

    # Get API key (config overrides env var overrides default)
    api_key = None
    if config is not None and getattr(config, "api_key", None):
        api_key = config.api_key
    else:
        api_key = os.getenv("OLLAMA_API_KEY", DEFAULT_API_KEY)

    # Create client with or without auth
    if api_key:
        headers = {"Authorization": f"Bearer {api_key}"}
        _client = Client(host=host, headers=headers)
    else:
        _client = Client(host=host)

    return _client


def reset_client():
    """Reset the client (useful when changing configuration)."""
    global _client
    _client = None


def ollama_llm_call(prompt: str, config: NLGConfig) -> str:
    """
    Call the LLM with a prompt.
    
    Args:
        prompt: The prompt to send
        config: NLG configuration
        
    Returns:
        Generated text response
    """
    client = get_ollama_client(config)
    
    # Use configured model or default to ReaderBench model
    model_name = config.model_name
    if model_name == "llama3:latest":
        # Replace default with ReaderBench model
        model_name = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)

    response = client.generate(
        model=model_name,
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