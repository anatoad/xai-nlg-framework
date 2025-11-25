# src/nlg/ollama_client.py
"""
Thin wrapper peste clientul Ollama, ca să-l putem injecta ușor în NLG.

Design:
- By default se conectează la Ollama local: http://localhost:11434
- Nu are nevoie de API key pentru local.
- Dacă vrei cândva cloud, poți seta OLLAMA_HOST_URL + OLLAMA_API_KEY
  sau NLGConfig.api_key, și le va folosi automat.
"""

import os
from typing import Optional
from config.settings import NLGConfig

# load .env dacă există, dar nu o facem hard dependency
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from ollama import Client
except ImportError:
    Client = None  # type: ignore


_client = None  # singleton simplu


def get_ollama_client(config: Optional[NLGConfig] = None):
    """
    Creează (sau re-folosește) un client Ollama.

    Local (default):
        - host: http://localhost:11434
        - fără API key

    Cloud (opțional, doar dacă vrei):
        - setezi OLLAMA_HOST_URL='https://ollama.com'
        - și OLLAMA_API_KEY sau NLGConfig.api_key
    """
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

    # Pentru local nu avem nevoie de API key, dar lăsăm opțiunea pentru cloud.
    api_key = None
    if config is not None and getattr(config, "api_key", None):
        api_key = config.api_key
    else:
        api_key = os.getenv("OLLAMA_API_KEY")

    if api_key:
        # caz de cloud: trimitem headerul de auth
        headers = {"Authorization": f"Bearer {api_key}"}
        _client = Client(host=host, headers=headers)
    else:
        # caz standard: Ollama local, fără auth
        _client = Client(host=host)

    return _client


def ollama_llm_call(prompt: str, config: NLGConfig) -> str:
    """
    Funcția pe care o injectăm în generatoare.
    Ia promptul, apelează Ollama și întoarce doar textul.
    """
    client = get_ollama_client(config)

    response = client.generate(
        model=config.model_name,
        prompt=prompt,
        options={
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
        },
    )

    # response poate fi dict sau obiect cu .response
    if isinstance(response, dict):
        return str(response.get("response", "")).strip()

    text = getattr(response, "response", "")
    return str(text).strip()
