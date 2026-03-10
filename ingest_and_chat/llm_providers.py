# ingest_and_chat/llm_providers.py
"""
Multi-provider LLM abstraction layer.

Supports:
  - Gemini (via langchain-google-genai, existing)
  - Claude (via anthropic SDK)
  - Ollama (via HTTP API to local server)

Each provider exposes a unified interface:
  invoke(prompt: str) -> str

Provider instances are lazy-loaded and cached per configuration.
"""

import os
import json
import logging
from typing import Optional

from .config import _setting

logger = logging.getLogger("ingest_and_chat")

# ---------------------------------------------------------------------------
# Provider Configuration (from Django settings / env vars)
# ---------------------------------------------------------------------------

# Claude
ANTHROPIC_API_KEY = _setting("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = _setting("CLAUDE_MODEL", "CLAUDE_MODEL", "claude-opus-4-20250514")

# Ollama
OLLAMA_HOST = _setting("OLLAMA_HOST", "OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = _setting("OLLAMA_MODEL", "OLLAMA_MODEL", "llama3")

# Gemini (reuses existing config)
GOOGLE_API_KEY = _setting("GOOGLE_API_KEY", "GOOGLE_API_KEY", "")
GEMINI_MODEL = _setting("LLM_MODEL", "LLM_MODEL", "gemini-2.5-pro")
GEMINI_TEMPERATURE = float(_setting("LLM_TEMPERATURE", "LLM_TEMPERATURE", 0.2))

# ---------------------------------------------------------------------------
# Cached provider instances
# ---------------------------------------------------------------------------
_provider_cache = {}


# ===========================================================================
# Unified Provider Wrapper
# ===========================================================================

class LLMProvider:
    """Uniform interface over different LLM backends."""

    def __init__(self, name: str):
        self.name = name

    def invoke(self, prompt: str) -> str:
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    """Uses the existing langchain-google-genai integration."""

    def __init__(self):
        super().__init__("gemini")
        from langchain_google_genai import ChatGoogleGenerativeAI
        self._llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=GEMINI_TEMPERATURE,
            timeout=120,
            max_retries=4,
        )

    def invoke(self, prompt: str) -> str:
        response = self._llm.invoke(prompt)
        return response.content


class ClaudeProvider(LLMProvider):
    """Uses the Anthropic Python SDK."""

    def __init__(self):
        super().__init__("claude")
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file "
                "or Django INGEST_AND_CHAT settings."
            )
        import anthropic
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self._model = CLAUDE_MODEL

    def invoke(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        # Extract text from content blocks
        parts = []
        for block in response.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts)


class OllamaProvider(LLMProvider):
    """Calls a local Ollama server via its HTTP API."""

    def __init__(self):
        super().__init__("ollama")
        self._host = OLLAMA_HOST.rstrip("/")
        self._model = OLLAMA_MODEL

    def invoke(self, prompt: str) -> str:
        import urllib.request
        import urllib.error

        url = f"{self._host}/api/generate"
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach Ollama at {self._host}. "
                f"Ensure Ollama is running. Error: {e}"
            ) from e


# ===========================================================================
# Provider Factory
# ===========================================================================

PROVIDER_MAP = {
    "gemini": GeminiProvider,
    "claude": ClaudeProvider,
    "ollama": OllamaProvider,
}

AVAILABLE_PROVIDERS = list(PROVIDER_MAP.keys())


def get_provider(name: str) -> LLMProvider:
    """
    Get a cached LLM provider instance by name.

    Raises ValueError if the provider name is unknown.
    Raises appropriate errors if credentials are missing.
    """
    name = name.lower().strip()
    if name not in PROVIDER_MAP:
        raise ValueError(
            f"Unknown LLM provider '{name}'. "
            f"Available: {', '.join(AVAILABLE_PROVIDERS)}"
        )

    if name not in _provider_cache:
        logger.info("Initializing LLM provider: %s", name)
        _provider_cache[name] = PROVIDER_MAP[name]()
        logger.info("LLM provider '%s' ready.", name)

    return _provider_cache[name]


def get_provider_info() -> list:
    """
    Return metadata about all providers for the frontend UI.
    Includes availability status based on configuration.
    """
    providers = []

    # Gemini
    providers.append({
        "id": "gemini",
        "name": "Gemini",
        "model": GEMINI_MODEL,
        "available": bool(GOOGLE_API_KEY),
        "reason": None if GOOGLE_API_KEY else "GOOGLE_API_KEY not set",
    })

    # Claude
    providers.append({
        "id": "claude",
        "name": "Claude",
        "model": CLAUDE_MODEL,
        "available": bool(ANTHROPIC_API_KEY),
        "reason": None if ANTHROPIC_API_KEY else "ANTHROPIC_API_KEY not set",
    })

    # Ollama
    ollama_available = True
    ollama_reason = None
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{OLLAMA_HOST.rstrip('/')}/api/tags",
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            resp.read()
    except Exception:
        ollama_available = False
        ollama_reason = f"Cannot reach Ollama at {OLLAMA_HOST}"

    providers.append({
        "id": "ollama",
        "name": "Ollama",
        "model": OLLAMA_MODEL,
        "available": ollama_available,
        "reason": ollama_reason,
    })

    return providers
