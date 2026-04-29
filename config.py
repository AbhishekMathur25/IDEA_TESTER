"""
Idea Tester — Central Configuration
Loads settings from .env and provides defaults.
"""

import os
from dotenv import load_dotenv

# Load .env from project root
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    load_dotenv(_env_path, override=True)


class Settings:
    """Application settings loaded from environment."""

    # ── Gemini LLM ─────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-3-flash-preview")
    GEMINI_THINKING_LEVEL: str = os.getenv("GEMINI_THINKING_LEVEL", "HIGH")

    # ── Groq LLM ──────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "qwen/qwen3-32b")

    # ── Ollama LLM ────────────────────────────────────
    OLLAMA_MODEL_NAME: str = os.getenv("OLLAMA_MODEL_NAME", "qwen3.5:397b-cloud")
    #OLLAMA_MODEL_NAME: str = os.getenv("OLLAMA_MODEL_NAME", "qwen3:4b")
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    # ── Zep Cloud (Knowledge Graph) ────────────────────
    ZEP_API_KEY: str = os.getenv("ZEP_API_KEY", "")

    # ── Web Search ─────────────────────────────────────
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # ── Simulation ─────────────────────────────────────
    SIM_ROUNDS: int = int(os.getenv("SIM_ROUNDS", "15"))

    # ── Paths ──────────────────────────────────────────
    PROJECT_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(PROJECT_DIR, "data")

    # ── Available LLM Providers ────────────────────────
    @classmethod
    def available_providers(cls) -> list[str]:
        """Return list of LLM providers with valid API keys."""
        providers = []
        if cls.GEMINI_API_KEY:
            providers.append("Gemini")
        if cls.GROQ_API_KEY:
            providers.append("Groq")
        providers.append("Ollama")  # Local, so always available
        return providers

    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of missing-config error messages."""
        errors = []
        if not cls.GEMINI_API_KEY and not cls.GROQ_API_KEY and "Ollama" not in cls.available_providers():
            errors.append("No LLM configured — set GEMINI_API_KEY, GROQ_API_KEY, or ensure Ollama is running.")
        if not cls.ZEP_API_KEY:
            errors.append("ZEP_API_KEY is not set (Zep Cloud graph memory disabled)")
        if not cls.TAVILY_API_KEY:
            errors.append("TAVILY_API_KEY is not set (research agents won't use web search)")
        return errors


settings = Settings()
