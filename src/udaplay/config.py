"""
Configuration management for UdaPlay.

Centralises all runtime settings so notebooks, the Streamlit UI, and tests
can all share the same defaults without hard-coding values.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root = two levels up from this file (src/udaplay/config.py → root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class Settings:
    """Runtime settings loaded from environment variables with sane defaults."""

    # --- API keys ---
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    tavily_api_key: str = field(
        default_factory=lambda: os.getenv("TAVILY_API_KEY", "")
    )

    # --- ChromaDB ---
    chroma_path: str = field(
        default_factory=lambda: str(_PROJECT_ROOT / "chromadb")
    )
    collection_name: str = "udaplay_games"

    # --- Models ---
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0

    # --- Data ---
    data_dir: str = field(
        default_factory=lambda: str(_PROJECT_ROOT / "data" / "games")
    )

    # --- Agent behaviour ---
    n_retrieval_results: int = 3
    memory_file: str = field(
        default_factory=lambda: str(_PROJECT_ROOT / "agent_memory.json")
    )

    # --- Long-term memory ---
    long_term_memory_file: str = field(
        default_factory=lambda: str(_PROJECT_ROOT / "data" / "memory" / "long_term_memory.json")
    )

    # --- Validation helpers ---
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key)

    def has_tavily_key(self) -> bool:
        return bool(self.tavily_api_key)

    def validate(self) -> None:
        """Raise EnvironmentError if required keys are missing."""
        if not self.has_openai_key():
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. "
                "Create a .env file with OPENAI_API_KEY=your_key."
            )


# Module-level singleton (lazy — keys read at import time)
settings = Settings()
