"""
UdaPlay — AI Research Agent for the Video Game Industry.

Public API
----------
>>> from udaplay import Settings, Agent, make_game_tools, LongTermMemory
"""
from udaplay.config import Settings, settings
from udaplay.agents import Agent
from udaplay.loaders import GameJSONLoader, format_game_document, build_metadata
from udaplay.long_term_memory import LongTermMemory
from udaplay.tools import make_game_tools, EvaluationReport
from udaplay.vector_db import VectorStore, VectorStoreManager

__version__ = "0.2.0"

__all__ = [
    "Settings",
    "settings",
    "Agent",
    "GameJSONLoader",
    "format_game_document",
    "build_metadata",
    "LongTermMemory",
    "make_game_tools",
    "EvaluationReport",
    "VectorStore",
    "VectorStoreManager",
]
