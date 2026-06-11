"""
Shared pytest fixtures for the UdaPlay test suite.

All fixtures here avoid real API calls so the unit test suite runs
without any credentials (OPENAI_API_KEY / TAVILY_API_KEY not required).

Integration tests that DO call the real APIs are marked with
``@pytest.mark.integration`` and are excluded by default:

    pytest -m "not integration" -q
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Sample game data
# ---------------------------------------------------------------------------

SAMPLE_GAMES = [
    {
        "Name": "Super Mario 64",
        "Platform": "Nintendo 64",
        "Genre": "Platformer",
        "Publisher": "Nintendo",
        "Description": "A groundbreaking 3D platformer featuring Mario's quest to rescue Princess Peach.",
        "YearOfRelease": 1996,
        "source_file": "009.json",
    },
    {
        "Name": "Pokémon Gold and Silver",
        "Platform": "Game Boy Color",
        "Genre": "Role-playing",
        "Publisher": "Nintendo",
        "Description": "Second-generation Pokémon games introducing new regions and gameplay mechanics.",
        "YearOfRelease": 1999,
        "source_file": "006.json",
    },
    {
        "Name": "Halo Infinite",
        "Platform": "Xbox Series X|S",
        "Genre": "First-person shooter",
        "Publisher": "Xbox Game Studios",
        "Description": "The latest Halo game featuring Master Chief in a new open-world setting.",
        "YearOfRelease": 2021,
        "source_file": "015.json",
    },
]

SAMPLE_GAME = SAMPLE_GAMES[0]


@pytest.fixture
def sample_game():
    return SAMPLE_GAME.copy()


@pytest.fixture
def sample_games():
    return [g.copy() for g in SAMPLE_GAMES]


# ---------------------------------------------------------------------------
# Data directory fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def games_dir(tmp_path):
    """Temporary directory with 3 valid game JSON files."""
    for game in SAMPLE_GAMES:
        file_path = tmp_path / game["source_file"]
        data = {k: v for k, v in game.items() if k != "source_file"}
        file_path.write_text(json.dumps(data), encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# ChromaDB mock fixtures
# ---------------------------------------------------------------------------

MOCK_RETRIEVAL_RESULT = {
    "documents": [[
        "Game: Super Mario 64. Platform: Nintendo 64. Genre: Platformer. "
        "Publisher: Nintendo. Released: 1996. "
        "Description: A groundbreaking 3D platformer featuring Mario's quest."
    ]],
    "metadatas": [[{
        "title": "Super Mario 64",
        "platform": "Nintendo 64",
        "publisher": "Nintendo",
        "genre": "Platformer",
        "release_year": 1996,
        "source_file": "009.json",
    }]],
    "distances": [[0.05]],
    "ids": [["009"]],
}

LOW_CONFIDENCE_RETRIEVAL_RESULT = {
    "documents": [[
        "Game: Gran Turismo. Platform: PlayStation 1. Genre: Racing. "
        "Publisher: Sony Computer Entertainment. Released: 1997. "
        "Description: A realistic racing simulator."
    ]],
    "metadatas": [[{
        "title": "Gran Turismo",
        "platform": "PlayStation 1",
        "publisher": "Sony Computer Entertainment",
        "genre": "Racing",
        "release_year": 1997,
        "source_file": "001.json",
    }]],
    "distances": [[0.85]],
    "ids": [["001"]],
}


@pytest.fixture
def mock_collection():
    """ChromaDB collection that returns a high-similarity Mario result."""
    col = MagicMock()
    col.query.return_value = MOCK_RETRIEVAL_RESULT
    col.count.return_value = 15
    return col


@pytest.fixture
def mock_collection_low_confidence():
    """ChromaDB collection that returns a low-similarity (irrelevant) result."""
    col = MagicMock()
    col.query.return_value = LOW_CONFIDENCE_RETRIEVAL_RESULT
    col.count.return_value = 15
    return col


# ---------------------------------------------------------------------------
# LLM response fixtures (mocked AIMessage)
# ---------------------------------------------------------------------------

def _make_ai_message(content: str):
    """Build a minimal AIMessage-like MagicMock with .content attribute."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    return msg


HIGH_CONFIDENCE_EVAL = json.dumps({
    "confidence": "high",
    "useful": True,
    "description": "Document directly answers the question about Super Mario 64.",
    "needs_web_search": False,
})

LOW_CONFIDENCE_EVAL = json.dumps({
    "confidence": "low",
    "useful": False,
    "description": "Retrieved documents are not relevant to the question about FIFA 21.",
    "needs_web_search": True,
})


@pytest.fixture
def mock_llm_high_confidence():
    llm = MagicMock()
    llm.invoke.return_value = _make_ai_message(HIGH_CONFIDENCE_EVAL)
    return llm


@pytest.fixture
def mock_llm_low_confidence():
    llm = MagicMock()
    llm.invoke.return_value = _make_ai_message(LOW_CONFIDENCE_EVAL)
    return llm


# ---------------------------------------------------------------------------
# Tavily mock
# ---------------------------------------------------------------------------

MOCK_TAVILY_RESPONSE = {
    "answer": "FIFA 21 was developed by EA Vancouver and published by Electronic Arts.",
    "results": [
        {
            "title": "FIFA 21 — Wikipedia",
            "url": "https://en.wikipedia.org/wiki/FIFA_21",
            "content": "FIFA 21 is a football simulation video game developed by EA Vancouver.",
        }
    ],
}


@pytest.fixture
def mock_tavily_client():
    client = MagicMock()
    client.search.return_value = MOCK_TAVILY_RESPONSE
    return client
