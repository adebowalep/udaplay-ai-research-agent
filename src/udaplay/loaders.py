"""
Document loaders for UdaPlay.

Provides:
- GameJSONLoader  — loads game JSON files into Documents / raw dicts
- PDFLoader       — loads PDF files page-by-page (retained from original lib)

The GameJSONLoader is the primary loader used by the RAG pipeline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pdfplumber

from udaplay.documents import Corpus, Document


# ---------------------------------------------------------------------------
# Game document helpers
# ---------------------------------------------------------------------------

def format_game_document(game: dict) -> str:
    """
    Converts a raw game dict into a rich natural-language string for embedding.

    Including all fields makes the document semantically searchable on any
    dimension (platform, genre, publisher, description, release year).
    """
    return (
        f"Game: {game['Name']}. "
        f"Platform: {game['Platform']}. "
        f"Genre: {game['Genre']}. "
        f"Publisher: {game['Publisher']}. "
        f"Released: {game['YearOfRelease']}. "
        f"Description: {game['Description']}"
    )


def build_metadata(game: dict) -> dict:
    """
    Extracts a flat metadata dict suitable for ChromaDB storage.

    ChromaDB requires metadata values to be str / int / float / bool.
    """
    return {
        "title":        game["Name"],
        "platform":     game["Platform"],
        "genre":        game["Genre"],
        "publisher":    game["Publisher"],
        "release_year": int(game["YearOfRelease"]),
        "source_file":  game.get("source_file", ""),
    }


# ---------------------------------------------------------------------------
# GameJSONLoader
# ---------------------------------------------------------------------------

class GameJSONLoader:
    """
    Loads all game JSON files from a directory.

    Each JSON file is expected to have the following fields:
        Name, Platform, Genre, Publisher, Description, YearOfRelease

    Example
    -------
    >>> loader = GameJSONLoader("data/games")
    >>> games = loader.load()
    >>> print(len(games))
    15
    >>> corpus = loader.as_corpus()
    >>> print(corpus[0].content[:60])
    'Game: Gran Turismo. Platform: PlayStation 1...'
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def load(self) -> List[dict]:
        """
        Load all .json files and return a list of game dicts.
        Each dict gains a ``source_file`` key with the file's basename.
        """
        games: List[dict] = []
        for file_name in sorted(os.listdir(self.data_dir)):
            if not file_name.endswith(".json"):
                continue
            file_path = self.data_dir / file_name
            with open(file_path, "r", encoding="utf-8") as f:
                game = json.load(f)
            game["source_file"] = file_name
            games.append(game)
        return games

    def as_corpus(self) -> Corpus:
        """
        Load games and return a :class:`Corpus` of :class:`Document` objects
        ready for insertion into a ChromaDB collection.

        Document IDs are set to the file stem (e.g. ``"001"`` from ``001.json``).
        """
        games = self.load()
        corpus = Corpus()
        for game in games:
            doc_id = Path(game["source_file"]).stem
            corpus.append(
                Document(
                    id=doc_id,
                    content=format_game_document(game),
                    metadata=build_metadata(game),
                )
            )
        return corpus


# ---------------------------------------------------------------------------
# PDFLoader (retained from original Udacity lib)
# ---------------------------------------------------------------------------

class PDFLoader:
    """
    Document loader for extracting text content from PDF files.

    Each page becomes a separate :class:`Document` object, enabling
    page-level search and retrieval in RAG applications.
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def load(self) -> Corpus:
        corpus = Corpus()
        with pdfplumber.open(self.pdf_path) as pdf:
            for num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    corpus.append(
                        Document(id=str(num), content=text)
                    )
        return corpus
