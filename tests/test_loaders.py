"""Tests for udaplay.loaders — GameJSONLoader and helpers."""
import json
import pytest

from udaplay.loaders import GameJSONLoader, format_game_document, build_metadata
from udaplay.documents import Corpus, Document


class TestFormatGameDocument:
    def test_includes_all_fields(self, sample_game):
        text = format_game_document(sample_game)
        assert "Super Mario 64" in text
        assert "Nintendo 64" in text
        assert "Platformer" in text
        assert "Nintendo" in text
        assert "1996" in text
        assert "3D platformer" in text

    def test_returns_string(self, sample_game):
        assert isinstance(format_game_document(sample_game), str)

    def test_format_structure(self, sample_game):
        text = format_game_document(sample_game)
        assert "Game:" in text
        assert "Platform:" in text
        assert "Genre:" in text
        assert "Publisher:" in text
        assert "Released:" in text
        assert "Description:" in text


class TestBuildMetadata:
    def test_all_expected_keys(self, sample_game):
        meta = build_metadata(sample_game)
        assert set(meta.keys()) == {"title", "platform", "genre", "publisher", "release_year", "source_file"}

    def test_values(self, sample_game):
        meta = build_metadata(sample_game)
        assert meta["title"] == "Super Mario 64"
        assert meta["platform"] == "Nintendo 64"
        assert meta["genre"] == "Platformer"
        assert meta["publisher"] == "Nintendo"
        assert meta["release_year"] == 1996
        assert meta["source_file"] == "009.json"

    def test_release_year_is_int(self, sample_game):
        meta = build_metadata(sample_game)
        assert isinstance(meta["release_year"], int)

    def test_missing_source_file(self, sample_game):
        """build_metadata should not crash if source_file is absent."""
        game = {k: v for k, v in sample_game.items() if k != "source_file"}
        meta = build_metadata(game)
        assert meta["source_file"] == ""


class TestGameJSONLoader:
    def test_load_returns_list(self, games_dir):
        loader = GameJSONLoader(games_dir)
        games = loader.load()
        assert isinstance(games, list)

    def test_load_count(self, games_dir):
        loader = GameJSONLoader(games_dir)
        games = loader.load()
        assert len(games) == 3   # conftest writes 3 files

    def test_source_file_injected(self, games_dir):
        loader = GameJSONLoader(games_dir)
        for game in loader.load():
            assert "source_file" in game
            assert game["source_file"].endswith(".json")

    def test_load_sorted_order(self, games_dir):
        loader = GameJSONLoader(games_dir)
        games = loader.load()
        source_files = [g["source_file"] for g in games]
        assert source_files == sorted(source_files)

    def test_as_corpus_returns_corpus(self, games_dir):
        loader = GameJSONLoader(games_dir)
        corpus = loader.as_corpus()
        assert isinstance(corpus, Corpus)

    def test_as_corpus_each_item_is_document(self, games_dir):
        loader = GameJSONLoader(games_dir)
        for doc in loader.as_corpus():
            assert isinstance(doc, Document)

    def test_as_corpus_document_ids(self, games_dir):
        loader = GameJSONLoader(games_dir)
        for doc in loader.as_corpus():
            # IDs should be file stems: "006", "009", "015"
            assert doc.id.isdigit()

    def test_as_corpus_content_not_empty(self, games_dir):
        loader = GameJSONLoader(games_dir)
        for doc in loader.as_corpus():
            assert len(doc.content) > 20

    def test_as_corpus_metadata_present(self, games_dir):
        loader = GameJSONLoader(games_dir)
        for doc in loader.as_corpus():
            assert doc.metadata is not None
            assert "title" in doc.metadata
            assert "release_year" in doc.metadata

    def test_ignores_non_json_files(self, games_dir):
        # Add a non-JSON file — loader should ignore it
        (games_dir / "README.txt").write_text("not a game")
        loader = GameJSONLoader(games_dir)
        assert len(loader.load()) == 3

    def test_empty_directory(self, tmp_path):
        loader = GameJSONLoader(tmp_path)
        assert loader.load() == []
        assert len(loader.as_corpus()) == 0
