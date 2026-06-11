"""
Tests for udaplay.tools — make_game_tools factory and all seven tools.

All tests mock ChromaDB, OpenAI LLM, Tavily, and LongTermMemory so no
real API calls are made.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from udaplay.config import Settings
from udaplay.tools import make_game_tools, EvaluationReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(tavily_key: str = "") -> Settings:
    return Settings(
        openai_api_key="sk-test",
        tavily_api_key=tavily_key,
        llm_model="gpt-4o-mini",
        n_retrieval_results=3,
    )


def _make_mock_memory(search_results=None):
    mem = MagicMock()
    mem.search.return_value = search_results or []
    mem.save.return_value = (True, "Saved")
    return mem


def _make_tools(collection, tavily_key: str = "", memory=None):
    settings = _make_settings(tavily_key)
    with patch("udaplay.tools.LLM"):
        tools = make_game_tools(collection, settings=settings, memory=memory)
    return tools  # 7-tuple


# ---------------------------------------------------------------------------
# retrieve_game
# ---------------------------------------------------------------------------

class TestRetrieveGame:
    def test_returns_string(self, mock_collection):
        retrieve, *_ = _make_tools(mock_collection)
        assert isinstance(retrieve("Mario Nintendo 64"), str)

    def test_contains_title(self, mock_collection):
        retrieve, *_ = _make_tools(mock_collection)
        assert "Super Mario 64" in retrieve("Mario platformer")

    def test_contains_similarity_score(self, mock_collection):
        retrieve, *_ = _make_tools(mock_collection)
        assert "Similarity" in retrieve("Mario")

    def test_contains_source_file(self, mock_collection):
        retrieve, *_ = _make_tools(mock_collection)
        assert "009.json" in retrieve("Mario")

    def test_empty_results(self, mock_collection):
        mock_collection.query.return_value = {
            "documents": [[]], "metadatas": [[]], "distances": [[]]
        }
        retrieve, *_ = _make_tools(mock_collection)
        assert "No documents found" in retrieve("unknown")

    def test_db_error_handled(self):
        bad = MagicMock()
        bad.query.side_effect = RuntimeError("DB error")
        retrieve, *_ = _make_tools(bad)
        assert "Error" in retrieve("any")

    def test_respects_n_results(self, mock_collection):
        settings = _make_settings()
        settings.n_retrieval_results = 5
        with patch("udaplay.tools.LLM"):
            retrieve, *_ = make_game_tools(mock_collection, settings=settings)
        retrieve("test")
        kw = mock_collection.query.call_args[1]
        assert kw["n_results"] == 5


# ---------------------------------------------------------------------------
# evaluate_retrieval
# ---------------------------------------------------------------------------

class TestEvaluateRetrieval:
    def _with_llm_response(self, collection, response_json: str):
        settings = _make_settings()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=response_json)
        with patch("udaplay.tools.LLM", return_value=mock_llm):
            _, evaluate, *_ = make_game_tools(collection, settings=settings)
        return evaluate

    def test_high_confidence(self, mock_collection):
        high = json.dumps({
            "confidence": "high", "useful": True,
            "description": "Direct match.", "needs_web_search": False,
        })
        evaluate = self._with_llm_response(mock_collection, high)
        result = json.loads(evaluate("Platform?", "Game: Super Mario 64. Platform: N64."))
        assert result["confidence"] == "high"
        assert result["needs_web_search"] is False

    def test_low_confidence_needs_web(self, mock_collection):
        low = json.dumps({
            "confidence": "low", "useful": False,
            "description": "Not relevant.", "needs_web_search": True,
        })
        evaluate = self._with_llm_response(mock_collection, low)
        result = json.loads(evaluate("FIFA 21?", "Game: Gran Turismo."))
        assert result["needs_web_search"] is True

    def test_returns_json_string(self, mock_collection):
        resp = json.dumps({
            "confidence": "medium", "useful": True,
            "description": "Partial.", "needs_web_search": False,
        })
        evaluate = self._with_llm_response(mock_collection, resp)
        result = json.loads(evaluate("test", "docs"))
        assert "confidence" in result

    def test_llm_error_defaults_to_web_search(self, mock_collection):
        settings = _make_settings()
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("OpenAI down")
        with patch("udaplay.tools.LLM", return_value=mock_llm):
            _, evaluate, *_ = make_game_tools(mock_collection, settings=settings)
        result = json.loads(evaluate("test", "docs"))
        assert result["needs_web_search"] is True


# ---------------------------------------------------------------------------
# game_web_search
# ---------------------------------------------------------------------------

class TestGameWebSearch:
    def test_no_api_key_returns_error(self, mock_collection):
        _, _, web, *_ = _make_tools(mock_collection, tavily_key="")
        assert "TAVILY_API_KEY" in web("FIFA 21")

    def test_with_key_calls_tavily(self, mock_collection, mock_tavily_client):
        with (
            patch("udaplay.tools.LLM"),
            patch("udaplay.tools.TavilyClient", return_value=mock_tavily_client, create=True),
        ):
            settings = _make_settings(tavily_key="tvly-test")
            _, _, web, *_ = make_game_tools(mock_collection, settings=settings)
            with patch("tavily.TavilyClient", return_value=mock_tavily_client):
                result = web("FIFA 21?")
        assert isinstance(result, str) and len(result) > 0


# ---------------------------------------------------------------------------
# search_memory
# ---------------------------------------------------------------------------

class TestSearchMemory:
    def test_no_memory_configured(self, mock_collection):
        tools = _make_tools(mock_collection, memory=None)
        search_mem = tools[3]
        result = json.loads(search_mem("FIFA 21"))
        assert result["found"] is False

    def test_returns_found_false_when_no_match(self, mock_collection):
        mem = _make_mock_memory(search_results=[])
        tools = _make_tools(mock_collection, memory=mem)
        search_mem = tools[3]
        result = json.loads(search_mem("obscure game 1983"))
        assert result["found"] is False
        assert result["matches"] == []

    def test_returns_found_true_with_matches(self, mock_collection):
        mem = _make_mock_memory(search_results=[{
            "question":    "Who developed FIFA 21?",
            "answer":      "EA Vancouver developed FIFA 21.",
            "source_type": "web_search",
            "confidence":  "high",
            "timestamp":   "2026-06-11T14:58:43",
            "sources":     ["https://en.wikipedia.org/wiki/FIFA_21"],
        }])
        tools = _make_tools(mock_collection, memory=mem)
        search_mem = tools[3]
        result = json.loads(search_mem("FIFA 21 developer"))
        assert result["found"] is True
        assert len(result["matches"]) == 1
        assert "FIFA" in result["matches"][0]["question"]

    def test_calls_memory_search(self, mock_collection):
        mem = _make_mock_memory()
        tools = _make_tools(mock_collection, memory=mem)
        search_mem = tools[3]
        search_mem("test query")
        mem.search.assert_called_once_with("test query", top_k=3)


# ---------------------------------------------------------------------------
# save_memory
# ---------------------------------------------------------------------------

class TestSaveMemory:
    def test_no_memory_configured(self, mock_collection):
        tools = _make_tools(mock_collection, memory=None)
        save_mem = tools[4]
        result = save_mem("Q?", "Answer.", "web_search", "https://a.com", "high")
        assert "not configured" in result.lower()

    def test_saves_valid_entry(self, mock_collection):
        mem = _make_mock_memory()
        tools = _make_tools(mock_collection, memory=mem)
        save_mem = tools[4]
        result = save_mem(
            "Who made FIFA 21?",
            "EA Vancouver made FIFA 21. Great football game.",
            "web_search",
            "https://en.wikipedia.org/wiki/FIFA_21",
            "high",
        )
        assert "saved" in result.lower()
        mem.save.assert_called_once()

    def test_save_rejected_returns_message(self, mock_collection):
        mem = MagicMock()
        mem.save.return_value = (False, "Rejected: answer too short to be useful")
        tools = _make_tools(mock_collection, memory=mem)
        save_mem = tools[4]
        result = save_mem("Q?", "No.", "web_search", "", "low")
        assert "not saved" in result.lower()

    def test_entry_fields_passed_to_memory(self, mock_collection):
        mem = _make_mock_memory()
        tools = _make_tools(mock_collection, memory=mem)
        save_mem = tools[4]
        save_mem("Who made FIFA 21?", "EA Vancouver.", "web_search",
                 "https://a.com, https://b.com", "high")
        call_args = mem.save.call_args[0][0]
        assert call_args["question"] == "Who made FIFA 21?"
        assert call_args["confidence"] == "high"
        assert call_args["source_type"] == "web_search"
        assert isinstance(call_args["sources"], list)
        assert len(call_args["sources"]) == 2


# ---------------------------------------------------------------------------
# summarize_game_profile
# ---------------------------------------------------------------------------

class TestSummarizeGameProfile:
    def _with_llm_response(self, collection, response_text: str):
        settings = _make_settings()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=response_text)
        with patch("udaplay.tools.LLM", return_value=mock_llm):
            tools = make_game_tools(collection, settings=settings)
        return tools[5]

    def test_returns_string(self, mock_collection):
        profile_text = (
            "Title      : Super Mario 64\n"
            "Publisher  : Nintendo\n"
            "Platform   : Nintendo 64\n"
            "Genre      : Platformer\n"
            "Release    : 1996\n"
            "Summary    : A groundbreaking 3D platformer.\n"
        )
        summarize = self._with_llm_response(mock_collection, profile_text)
        result = summarize("Game: Super Mario 64. Platform: Nintendo 64...")
        assert isinstance(result, str)
        assert "Super Mario 64" in result

    def test_llm_error_falls_back_gracefully(self, mock_collection):
        settings = _make_settings()
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("LLM timeout")
        with patch("udaplay.tools.LLM", return_value=mock_llm):
            tools = make_game_tools(mock_collection, settings=settings)
        summarize = tools[5]
        result = summarize("Game: Super Mario 64. Platform: Nintendo 64.")
        assert isinstance(result, str)
        assert "Could not generate" in result or "Nintendo 64" in result


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

class TestFormatReport:
    @pytest.fixture
    def fmt(self, mock_collection):
        tools = _make_tools(mock_collection)
        return tools[6]

    def test_returns_valid_json(self, fmt):
        result = fmt(
            "Who made FIFA 21?", "EA Vancouver.", "high", "web_search",
            "search_memory, retrieve_game, evaluate_retrieval, game_web_search",
            "https://en.wikipedia.org/wiki/FIFA_21", "true", "false",
        )
        assert isinstance(json.loads(result), dict)

    def test_all_fields_present(self, fmt):
        result = json.loads(fmt(
            "Q?", "Answer.", "high", "local_rag",
            "retrieve_game, evaluate_retrieval",
            "009.json", "false", "false",
        ))
        expected = {
            "question", "answer", "confidence", "source_type",
            "tools_used", "sources", "web_search_used", "memory_used"
        }
        assert expected.issubset(set(result.keys()))

    def test_bool_fields_parsed_correctly(self, fmt):
        result = json.loads(fmt(
            "Q?", "Answer.", "high", "web_search",
            "game_web_search", "https://a.com", "true", "false",
        ))
        assert result["web_search_used"] is True
        assert result["memory_used"] is False

    def test_tools_used_is_list(self, fmt):
        result = json.loads(fmt(
            "Q?", "Answer.", "medium", "local_rag",
            "search_memory, retrieve_game, evaluate_retrieval",
            "009.json", "false", "false",
        ))
        assert isinstance(result["tools_used"], list)
        assert "search_memory" in result["tools_used"]

    def test_sources_is_list(self, fmt):
        result = json.loads(fmt(
            "Q?", "Answer.", "high", "web_search", "game_web_search",
            "https://a.com, https://b.com", "true", "false",
        ))
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) == 2

    def test_empty_sources(self, fmt):
        result = json.loads(fmt("Q?", "Answer.", "high", "local_rag", "retrieve_game", "", "false", "false"))
        assert result["sources"] == []

    @pytest.mark.parametrize("source_type", ["local_rag", "web_search", "memory"])
    def test_all_source_types(self, fmt, source_type):
        result = json.loads(fmt("Q?", "Answer.", "high", source_type, "tool", "", "false", "false"))
        assert result["source_type"] == source_type


# ---------------------------------------------------------------------------
# EvaluationReport model
# ---------------------------------------------------------------------------

class TestEvaluationReport:
    def test_valid_high_confidence(self):
        r = EvaluationReport(confidence="high", useful=True,
                             description="Match.", needs_web_search=False)
        assert r.confidence == "high"

    def test_serialise_to_json(self):
        r = EvaluationReport(confidence="low", useful=False,
                             description="No match.", needs_web_search=True)
        data = json.loads(r.model_dump_json())
        assert data["needs_web_search"] is True

    def test_missing_required_field_raises(self):
        with pytest.raises(Exception):
            EvaluationReport(useful=True, description="x", needs_web_search=False)
