"""
Tests for the UdaPlay agent workflow — three decision paths.

  Path 1 — Local RAG sufficient:
      search_memory → retrieve_game → evaluate_retrieval (high) → format_report

  Path 2 — Memory fallback:
      search_memory (match found) → retrieve_game → evaluate_retrieval (low)
      → agent uses memory answer → format_report

  Path 3 — Web search fallback:
      search_memory (no match) → retrieve_game → evaluate_retrieval (low)
      → game_web_search → save_memory → format_report

Patching note: LLM is instantiated inside Agent._llm_step on every invoke()
call, so fixtures yield *inside* the patch context to keep it active.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from udaplay.agents import Agent
from udaplay.tooling import Tool


# ---------------------------------------------------------------------------
# String fixtures
# ---------------------------------------------------------------------------

LOCAL_RAG_RESULT = (
    "Retrieved 1 result(s) from local database:\n"
    "[Result 1]\n"
    "  Title     : Pokémon Gold and Silver\n"
    "  Platform  : Game Boy Color\n"
    "  Publisher : Nintendo\n"
    "  Genre     : Role-playing\n"
    "  Year      : 1999\n"
    "  Source    : 006.json\n"
    "  Similarity: 0.96\n"
    "  Content   : Game: Pokémon Gold and Silver.\n"
)

HIGH_CONF_EVAL = json.dumps({
    "confidence": "high", "useful": True,
    "description": "Document directly answers the question.",
    "needs_web_search": False,
})

LOW_CONF_EVAL = json.dumps({
    "confidence": "low", "useful": False,
    "description": "Retrieved documents are not relevant to FIFA 21.",
    "needs_web_search": True,
})

MEMORY_MATCH_RESULT = json.dumps({
    "found": True,
    "matches": [{
        "question":    "Who developed FIFA 21?",
        "answer":      "EA Vancouver developed FIFA 21 for Electronic Arts.",
        "source_type": "web_search",
        "confidence":  "high",
        "timestamp":   "2026-06-11T14:58:43",
        "sources":     ["https://en.wikipedia.org/wiki/FIFA_21"],
    }],
})

MEMORY_NO_MATCH = json.dumps({"found": False, "matches": []})

WEB_SEARCH_RESULT = (
    "Web search results from Tavily:\n"
    "Direct answer: FIFA 21 was developed by EA Vancouver.\n"
    "[Web Result 1]\n"
    "  URL    : https://en.wikipedia.org/wiki/FIFA_21\n"
    "  Content: FIFA 21 is a football simulation game.\n"
)

LOCAL_RAG_ANSWER = (
    "Pokémon Gold and Silver were released in 1999 for the Game Boy Color.\n\n"
    "**Confidence:** high | **Source:** Local RAG | **Citation:** 006.json"
)

MEMORY_ANSWER = (
    "Based on previous research: FIFA 21 was developed by EA Vancouver.\n\n"
    "**Confidence:** high | **Source:** Memory"
)

WEB_ANSWER = (
    "FIFA 21 was developed by EA Vancouver and EA Romania.\n\n"
    "**Confidence:** high | **Source:** Web (Tavily)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tool(name: str, return_value: str) -> Tool:
    # Python 3.13 tightened get_type_hints() — MagicMock is rejected.
    # Use a real function for Tool construction (satisfies signature + hints),
    # then replace .func with the MagicMock so assert_called() works in tests.
    mock = MagicMock(return_value=return_value)

    def fn(*args, **kwargs):
        return mock(*args, **kwargs)

    fn.__name__ = name
    fn.__doc__ = f"Mock {name}"
    t = Tool(func=fn, name=name, description=f"Mock {name}")
    t.func = mock  # swap in the mock so fixture assertions work
    return t


def _tc(name: str, args: str) -> ChatCompletionMessageToolCall:
    """Real ChatCompletionMessageToolCall — pydantic rejects MagicMock in AIMessage."""
    return ChatCompletionMessageToolCall(
        id=f"call_{name}",
        type="function",
        function=Function(name=name, arguments=args),
    )


def _llm_class(turns: list) -> MagicMock:
    """
    Build a mock LLM class whose instance replays *turns* on successive
    invoke() calls.  Each turn is (tool_call | None, content | None).
    """
    responses = []
    for tc_obj, content in turns:
        msg = MagicMock()
        msg.content    = content or ""
        msg.tool_calls = [tc_obj] if tc_obj else None
        msg.token_usage = None
        responses.append(msg)

    instance = MagicMock()
    instance.invoke.side_effect = responses
    return MagicMock(return_value=instance)


# ---------------------------------------------------------------------------
# Fixtures — patch active for the full test via yield
# ---------------------------------------------------------------------------

@pytest.fixture
def local_rag_setup():
    """Path 1: search_memory → retrieve → evaluate (high) → answer."""
    search_mem = _tool("search_memory",       MEMORY_NO_MATCH)
    retrieve   = _tool("retrieve_game",       LOCAL_RAG_RESULT)
    evaluate   = _tool("evaluate_retrieval",  HIGH_CONF_EVAL)
    web        = _tool("game_web_search",     "Should not be called")
    save_mem   = _tool("save_memory",         "Memory saved successfully.")
    summarize  = _tool("summarize_game_profile", "Profile text")
    fmt        = _tool("format_report",       json.dumps({
        "question": "test", "answer": LOCAL_RAG_ANSWER,
        "confidence": "high", "source_type": "local_rag",
        "tools_used": ["search_memory", "retrieve_game", "evaluate_retrieval"],
        "sources": ["006.json"], "web_search_used": False, "memory_used": False,
    }))

    mock_llm = _llm_class([
        (_tc("search_memory",      '{"query": "Pokemon Gold Silver"}'), None),
        (_tc("retrieve_game",      '{"query": "Pokemon Gold Silver"}'), None),
        (_tc("evaluate_retrieval", '{"question": "test", "retrieved_docs": "..."}'), None),
        (_tc("format_report",      '{"question":"test","answer":"...","confidence":"high","source_type":"local_rag","tools_used":"retrieve_game","sources":"006.json","web_search_used":"false","memory_used":"false"}'), None),
        (None, LOCAL_RAG_ANSWER),
    ])

    with patch("udaplay.agents.LLM", mock_llm):
        agent = Agent("gpt-4o-mini", "You are UdaPlay.",
                      [search_mem, retrieve, evaluate, web, save_mem, summarize, fmt],
                      temperature=0.0)
        yield agent, search_mem, retrieve, evaluate, web, save_mem


@pytest.fixture
def memory_fallback_setup():
    """Path 2: search_memory finds match → low local RAG → answer from memory."""
    search_mem = _tool("search_memory",       MEMORY_MATCH_RESULT)
    retrieve   = _tool("retrieve_game",       LOCAL_RAG_RESULT)
    evaluate   = _tool("evaluate_retrieval",  LOW_CONF_EVAL)
    web        = _tool("game_web_search",     "Should not be called")
    save_mem   = _tool("save_memory",         "Memory saved successfully.")
    summarize  = _tool("summarize_game_profile", "Profile text")
    fmt        = _tool("format_report",       json.dumps({
        "question": "test", "answer": MEMORY_ANSWER,
        "confidence": "high", "source_type": "memory",
        "tools_used": ["search_memory", "retrieve_game", "evaluate_retrieval"],
        "sources": ["https://en.wikipedia.org/wiki/FIFA_21"],
        "web_search_used": False, "memory_used": True,
    }))

    mock_llm = _llm_class([
        (_tc("search_memory",      '{"query": "FIFA 21"}'), None),
        (_tc("retrieve_game",      '{"query": "FIFA 21"}'), None),
        (_tc("evaluate_retrieval", '{"question": "test", "retrieved_docs": "..."}'), None),
        (_tc("format_report",      '{"question":"test","answer":"...","confidence":"high","source_type":"memory","tools_used":"search_memory","sources":"","web_search_used":"false","memory_used":"true"}'), None),
        (None, MEMORY_ANSWER),
    ])

    with patch("udaplay.agents.LLM", mock_llm):
        agent = Agent("gpt-4o-mini", "You are UdaPlay.",
                      [search_mem, retrieve, evaluate, web, save_mem, summarize, fmt],
                      temperature=0.0)
        yield agent, search_mem, retrieve, evaluate, web, save_mem


@pytest.fixture
def web_fallback_setup():
    """Path 3: no memory match → low local RAG → web search → save memory."""
    search_mem = _tool("search_memory",       MEMORY_NO_MATCH)
    retrieve   = _tool("retrieve_game",       LOCAL_RAG_RESULT)
    evaluate   = _tool("evaluate_retrieval",  LOW_CONF_EVAL)
    web        = _tool("game_web_search",     WEB_SEARCH_RESULT)
    save_mem   = _tool("save_memory",         "Memory saved successfully.")
    summarize  = _tool("summarize_game_profile", "Profile text")
    fmt        = _tool("format_report",       json.dumps({
        "question": "test", "answer": WEB_ANSWER,
        "confidence": "high", "source_type": "web_search",
        "tools_used": ["search_memory", "retrieve_game", "evaluate_retrieval", "game_web_search", "save_memory"],
        "sources": ["https://en.wikipedia.org/wiki/FIFA_21"],
        "web_search_used": True, "memory_used": False,
    }))

    mock_llm = _llm_class([
        (_tc("search_memory",      '{"query": "FIFA 21"}'), None),
        (_tc("retrieve_game",      '{"query": "FIFA 21"}'), None),
        (_tc("evaluate_retrieval", '{"question": "test", "retrieved_docs": "..."}'), None),
        (_tc("game_web_search",    '{"question": "FIFA 21 developer"}'), None),
        (_tc("save_memory",        '{"question":"test","answer":"...","source_type":"web_search","sources":"https://en.wikipedia.org/wiki/FIFA_21","confidence":"high"}'), None),
        (_tc("format_report",      '{"question":"test","answer":"...","confidence":"high","source_type":"web_search","tools_used":"game_web_search","sources":"https://a.com","web_search_used":"true","memory_used":"false"}'), None),
        (None, WEB_ANSWER),
    ])

    with patch("udaplay.agents.LLM", mock_llm):
        agent = Agent("gpt-4o-mini", "You are UdaPlay.",
                      [search_mem, retrieve, evaluate, web, save_mem, summarize, fmt],
                      temperature=0.0)
        yield agent, search_mem, retrieve, evaluate, web, save_mem


# ---------------------------------------------------------------------------
# Path 1 — Local RAG sufficient
# ---------------------------------------------------------------------------

class TestLocalRagPath:
    def test_search_memory_called_first(self, local_rag_setup):
        agent, search_mem, *_ = local_rag_setup
        agent.invoke("When was Pokémon Gold and Silver released?")
        search_mem.func.assert_called()

    def test_retrieve_game_called(self, local_rag_setup):
        agent, _, retrieve, *_ = local_rag_setup
        agent.invoke("When was Pokémon Gold and Silver released?")
        retrieve.func.assert_called()

    def test_evaluate_retrieval_called(self, local_rag_setup):
        agent, _, _, evaluate, *_ = local_rag_setup
        agent.invoke("When was Pokémon Gold and Silver released?")
        evaluate.func.assert_called()

    def test_web_search_not_called(self, local_rag_setup):
        agent, _, _, _, web, *_ = local_rag_setup
        agent.invoke("When was Pokémon Gold and Silver released?")
        web.func.assert_not_called()

    def test_run_completes(self, local_rag_setup):
        agent, *_ = local_rag_setup
        run = agent.invoke("When was Pokémon Gold and Silver released?")
        assert run is not None
        assert run.get_final_state() is not None


# ---------------------------------------------------------------------------
# Path 2 — Memory fallback (no web search)
# ---------------------------------------------------------------------------

class TestMemoryFallbackPath:
    def test_search_memory_called(self, memory_fallback_setup):
        agent, search_mem, *_ = memory_fallback_setup
        agent.invoke("Who developed FIFA 21?")
        search_mem.func.assert_called()

    def test_retrieve_game_called(self, memory_fallback_setup):
        agent, _, retrieve, *_ = memory_fallback_setup
        agent.invoke("Who developed FIFA 21?")
        retrieve.func.assert_called()

    def test_web_search_not_called(self, memory_fallback_setup):
        agent, _, _, _, web, *_ = memory_fallback_setup
        agent.invoke("Who developed FIFA 21?")
        web.func.assert_not_called()

    def test_run_completes(self, memory_fallback_setup):
        agent, *_ = memory_fallback_setup
        run = agent.invoke("Who developed FIFA 21?")
        assert run is not None
        assert run.get_final_state() is not None


# ---------------------------------------------------------------------------
# Path 3 — Web search fallback + save memory
# ---------------------------------------------------------------------------

class TestWebFallbackPath:
    def test_search_memory_called(self, web_fallback_setup):
        agent, search_mem, *_ = web_fallback_setup
        agent.invoke("Who developed FIFA 21?")
        search_mem.func.assert_called()

    def test_retrieve_game_called(self, web_fallback_setup):
        agent, _, retrieve, *_ = web_fallback_setup
        agent.invoke("Who developed FIFA 21?")
        retrieve.func.assert_called()

    def test_evaluate_retrieval_called(self, web_fallback_setup):
        agent, _, _, evaluate, *_ = web_fallback_setup
        agent.invoke("Who developed FIFA 21?")
        evaluate.func.assert_called()

    def test_web_search_called_as_fallback(self, web_fallback_setup):
        agent, _, _, _, web, *_ = web_fallback_setup
        agent.invoke("Who developed FIFA 21?")
        web.func.assert_called()

    def test_save_memory_called_after_web_search(self, web_fallback_setup):
        agent, *_, save_mem = web_fallback_setup
        agent.invoke("Who developed FIFA 21?")
        save_mem.func.assert_called()

    def test_run_completes(self, web_fallback_setup):
        agent, *_ = web_fallback_setup
        run = agent.invoke("Who developed FIFA 21?")
        assert run is not None
        assert run.get_final_state() is not None


# ---------------------------------------------------------------------------
# Session memory
# ---------------------------------------------------------------------------

class TestSessionMemory:
    def _simple_llm_class(self):
        msg = MagicMock()
        msg.content = "Answer."
        msg.tool_calls = None
        msg.token_usage = None
        return MagicMock(return_value=MagicMock(invoke=MagicMock(return_value=msg)))

    def test_same_session_accumulates_messages(self):
        mock_llm = self._simple_llm_class()
        with patch("udaplay.agents.LLM", mock_llm):
            agent = Agent("gpt-4o-mini", "You are UdaPlay.", [], temperature=0.0)
            agent.invoke("First question",  session_id="sess")
            agent.invoke("Second question", session_id="sess")
        instance = mock_llm.return_value
        assert instance.invoke.call_count == 2
        first  = instance.invoke.call_args_list[0][0][0]
        second = instance.invoke.call_args_list[1][0][0]
        assert len(second) >= len(first)

    def test_different_sessions_are_isolated(self):
        mock_llm = self._simple_llm_class()
        with patch("udaplay.agents.LLM", mock_llm):
            agent = Agent("gpt-4o-mini", "You are UdaPlay.", [], temperature=0.0)
            agent.invoke("Session A", session_id="session_a")
            agent.invoke("Session B", session_id="session_b")
        instance = mock_llm.return_value
        for call_args in instance.invoke.call_args_list:
            msgs = call_args[0][0]
            assert len(msgs) >= 2
