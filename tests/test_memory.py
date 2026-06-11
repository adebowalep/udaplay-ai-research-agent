"""
Tests for:
  - udaplay.memory        — ShortTermMemory (in-session conversation store)
  - udaplay.long_term_memory — LongTermMemory (persistent JSON-backed store)
"""
import json
import pytest

from udaplay.memory import ShortTermMemory, SessionNotFoundError
from udaplay.long_term_memory import LongTermMemory


# ===========================================================================
# ShortTermMemory
# ===========================================================================

class TestShortTermMemoryBasics:
    def test_default_session_exists(self):
        mem = ShortTermMemory()
        assert "default" in mem.get_all_sessions()

    def test_add_and_retrieve(self):
        mem = ShortTermMemory()
        mem.add("hello")
        assert mem.get_last_object() == "hello"

    def test_add_multiple(self):
        mem = ShortTermMemory()
        mem.add("first")
        mem.add("second")
        mem.add("third")
        objects = mem.get_all_objects()
        assert len(objects) == 3
        assert objects[-1] == "third"

    def test_get_all_empty(self):
        mem = ShortTermMemory()
        assert mem.get_all_objects() == []

    def test_get_last_empty(self):
        mem = ShortTermMemory()
        assert mem.get_last_object() is None

    def test_pop_removes_last(self):
        mem = ShortTermMemory()
        mem.add("a")
        mem.add("b")
        popped = mem.pop()
        assert popped == "b"
        assert len(mem.get_all_objects()) == 1

    def test_pop_empty(self):
        mem = ShortTermMemory()
        assert mem.pop() is None


class TestShortTermMemorySessions:
    def test_create_session(self):
        mem = ShortTermMemory()
        result = mem.create_session("session_a")
        assert result is True
        assert "session_a" in mem.get_all_sessions()

    def test_create_duplicate_session(self):
        mem = ShortTermMemory()
        mem.create_session("s1")
        result = mem.create_session("s1")
        assert result is False

    def test_sessions_are_isolated(self):
        mem = ShortTermMemory()
        mem.create_session("s1")
        mem.create_session("s2")
        mem.add("item_in_s1", session_id="s1")
        mem.add("item_in_s2", session_id="s2")
        assert mem.get_all_objects("s1") == ["item_in_s1"]
        assert mem.get_all_objects("s2") == ["item_in_s2"]

    def test_add_to_missing_session_raises(self):
        mem = ShortTermMemory()
        with pytest.raises(SessionNotFoundError):
            mem.add("something", session_id="ghost_session")

    def test_delete_session(self):
        mem = ShortTermMemory()
        mem.create_session("deletable")
        mem.delete_session("deletable")
        assert "deletable" not in mem.get_all_sessions()

    def test_cannot_delete_default_session(self):
        mem = ShortTermMemory()
        with pytest.raises(ValueError):
            mem.delete_session("default")

    def test_reset_session(self):
        mem = ShortTermMemory()
        mem.add("x")
        mem.add("y")
        mem.reset()
        assert mem.get_all_objects() == []

    def test_reset_specific_session(self):
        mem = ShortTermMemory()
        mem.create_session("s")
        mem.add("item", session_id="s")
        mem.add("default_item")
        mem.reset("s")
        assert mem.get_all_objects("s") == []
        assert mem.get_all_objects() == ["default_item"]


class TestShortTermMemoryMultiTurn:
    def test_stores_and_retrieves_dict_objects(self):
        mem = ShortTermMemory()
        turn1 = {"question": "Who made Mario?", "answer": "Nintendo"}
        turn2 = {"question": "When was Mario 64?", "answer": "1996"}
        mem.add(turn1)
        mem.add(turn2)
        assert mem.get_last_object() == turn2

    def test_deep_copy_prevents_mutation(self):
        mem = ShortTermMemory()
        original = {"key": "value"}
        mem.add(original)
        retrieved = mem.get_last_object()
        retrieved["key"] = "mutated"
        assert mem.get_last_object()["key"] == "value"


# ===========================================================================
# LongTermMemory
# ===========================================================================

GOOD_ENTRY = {
    "question":    "Who developed FIFA 21?",
    "answer":      "FIFA 21 was developed by EA Vancouver and published by Electronic Arts.",
    "source_type": "web_search",
    "sources":     ["https://en.wikipedia.org/wiki/FIFA_21"],
    "confidence":  "high",
}

POKEMON_ENTRY = {
    "question":    "When was Pokémon Gold and Silver released?",
    "answer":      "Pokémon Gold and Silver were released in 1999 for the Game Boy Color.",
    "source_type": "local_rag",
    "sources":     ["006.json"],
    "confidence":  "high",
}


@pytest.fixture
def ltm(tmp_path):
    """Fresh LongTermMemory backed by a temp file."""
    return LongTermMemory(tmp_path / "memory" / "test_ltm.json")


class TestLongTermMemorySave:
    def test_save_valid_entry(self, ltm):
        ok, reason = ltm.save(GOOD_ENTRY.copy())
        assert ok is True
        assert "saved" in reason.lower()

    def test_saved_entry_is_retrievable(self, ltm):
        ltm.save(GOOD_ENTRY.copy())
        entries = ltm.all_entries()
        assert len(entries) == 1
        assert entries[0]["question"] == GOOD_ENTRY["question"]

    def test_timestamp_auto_added(self, ltm):
        entry = GOOD_ENTRY.copy()
        entry.pop("timestamp", None)
        ltm.save(entry)
        saved = ltm.all_entries()[0]
        assert "timestamp" in saved
        assert "T" in saved["timestamp"] or saved["timestamp"]  # ISO format

    def test_tags_auto_generated(self, ltm):
        entry = GOOD_ENTRY.copy()
        entry.pop("tags", None)
        ltm.save(entry)
        saved = ltm.all_entries()[0]
        assert isinstance(saved["tags"], list)
        assert len(saved["tags"]) > 0

    def test_multiple_saves_append(self, ltm):
        ltm.save(GOOD_ENTRY.copy())
        ltm.save(POKEMON_ENTRY.copy())
        assert ltm.count() == 2

    def test_sources_string_normalised_to_list(self, ltm):
        entry = GOOD_ENTRY.copy()
        entry["sources"] = "https://a.com, https://b.com"
        ltm.save(entry)
        saved = ltm.all_entries()[0]
        assert isinstance(saved["sources"], list)
        assert len(saved["sources"]) == 2

    def test_persists_across_instances(self, tmp_path):
        path = tmp_path / "memory" / "ltm.json"
        ltm1 = LongTermMemory(path)
        ltm1.save(GOOD_ENTRY.copy())

        ltm2 = LongTermMemory(path)
        assert ltm2.count() == 1
        assert ltm2.all_entries()[0]["question"] == GOOD_ENTRY["question"]


class TestLongTermMemoryReject:
    def test_rejects_empty_answer(self, ltm):
        entry = GOOD_ENTRY.copy()
        entry["answer"] = ""
        ok, reason = ltm.save(entry)
        assert ok is False
        assert "empty" in reason.lower()

    def test_rejects_empty_question(self, ltm):
        entry = GOOD_ENTRY.copy()
        entry["question"] = "   "
        ok, reason = ltm.save(entry)
        assert ok is False

    def test_rejects_very_short_answer(self, ltm):
        entry = GOOD_ENTRY.copy()
        entry["answer"] = "No."
        ok, reason = ltm.save(entry)
        assert ok is False
        assert "short" in reason.lower()

    def test_rejects_openai_key_in_answer(self, ltm):
        entry = GOOD_ENTRY.copy()
        entry["answer"] = "The API key is sk-abc1234567890abcdefghijk and it works."
        ok, reason = ltm.save(entry)
        assert ok is False
        assert "sensitive" in reason.lower()

    def test_rejects_tavily_key_in_question(self, ltm):
        entry = GOOD_ENTRY.copy()
        entry["question"] = "My tvly-abc1234567890abcdef key is exposed?"
        ok, reason = ltm.save(entry)
        assert ok is False

    def test_rejects_api_key_pattern_in_sources(self, ltm):
        entry = GOOD_ENTRY.copy()
        entry["sources"] = ["api_key=sk-abc1234567890abcdef"]
        ok, reason = ltm.save(entry)
        assert ok is False

    def test_does_not_save_rejected_entry(self, ltm):
        entry = GOOD_ENTRY.copy()
        entry["answer"] = ""
        ltm.save(entry)
        assert ltm.count() == 0


class TestLongTermMemorySearch:
    def test_finds_relevant_entry(self, ltm):
        ltm.save(GOOD_ENTRY.copy())
        results = ltm.search("FIFA 21 developer")
        assert len(results) == 1
        assert "FIFA" in results[0]["question"]

    def test_returns_empty_for_no_match(self, ltm):
        ltm.save(GOOD_ENTRY.copy())
        results = ltm.search("minecraft bedrock edition")
        assert results == []

    def test_returns_top_k_results(self, ltm):
        for i in range(5):
            ltm.save({
                "question":    f"Question about FIFA {i}",
                "answer":      f"FIFA {i} was developed by EA Sports. Great football game.",
                "source_type": "web_search",
                "sources":     [f"https://example.com/{i}"],
                "confidence":  "high",
            })
        results = ltm.search("FIFA football EA Sports", top_k=3)
        assert len(results) <= 3

    def test_more_relevant_entry_ranked_first(self, ltm):
        ltm.save(GOOD_ENTRY.copy())     # FIFA-focused
        ltm.save(POKEMON_ENTRY.copy())  # Pokemon-focused
        results = ltm.search("FIFA 21 EA Vancouver developer")
        assert "FIFA" in results[0]["question"]

    def test_empty_query_returns_empty(self, ltm):
        ltm.save(GOOD_ENTRY.copy())
        assert ltm.search("") == []
        assert ltm.search("   ") == []


class TestLongTermMemoryClear:
    def test_clear_removes_all_entries(self, ltm):
        ltm.save(GOOD_ENTRY.copy())
        ltm.save(POKEMON_ENTRY.copy())
        ltm.clear()
        assert ltm.count() == 0
        assert ltm.all_entries() == []

    def test_save_after_clear_works(self, ltm):
        ltm.save(GOOD_ENTRY.copy())
        ltm.clear()
        ltm.save(POKEMON_ENTRY.copy())
        assert ltm.count() == 1
