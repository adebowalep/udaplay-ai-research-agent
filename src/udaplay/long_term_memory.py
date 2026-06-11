"""
UdaPlay long-term memory — JSON-backed persistent store.

Why JSON over SQLite
--------------------
* Zero extra dependencies and no schema migrations.
* Human-readable: reviewers can inspect the file directly.
* Simple keyword search is sufficient for ≤ a few thousand entries.
* Trivially mockable in tests via ``tmp_path``.

Entry schema
------------
Every saved entry is a dict with these keys:

    question    : str   — the user's original question
    answer      : str   — the final answer
    source_type : str   — "local_rag" | "web_search" | "memory"
    sources     : list  — citation URLs or file names
    confidence  : str   — "high" | "medium" | "low"
    timestamp   : str   — ISO-8601 UTC datetime
    tags        : list  — auto-generated keyword tags

Usage
-----
>>> from udaplay.long_term_memory import LongTermMemory
>>> ltm = LongTermMemory("data/memory/long_term_memory.json")
>>> ltm.save({"question": "...", "answer": "...", ...})
>>> results = ltm.search("FIFA 21 developer")
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Stop-words excluded from auto-generated tags
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "of", "and",
    "or", "but", "for", "with", "was", "were", "are", "be", "been", "by",
    "do", "did", "does", "has", "have", "had", "not", "this", "that",
    "what", "when", "where", "who", "how", "which", "its", "i", "my",
    "me", "we", "you", "your", "tell", "about", "give", "list", "can",
})

# Patterns whose presence in any value field means the entry is rejected
_SENSITIVE_PATTERNS = (
    r"sk-[A-Za-z0-9]{10,}",  # OpenAI keys
    r"tvly-[A-Za-z0-9]{10,}",  # Tavily keys
    r"api[_\-]?key\s*=\s*\S+",
    r"password\s*=\s*\S+",
    r"secret\s*=\s*\S+",
    r"token\s*=\s*\S+",
)
_SENSITIVE_RE = re.compile("|".join(_SENSITIVE_PATTERNS), re.IGNORECASE)


def _has_sensitive_data(text: str) -> bool:
    return bool(_SENSITIVE_RE.search(text))


def _auto_tags(text: str) -> list[str]:
    """Extract up to 8 meaningful keywords from text."""
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    seen: dict[str, int] = {}
    for w in words:
        if w not in _STOP_WORDS and len(w) > 2:
            seen[w] = seen.get(w, 0) + 1
    # Sort by frequency, take top 8
    return [w for w, _ in sorted(seen.items(), key=lambda x: -x[1])[:8]]


def _keyword_score(query: str, entry: dict) -> int:
    """Simple overlap score between query tokens and entry text."""
    tokens = set(re.findall(r"[a-zA-Z0-9]+", query.lower())) - _STOP_WORDS
    haystack = " ".join([
        entry.get("question", ""),
        entry.get("answer", ""),
        " ".join(entry.get("tags", [])),
    ]).lower()
    return sum(1 for t in tokens if t in haystack)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LongTermMemory:
    """
    Persistent, append-only JSON store for UdaPlay agent answers.

    Parameters
    ----------
    file_path : str | Path
        Path to the JSON file.  Parent directories are created automatically.
        Defaults to ``data/memory/long_term_memory.json`` relative to the
        project root (resolved at call time).
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _load(self) -> list[dict]:
        if not self.file_path.exists():
            return []
        try:
            with open(self.file_path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []

    def _dump(self, entries: list[dict]) -> None:
        with open(self.file_path, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def all_entries(self) -> list[dict]:
        """Return every saved entry (most recent last)."""
        return self._load()

    def count(self) -> int:
        """Return the number of stored entries."""
        return len(self._load())

    def save(self, entry: dict) -> Tuple[bool, str]:
        """
        Validate and append *entry* to the store.

        Returns
        -------
        (True, "Saved")          on success
        (False, reason_string)   if the entry was rejected
        """
        answer = str(entry.get("answer", "")).strip()
        question = str(entry.get("question", "")).strip()

        # Reject empties
        if not answer:
            return False, "Rejected: answer is empty"
        if not question:
            return False, "Rejected: question is empty"

        # Reject short, useless answers
        if len(answer) < 10:
            return False, "Rejected: answer too short to be useful"

        # Reject anything that looks like it contains secrets
        for field_val in [question, answer] + list(entry.get("sources", [])):
            if _has_sensitive_data(str(field_val)):
                return False, "Rejected: entry appears to contain sensitive data"

        # Normalise / fill defaults
        now = datetime.now(timezone.utc).isoformat()
        sources = entry.get("sources", [])
        if isinstance(sources, str):
            sources = [s.strip() for s in sources.split(",") if s.strip()]

        normalised = {
            "question":    question,
            "answer":      answer,
            "source_type": entry.get("source_type", "unknown"),
            "sources":     sources,
            "confidence":  entry.get("confidence", "medium"),
            "timestamp":   entry.get("timestamp", now),
            "tags":        entry.get("tags") or _auto_tags(question + " " + answer),
        }

        entries = self._load()
        entries.append(normalised)
        self._dump(entries)
        return True, "Saved"

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Return up to *top_k* entries most relevant to *query*.

        Uses simple keyword overlap scoring — fast, dependency-free, and
        sufficient for the memory sizes this project targets.
        """
        if not query.strip():
            return []

        entries = self._load()
        scored = [
            (entry, _keyword_score(query, entry))
            for entry in entries
        ]
        # Keep only entries with at least one token match
        scored = [(e, s) for e, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]

    def clear(self) -> None:
        """Erase all entries (useful in tests)."""
        self._dump([])
