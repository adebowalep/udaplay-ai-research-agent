"""
UdaPlay agent tools.

Provides ``make_game_tools`` which creates all tools the UdaPlay agent uses:

    Core (original) tools
    1. retrieve_game        — semantic search in local ChromaDB
    2. evaluate_retrieval   — LLM-as-judge quality assessment
    3. game_web_search      — Tavily web search fallback

    Enhanced tools (long-term memory + output formatting)
    4. search_memory        — search persistent long-term memory
    5. save_memory          — persist a useful answer to memory
    6. summarize_game_profile — produce a clean, structured game profile
    7. format_report        — format the final answer as a structured JSON report

Tools are created as closures so they capture the ChromaDB collection,
API keys, and memory instance at construction time.

Usage
-----
>>> from udaplay.config import Settings
>>> from udaplay.long_term_memory import LongTermMemory
>>> from udaplay.tools import make_game_tools
>>> ltm = LongTermMemory("data/memory/long_term_memory.json")
>>> tools = make_game_tools(collection, settings=Settings(), memory=ltm)
"""
from __future__ import annotations

import json
from typing import Tuple

from pydantic import BaseModel, Field

from udaplay.config import Settings
from udaplay.llm import LLM
from udaplay.tooling import tool


# ---------------------------------------------------------------------------
# Evaluation model
# ---------------------------------------------------------------------------

class EvaluationReport(BaseModel):
    """Structured output from the retrieval quality evaluator."""

    confidence: str = Field(
        description="Confidence level: 'high', 'medium', or 'low'"
    )
    useful: bool = Field(
        description="True if retrieved documents are sufficient to answer the question"
    )
    description: str = Field(
        description="Brief explanation of why the documents are or are not useful"
    )
    needs_web_search: bool = Field(
        description="True if a web search is required for an accurate answer"
    )


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def make_game_tools(
    collection,
    settings: Settings | None = None,
    memory=None,  # LongTermMemory | None
) -> Tuple:
    """
    Create and return the UdaPlay agent tools.

    Parameters
    ----------
    collection : chromadb.Collection
        An already-connected ChromaDB collection with game documents.
    settings : Settings, optional
        Runtime settings. If None, loads from environment variables.
    memory : LongTermMemory, optional
        Long-term memory instance. If None, memory tools return a
        "memory not configured" message instead of raising.

    Returns
    -------
    tuple of Tool
        ``(retrieve_game, evaluate_retrieval, game_web_search,
           search_memory, save_memory, summarize_game_profile, format_report)``
    """
    if settings is None:
        settings = Settings()

    _eval_llm      = LLM(model=settings.llm_model, temperature=0.0)
    _summarize_llm = LLM(model=settings.llm_model, temperature=0.0)
    _tavily_api_key = settings.tavily_api_key
    _n_results      = settings.n_retrieval_results
    _memory         = memory  # may be None; tools handle gracefully

    # ------------------------------------------------------------------
    # Tool 1: retrieve_game
    # ------------------------------------------------------------------

    @tool
    def retrieve_game(query: str) -> str:
        """
        Search the local video game vector database for information relevant
        to the query.  Returns the top matching games with metadata (title,
        platform, publisher, year, genre) and a relevance score.
        Always call this tool before any web search.

        Args:
            query: A natural language question about the video game industry.
        """
        try:
            results = collection.query(
                query_texts=[query],
                n_results=_n_results,
                include=["documents", "metadatas", "distances"],
            )

            docs      = results["documents"][0]
            metas     = results["metadatas"][0]
            distances = results["distances"][0]

            if not docs:
                return "No documents found in the local database for this query."

            lines = [f"Retrieved {len(docs)} result(s) from local database:\n"]
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
                similarity = round(1 - dist, 4)
                lines.append(
                    f"[Result {i}]\n"
                    f"  Title     : {meta.get('title', 'N/A')}\n"
                    f"  Platform  : {meta.get('platform', 'N/A')}\n"
                    f"  Publisher : {meta.get('publisher', 'N/A')}\n"
                    f"  Genre     : {meta.get('genre', 'N/A')}\n"
                    f"  Year      : {meta.get('release_year', 'N/A')}\n"
                    f"  Source    : {meta.get('source_file', 'N/A')}\n"
                    f"  Similarity: {similarity}\n"
                    f"  Content   : {doc}\n"
                )
            return "\n".join(lines)

        except Exception as exc:
            return f"Error querying local database: {exc}"

    # ------------------------------------------------------------------
    # Tool 2: evaluate_retrieval
    # ------------------------------------------------------------------

    @tool
    def evaluate_retrieval(question: str, retrieved_docs: str) -> str:
        """
        Evaluate whether the locally retrieved documents are sufficient to
        answer the user's question.  Uses an LLM judge to assess confidence
        and relevance.

        Args:
            question: The original user question.
            retrieved_docs: The documents returned by retrieve_game.
        """
        prompt = (
            "You are an evaluation assistant for a video game research agent.\n"
            "Your task is to evaluate whether the retrieved documents are sufficient "
            "to accurately answer the user's question.\n\n"
            f"User question: {question}\n\n"
            f"Retrieved documents:\n{retrieved_docs}\n\n"
            "Evaluate the documents and decide:\n"
            "- confidence: 'high' if documents directly answer the question, "
            "'medium' if partially relevant, 'low' if not relevant.\n"
            "- useful: true if the documents are sufficient to answer the question.\n"
            "- description: brief explanation of your evaluation.\n"
            "- needs_web_search: true if a web search is needed for a complete answer.\n"
            "\nRespond with valid JSON only, matching the EvaluationReport schema."
        )

        try:
            response = _eval_llm.invoke(prompt, response_format=EvaluationReport)
            content = getattr(response, "content", None) or ""
            try:
                parsed = json.loads(content)
                return json.dumps(parsed)
            except json.JSONDecodeError:
                return content

        except Exception as exc:
            fallback = EvaluationReport(
                confidence="low",
                useful=False,
                description=f"Evaluation error: {exc}. Defaulting to web search.",
                needs_web_search=True,
            )
            return fallback.model_dump_json()

    # ------------------------------------------------------------------
    # Tool 3: game_web_search
    # ------------------------------------------------------------------

    @tool
    def game_web_search(question: str) -> str:
        """
        Search the web for video game information using Tavily when the local
        database does not have a sufficient answer.  Returns web results with
        source URLs.

        Args:
            question: A question about the video game industry.
        """
        if not _tavily_api_key:
            return (
                "ERROR: TAVILY_API_KEY is not set. "
                "To enable web search, add TAVILY_API_KEY=your_key to your .env file. "
                "Get a free key at https://app.tavily.com"
            )

        try:
            from tavily import TavilyClient  # optional dependency

            client = TavilyClient(api_key=_tavily_api_key)
            response = client.search(
                query=question,
                search_depth="basic",
                max_results=3,
                include_answer=True,
            )

            lines = ["Web search results from Tavily:\n"]

            if response.get("answer"):
                lines.append(f"Direct answer: {response['answer']}\n")

            for i, result in enumerate(response.get("results", []), 1):
                lines.append(
                    f"[Web Result {i}]\n"
                    f"  Title  : {result.get('title', 'N/A')}\n"
                    f"  URL    : {result.get('url', 'N/A')}\n"
                    f"  Content: {result.get('content', '')[:300]}\n"
                )

            return "\n".join(lines) if len(lines) > 1 else "Web search returned no results."

        except ImportError:
            return "ERROR: tavily-python not installed. Run: pip install tavily-python"
        except Exception as exc:
            return f"Web search error: {exc}"

    # ------------------------------------------------------------------
    # Tool 4: search_memory
    # ------------------------------------------------------------------

    @tool
    def search_memory(query: str) -> str:
        """
        Search long-term memory for previous answers about this topic.
        Call this FIRST, before retrieve_game, to check whether UdaPlay has
        already researched this question.  Returns matching entries with
        timestamps and sources.

        Args:
            query: Keywords or a question to search previous research for.
        """
        if _memory is None:
            return json.dumps({"found": False, "matches": [], "note": "Memory not configured."})

        matches = _memory.search(query, top_k=3)
        if not matches:
            return json.dumps({"found": False, "matches": []})

        # Strip bulky answer text for the search result preview
        previews = []
        for m in matches:
            previews.append({
                "question":    m.get("question", ""),
                "answer":      m.get("answer", "")[:400],
                "source_type": m.get("source_type", ""),
                "confidence":  m.get("confidence", ""),
                "timestamp":   m.get("timestamp", ""),
                "sources":     m.get("sources", []),
            })
        return json.dumps({"found": True, "matches": previews})

    # ------------------------------------------------------------------
    # Tool 5: save_memory
    # ------------------------------------------------------------------

    @tool
    def save_memory(
        question: str,
        answer: str,
        source_type: str,
        sources: str,
        confidence: str,
    ) -> str:
        """
        Save a useful final answer to long-term memory so it can be reused
        in future sessions.  Do NOT call this for failed, empty, or very
        short answers.  Do NOT include API keys or file paths in any field.

        Args:
            question: The original user question.
            answer: The final answer text.
            source_type: Where the answer came from — "local_rag", "web_search", or "memory".
            sources: Comma-separated list of citation URLs or filenames.
            confidence: Answer confidence — "high", "medium", or "low".
        """
        if _memory is None:
            return "Memory not configured — answer not saved."

        entry = {
            "question":    question,
            "answer":      answer,
            "source_type": source_type,
            "sources":     [s.strip() for s in sources.split(",") if s.strip()],
            "confidence":  confidence,
        }
        ok, reason = _memory.save(entry)
        return f"Memory saved successfully." if ok else f"Memory not saved: {reason}"

    # ------------------------------------------------------------------
    # Tool 6: summarize_game_profile
    # ------------------------------------------------------------------

    @tool
    def summarize_game_profile(game_info: str) -> str:
        """
        Given raw retrieved game information, produce a clean, structured
        game profile that is easy to read.  Use this when the user asks for
        details about a specific game and local RAG has relevant results.

        Args:
            game_info: Raw text from retrieve_game output.
        """
        prompt = (
            "You are a video game encyclopedia assistant.\n"
            "Given the following raw game database entry, produce a clean, "
            "structured profile in this exact format:\n\n"
            "Title      : <name>\n"
            "Developer  : <developer if known, else N/A>\n"
            "Publisher  : <publisher>\n"
            "Platform   : <platform>\n"
            "Genre      : <genre>\n"
            "Release    : <year>\n"
            "Summary    : <one or two sentence description>\n\n"
            "Raw data:\n"
            f"{game_info}\n\n"
            "Output ONLY the profile, no extra commentary."
        )
        try:
            response = _summarize_llm.invoke(prompt)
            return getattr(response, "content", str(response)).strip()
        except Exception as exc:
            return f"Could not generate profile: {exc}\n\nRaw data:\n{game_info}"

    # ------------------------------------------------------------------
    # Tool 7: format_report
    # ------------------------------------------------------------------

    @tool
    def format_report(
        question: str,
        answer: str,
        confidence: str,
        source_type: str,
        tools_used: str,
        sources: str,
        web_search_used: str,
        memory_used: str,
    ) -> str:
        """
        Format the final answer as a structured JSON report.  Always call
        this as the last step before finishing, so the answer is machine-
        readable and consistently structured.

        Args:
            question: The original user question.
            answer: The final answer text.
            confidence: Answer confidence — "high", "medium", or "low".
            source_type: Where the answer came from — "local_rag", "web_search", or "memory".
            tools_used: Comma-separated list of tools called during this run.
            sources: Comma-separated list of citation URLs or filenames.
            web_search_used: "true" or "false".
            memory_used: "true" or "false".
        """
        def _bool(s: str) -> bool:
            return str(s).strip().lower() in ("true", "1", "yes")

        def _list(s: str) -> list[str]:
            return [x.strip() for x in s.split(",") if x.strip()]

        report = {
            "question":        question,
            "answer":          answer,
            "confidence":      confidence,
            "source_type":     source_type,
            "tools_used":      _list(tools_used),
            "sources":         _list(sources),
            "web_search_used": _bool(web_search_used),
            "memory_used":     _bool(memory_used),
        }
        return json.dumps(report, ensure_ascii=False, indent=2)

    return (
        retrieve_game,
        evaluate_retrieval,
        game_web_search,
        search_memory,
        save_memory,
        summarize_game_profile,
        format_report,
    )
