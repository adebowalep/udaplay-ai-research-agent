# UdaPlay — AI Research Agent for the Video Game Industry

UdaPlay is an AI-powered research agent that answers natural language questions about video games. It first searches a local vector database of game records; if the answer is not found or confidence is low, it falls back to live web search via Tavily. All answers are structured, cited, and persisted to long-term memory.

---

## Project Scenario

Executives, analysts, and gamers want to ask questions like:

- *Who developed FIFA 21?*
- *When was God of War Ragnarök released?*
- *What platform was Pokémon Red launched on?*
- *What is Rockstar Games working on right now?*

UdaPlay handles all of these — routing to local knowledge when possible, and falling back to the web when needed.

---

## Architecture Overview

```
User Question
     │
     ▼
┌─────────────────────┐
│   retrieve_game     │  ← Semantic search in local ChromaDB
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ evaluate_retrieval  │  ← LLM judge: is this sufficient?
└─────────────────────┘
     │
     ├── Confidence HIGH → Answer from local RAG
     │
     └── Confidence LOW  → ┌─────────────────────┐
                           │  game_web_search     │  ← Tavily API
                           └─────────────────────┘
                                    │
                                    ▼
                           Final structured answer
                                    │
                                    ▼
                           agent_memory.json  (long-term memory)
```

The agent is built on a `StateMachine` (in `lib/state_machine.py`) with in-session `ShortTermMemory` (in `lib/memory.py`). Query results are additionally persisted to a JSON file for long-term retention across sessions.

---

## Tools Implemented

### 1. `retrieve_game`

Searches the ChromaDB vector database using OpenAI embeddings (text-embedding-ada-002). Returns the top-3 most semantically similar game records with metadata and relevance scores.

```
Input : query (str) — natural language question
Output: formatted string with title, platform, publisher, genre, year, source file, similarity score
```

### 2. `evaluate_retrieval`

An LLM-as-judge (GPT-4o-mini) that reads the user's question and the retrieved documents and decides whether the local result is sufficient, or whether a web search is required.

```
Input : question (str), retrieved_docs (str)
Output: JSON — { confidence, useful, description, needs_web_search }
```

### 3. `game_web_search`

Web search via the Tavily API. Used only when local retrieval confidence is low. Fails gracefully with a clear error message if `TAVILY_API_KEY` is missing.

```
Input : question (str)
Output: formatted string with web results (title, URL, content snippet)
```

---

## How Part 1 (RAG) Works

Notebook: `starter/Udaplay_01_solution_project.ipynb`

1. Loads all 15 game JSON files from `starter/games/`
2. Formats each game into a rich natural-language document
3. Builds metadata: `title`, `platform`, `genre`, `publisher`, `release_year`, `source_file`
4. Creates a persistent ChromaDB collection at `starter/chromadb/` using OpenAI embeddings
5. Adds all documents to the collection
6. Demonstrates semantic search with 5 example queries

---

## How Part 2 (Agent) Works

Notebook: `starter/Udaplay_02_solution_project.ipynb`

1. Reconnects to the ChromaDB built in Part 1
2. Defines the three tools above
3. Creates an `Agent` instance (from `lib/agents.py`) with the tools and system instructions
4. The agent's internal state machine handles the retrieve → evaluate → [web search] → answer loop
5. A `run_agent_query()` helper extracts the tool trace, structured evaluation, final answer, and citations from each run
6. Results are printed in human-readable and JSON formats, then saved to `agent_memory.json`
7. Runs 5 example queries (4 required by rubric + 1 local RAG hit)

---

## How to Run

### Prerequisites

- Python 3.11+
- OpenAI API key
- Tavily API key (optional — web search gracefully disabled without it)

### Installation

```bash
cd starter
pip install -r requirements.txt
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

### Run Part 1 first (builds the database)

```bash
jupyter notebook Udaplay_01_solution_project.ipynb
```

Run all cells top-to-bottom. This creates `starter/chromadb/`.

### Then run Part 2 (the agent)

```bash
jupyter notebook Udaplay_02_solution_project.ipynb
```

Run all cells top-to-bottom. The agent runs 5 queries and saves results to `starter/agent_memory.json`.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | Used for text-embedding-ada-002 and GPT-4o-mini |
| `TAVILY_API_KEY` | No | Enables web search fallback. Get free key at https://app.tavily.com |

---

## Example Output

```
QUESTION: Who developed FIFA 21?

TOOL USAGE TRACE:
  1. retrieve_game
  2. evaluate_retrieval
  3. game_web_search

RETRIEVAL EVALUATION:
  Confidence      : low
  Useful          : False
  Needs web search: True
  Reason          : Retrieved documents are about Gran Turismo and GTA, not FIFA 21.

WEB SEARCH USED: True

FINAL ANSWER:
FIFA 21 was developed by EA Vancouver and EA Romania, published by Electronic Arts.

---
**Answer:** FIFA 21 was developed by EA Vancouver and EA Romania, published by Electronic Arts.
**Confidence:** low (local) → answered via web search
**Source:** Web search (Tavily)
**Citations:**
- https://en.wikipedia.org/wiki/FIFA_21
---
```

```json
{
  "question": "Who developed FIFA 21?",
  "answer": "FIFA 21 was developed by EA Vancouver and EA Romania...",
  "confidence": "low",
  "source_type": "web_search (Tavily)",
  "sources": ["https://en.wikipedia.org/wiki/FIFA_21"],
  "web_search_used": true
}
```

---

## Project Limitations

- The local dataset contains only 15 games; most queries about specific recent titles will use web search.
- The JSON files do not include a "Developer" field — only "Publisher". Developer details come from web search.
- Long-term memory is a flat JSON file; it is not vector-searchable. For production, replace with a vector store.
- In-session conversation history (`ShortTermMemory`) does not persist across Python kernel restarts.
- Web search requires a Tavily API key. Without it, `game_web_search` returns a graceful error message.

---

## Submission Checklist

- [x] `starter/Udaplay_01_solution_project.ipynb` — complete, runs top-to-bottom
- [x] `starter/Udaplay_02_solution_project.ipynb` — complete, runs top-to-bottom
- [x] ChromaDB created and persisted in `starter/chromadb/`
- [x] `retrieve_game` tool implemented with metadata and similarity scores
- [x] `evaluate_retrieval` tool implemented (LLM-as-judge, structured JSON output)
- [x] `game_web_search` tool implemented (Tavily, graceful failure without API key)
- [x] Agent workflow: retrieve → evaluate → [web search] → answer
- [x] In-session conversation state maintained via `ShortTermMemory`
- [x] Long-term memory persisted to `agent_memory.json`
- [x] Structured JSON output per query
- [x] Tool usage trace per query
- [x] Citations / sources included
- [x] Required queries: FIFA 21, God of War Ragnarök, Pokémon Red, Rockstar Games
- [x] Web fallback demonstrated
- [x] Local RAG demonstrated (Pokémon Gold/Silver)
- [x] `.env.example` included; `.env` excluded from submission
- [x] `requirements.txt` included
- [x] No API keys exposed in notebooks

## Built With

- [ChromaDB](https://www.trychroma.com/) — vector database
- [OpenAI](https://openai.com/) — embeddings (text-embedding-ada-002) and LLM (GPT-4o-mini)
- [Tavily](https://app.tavily.com/) — web search API
- [LangChain-style custom lib](starter/lib/) — StateMachine, Agent, Tool, Memory abstractions

## License

[License](LICENSE.md)
