# Rickshaw

A multi-LLM provider harness with a normalized interface and user-selectable reasoning effort levels.

## Setup

```bash
# Clone and install
git clone https://github.com/rickyviank/Rickshaw.git
cd Rickshaw
pip install -e ".[dev]"

# Configure credentials
cp .env.example .env
# Edit .env with your API keys
```

### Required environment variables

| Variable | Required | Description |
|---|---|---|
| `RICKSHAW_PROVIDER` | No | Default provider (`openai` or `devin`). Defaults to `openai`. |
| `RICKSHAW_EFFORT` | No | Default effort level: `low`, `medium`, `high`. Defaults to `medium`. |
| `OPENAI_API_KEY` | For OpenAI | OpenAI API key. |
| `OPENAI_BASE_URL` | No | Override the OpenAI API base URL. |
| `OPENAI_MODEL` | No | Chat model to use (default: `gpt-4o`). |
| `OPENAI_EMBEDDING_MODEL` | No | Embedding model (default: `text-embedding-3-small`). |
| `DEVIN_API_KEY` | For Devin | Devin API key. |
| `DEVIN_BASE_URL` | No | Override the Devin API base URL. |
| `RICKSHAW_EMBEDDING_PROVIDER` | No | Separate embedding provider (e.g. `openai`) independent of the chat provider. |

You may also supply values in a `config.yaml` file in the working directory.

## Supported providers

- **openai** — OpenAI chat completions and embeddings APIs.
- **devin** — Devin coding agent API (skeleton; fill in TODOs from Devin API docs).

Adding a new provider: subclass `rickshaw.providers.base.LLMProvider`, implement the abstract methods, and register it:

```python
from rickshaw.providers.factory import register
register("my_llm", MyLLMProvider)
```

## Usage

```bash
# Interactive REPL
rickshaw --provider openai --effort high

# Or via python -m
python -m rickshaw --provider openai

# Validate connectivity only
rickshaw --provider openai --validate-only
```

### Effort levels

Rickshaw normalizes reasoning effort into three levels: **low**, **medium**, **high**.

- Set the session default with `--effort`:
  ```bash
  rickshaw --effort high
  ```
- Override per-turn inside the REPL:
  ```
  you> /effort low
    Effort set to low for subsequent turns.
  you> Summarize this document
  ```
- Each turn displays the effort used:
  ```
  [effort: high]  (gpt-4o)
  Here is the response...
  ```
- If the active provider does not honor the chosen effort level, a warning is shown.

### Provider capabilities

Each provider reports its capabilities via `provider.capabilities()`:

```python
from rickshaw.providers import get_provider

p = get_provider("openai", api_key="sk-...")
caps = p.capabilities()
print(caps.streaming)    # True
print(caps.embeddings)   # True
print(caps.effort_levels)  # [Effort.LOW, Effort.MEDIUM, Effort.HIGH]
```

## Normalized Tool Calling

The provider interface supports normalized tool calls via `ToolSpec` and `ToolCall`:

```python
from rickshaw.providers import ToolSpec, ToolCall, get_provider

provider = get_provider("openai", api_key="sk-...")
tools = [
    ToolSpec(
        name="remember",
        description="Store a fact in memory.",
        parameters={
            "type": "object",
            "properties": {"fact": {"type": "string"}},
            "required": ["fact"],
        },
        category="memory",   # classification hint ("memory" | "general")
        side_effect=True,     # read-only tools set this False
    )
]

# tools *advertises* which tools are available; tool_choice controls whether the
# model is encouraged/required/forbidden to use them ("auto" | "none" | "required").
response = provider.complete(messages, tools=tools, tool_choice="auto")
for tc in response.tool_calls:
    print(tc.name, tc.arguments)  # e.g. "remember" {"fact": "..."}
```

- `ToolCall` is a pure normalized dataclass; provider-specific parsing lives on
  each provider (`OpenAIProvider._parse_tool_calls`), not on the base type.
- `tool_choice` defaults to `None` (provider decides). `OpenAIProvider` forwards
  it to the API; `DevinProvider` accepts but ignores it.
- `Response.tool_calls` defaults to `[]` — existing code is unaffected.
- Providers without function-calling (e.g. Devin) accept the `tools` parameter but ignore it.

### Tool registry (generalized dispatch)

Tool dispatch is decoupled from any specific backend via `ToolRegistry`, which
validates arguments against each tool's JSON schema and supports sync **and**
async handlers:

```python
from rickshaw.memory import MemoryService
from rickshaw.memory.tools import build_memory_registry
from rickshaw.providers.base import ToolCall

registry = build_memory_registry(MemoryService())
# register additional (even async) tools: registry.register(name, handler, spec)

result = registry.dispatch(ToolCall(id="1", name="recall", arguments={"query": "prefs"}))
# or: await registry.async_dispatch(tool_call)
```

The `Orchestrator` accepts a `ToolRegistry` via DI (defaulting to the memory
registry) and returns a structured `TurnResult(text, warnings, tool_calls_made,
degraded)` so callers can detect degradation without parsing the text.

## Semantic Memory Layer

A fully offline semantic memory layer enables persistent, ranked context retrieval:

```python
from rickshaw.memory import MemoryService
from rickshaw.memory.embedder import TFIDFEmbedder

memory = MemoryService(embedder=TFIDFEmbedder())

# Store a fact
record = memory.write("User prefers dark mode")

# Retrieve relevant context (sensitive records are excluded here, before ranking)
context = memory.assemble_context("What are the user's preferences?")
```

The default `TFIDFEmbedder` is an offline, semantically-meaningful embedder
(fit-on-the-fly TF-IDF + feature hashing, L2-normalized) — a stepping stone
toward learned embeddings (see [FUTURE.md](FUTURE.md)).

### Architecture

| Component | Module | Description |
|---|---|---|
| **MemoryRecord** | `rickshaw/memory/record.py` | Core data unit with scope, type, importance, embedding |
| **Embedder** | `rickshaw/memory/embedder.py` | `TFIDFEmbedder` (offline, semantic) or `ProviderEmbedder` (API-backed) |
| **Store** | `rickshaw/memory/store.py` | SQLite persistence; scope-filtered search via `sqlite-vector` (KNN) with brute-force cosine fallback |
| **Ranker** | `rickshaw/memory/ranker.py` | Weighted-sum scoring (relevance + recency + importance) with MMR diversity |
| **MemoryService** | `rickshaw/memory/service.py` | Facade: dedupe-on-write, sensitive filtering, ranked retrieval, `remember`/`recall`/`forget` |
| **Memory Tools** | `rickshaw/memory/tools.py` | Tool specs + `build_memory_registry` wiring memory ops into a `ToolRegistry` |
| **ToolRegistry** | `rickshaw/tool_registry.py` | Backend-agnostic tool dispatch with schema validation + sync/async handlers |
| **PromptBuilder** | `rickshaw/prompt/builder.py` | Token-budgeted prompt assembly (sensitive records already excluded upstream) |
| **Orchestrator** | `rickshaw/orchestrator.py` | Turn loop with retry/backoff; returns a `TurnResult` |
| **Worker** | `rickshaw/worker.py` | Deferred importance scoring, compaction/reflection, TTL eviction |
| **JobQueue** | `rickshaw/queue.py` | In-memory FIFO queue for deferred work items |

### Offline demo

```bash
python examples/offline_demo.py
```

Runs a full turn cycle using `TFIDFEmbedder` and a fake provider — no API keys needed.

### Optional: indexed vector search

`MemoryStore` uses the [`sqlite-vector`](https://github.com/sqliteai/sqlite-vector)
extension for indexed KNN search when available, and transparently falls back to
a brute-force cosine scan otherwise (a warning is logged). To enable it:

```bash
pip install -e ".[vector]"   # requires a Python sqlite3 built with extension loading
```

## Tests

```bash
pytest
```

## License

MIT
