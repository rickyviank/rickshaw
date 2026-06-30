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

## Tests

```bash
pytest
```

## License

MIT
