# Rickshaw

A multi-LLM provider harness with a normalized interface, user-selectable reasoning effort levels, and an optional embedding-backed ontology layer.

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

## Ontology layer (optional)

Rickshaw includes a symbolic ontology graph for user-defined concept schemas:

```python
from rickshaw.ontology import Entity, OntologyGraph, Relationship

graph = OntologyGraph("my_ontology.json")
graph.add_entity(Entity(id="py", entity_type="language", fields={"label": "Python"}))
graph.add_entity(Entity(id="js", entity_type="language", fields={"label": "JavaScript"}))
graph.add_relationship(Relationship(source_id="py", target_id="js", relation_type="similar_to"))
graph.save()
```

### Embedding-backed concept matching

When an embedding-capable provider is configured (e.g. OpenAI), the `ConceptMatcher` can:

- **Classify/auto-tag** new text against existing entities.
- **Detect synonym/duplicate** concepts.
- **Suggest fuzzy links** between text and entities.

All results are **suggestions only** — they never silently decide.

```python
from rickshaw.ontology.concept_matcher import ConceptMatcher

matcher = ConceptMatcher(provider=embedding_provider, graph=graph)
if matcher.available:
    matches = matcher.classify("Python programming")
    for m in matches:
        print(f"  {m.entity.id}: {m.score:.3f}")
```

If no embedding-capable provider is configured, `matcher.available` returns `False` and calling embedding methods raises a clear error.

## Tests

```bash
pytest
```

## License

MIT
