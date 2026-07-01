# Future Roadmap

Concise roadmap for parts of Rickshaw that are intentionally simple today and
are expected to evolve. This is a roadmap, not a design doc.

## Embeddings

Today: `TFIDFEmbedder` — offline, fit-on-the-fly TF-IDF with feature hashing and
L2 normalization. Captures **lexical** (word-overlap) similarity only.

Next:
- Swap in a learned sentence encoder (e.g. `all-MiniLM-L6-v2` via
  `sentence-transformers`) behind the existing `Embedder` ABC for true semantic
  similarity, no schema change (dimension is configurable).
- Optionally support provider-backed embeddings for higher quality.
- Persist the IDF/vocabulary table so vectors are stable across restarts.

## Vector Search

Today: SQLite is the source of truth; `MemoryStore` mirrors embeddings into a
ChromaDB index for indexed KNN search (scope filtering via Chroma metadata),
falling back to a brute-force cosine scan when ChromaDB is unavailable.

Next:
- Keep SQLite and the Chroma index consistent under concurrent access / crashes
  (re-index / backfill on startup if they diverge).
- Store embeddings as binary blobs even on the fallback path (avoid JSON).
- Evaluate ANN index tuning (HNSW params) for large stores.

## Deferred Worker

Today: `DeferredWorker` shares the **same** injected `LLMProvider` as the
orchestrator and runs synchronously via explicit `process_batch()`.

Next:
- **Background processing**: run the worker in its own thread/process with an
  event loop so it never blocks the foreground turn loop.
- **Per-request provider selection**: allow a separate, cheaper provider for
  background tasks (importance scoring, compaction), distinct from the
  foreground chat provider — e.g. `RICKSHAW_BACKGROUND_PROVIDER=openai`,
  `RICKSHAW_BACKGROUND_MODEL=gpt-4o-mini`.
- **Priority queue**: foreground provider calls take precedence; background work
  pauses when the user sends a new message.

(See the `FUTURE:` comment in `rickshaw/worker.py`.)

## Deduplication

Today: dedupe-on-write via a fixed embedding-similarity threshold (0.92),
comparing against the top-5 most similar existing records across all scopes;
duplicates are silently discarded.

Next:
- **Scope-aware dedup**: a session-scoped fact and a global-scoped fact with
  similar text are not duplicates.
- **Update-on-duplicate**: instead of discarding, bump the existing record's
  `last_used_at` / `use_count` to reflect the re-mention.
- **Adaptive threshold**: calibrate the threshold per embedding model, since
  similarity distributions differ across models.
- **Truly semantic dedup**: emerges naturally once the embedder is upgraded from
  TF-IDF to a learned model (see Embeddings).

(See the `FUTURE:` comment in `rickshaw/memory/service.py`.)
