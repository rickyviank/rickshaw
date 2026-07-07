# DRAFT — Make Rickshaw's TUI feel like Claude Code / Codex

> Living document. Captures best-guess approach, open assumptions, and decisions
> as they are resolved during the interview. Not the final PRD.

## Understanding of the request (to confirm)

Ricky wants Rickshaw's terminal UI to look and feel like modern agentic-CLI
harnesses — **Claude Code** and **OpenAI Codex CLI** in particular — *before*
adding more capabilities. Scope for THIS effort (per Q0 answer) is the
**interactive TUI look & feel**: the input box, streaming output rendering,
slash commands, keybindings, and general chrome — NOT the agentic coding
workflow (file editing / shell tools) yet.

## Current state (from code read)

Implemented in `rickshaw/tui.py` (Textual `App`, ~1334 lines):
- Layout: `Static` banner header → `VerticalScroll` transcript → 1-line `hint`
  → borderless `Input` pinned at bottom.
- Near-monochrome dark theme (`#0e0f11` bg), single amber accent (`›` user mark),
  hairline `Rule` between turns. No footer/status bar chrome.
- Streaming: assistant text rendered into a `Markdown` widget, updated per delta.
- Slash commands: `/help /status /settings /models /clear /provider /effort
  /model /login /memory /quit /exit` via `_COMMANDS`. Inline autocomplete via
  Textual `SuggestFromList` (ghost-text suggestion, not a dropdown menu).
- "Thinking" feedback: static hint text `thinking…  ·  esc to interrupt` (no
  spinner/elapsed timer). Input disabled during a turn.
- Keys: `Esc` interrupt / cancel wizard, `Ctrl+L` clear, `Ctrl+C` quit.
- Wizards (provider picker, /settings, /provider add, OAuth /login) are driven
  through the same single-line Input with step state.
- End-of-turn meta line: token count · tool calls; degraded banner when offline.

## Gap analysis vs Claude Code / Codex (look & feel)

Candidate improvements (to be prioritized/decided in interview):
1. **Welcome/splash panel** — bordered rounded box on launch with logo, cwd,
   active model, quick tips (Claude Code / Codex both show this).
2. **Bordered input box** — rounded/framed multi-line input vs current borderless
   single line; placeholder + prompt glyph inside the frame.
3. **Slash-command dropdown** — a real popup menu listing command + description
   as you type `/`, vs the current ghost-text suggester.
4. **Input history** — Up/Down to recall previous submissions.
5. **Multi-line composing** — Shift+Enter (or `\`) for newlines; Enter submits.
6. **Animated thinking indicator** — spinner + elapsed seconds + live token
   count + "esc to interrupt", inline in transcript (Codex/Claude style).
7. **Persistent status/footer bar** — model · provider · effort · context/token
   usage, always visible.
8. **Message styling** — clearer visual separation of user vs assistant (e.g.
   Codex-style boxed/prefixed blocks) instead of only a `›` mark + rule.
9. **Keybinding parity** — Ctrl+C twice to quit, Esc to clear input line,
   Ctrl+J newline, etc.
10. **Theme** — palette/accent tuning to match the reference harness.

## Open assumptions (to resolve in interview)

- A1: Reference/north-star is a *blend* of Claude Code + Codex, prioritizing
  visual polish over 1:1 feature parity. (UNCONFIRMED)
- A2: Stay on Textual (no rewrite to Rich-only or another framework). (UNCONFIRMED)
- A3: Backward compatibility: all existing slash commands, flags, and wizard
  flows must keep working. (UNCONFIRMED)
- A4: No new runtime dependencies beyond Textual/Rich already present.
  (UNCONFIRMED)
- A5: Terminal support target (truecolor vs 256-color, min width). (UNCONFIRMED)
- A6: Success criteria are subjective ("feels like Claude Code") vs a concrete
  checklist of elements. (UNCONFIRMED)

## Unexplored user journeys (to cover in interview)

- J1: First launch (no provider configured) — splash + picker experience.
- J2: Composing a multi-line prompt / editing before submit.
- J3: Discovering & running a slash command via the menu.
- J4: Watching a long streaming turn (thinking → streaming → done) and
  interrupting mid-flight.
- J5: Recalling and re-running a previous message.
- J6: Narrow terminal / resize behavior.
- J7: Degraded/offline turn presentation.

## ⚠️ MAJOR PIVOT (Q18) — framework & architecture

Ricky's answer to the constraints question changes the foundation of this work:

- **Framework:** switch **from Textual (Python) to Ink** (React/TypeScript CLI
  framework — the same lib Claude Code / Codex CLI are built on).
- **Architecture:** split into a **TypeScript/Ink frontend + Python backend**.
- **Backward compatibility:** **dropped** — project is pre-users / dev-mode, so
  we are free to break the current Python `tui.py` and its flows.
- **Dependencies:** ideally no new runtime deps, **but Ink (+ Node toolchain) is
  now allowed**.
- **Terminal support:** truecolor target; graceful 256-color degrade; min width
  ~80 cols (welcome panel / status bar adapt, don't break).

**Implications (surfaced to user):**
- The entire existing `rickshaw/tui.py` (Textual) is effectively **replaced**,
  not enhanced. All UI-layer logic living in it today — slash commands, provider
  picker, `/settings`/`/provider add`/OAuth wizards, streaming render, hints —
  must be rebuilt in Ink and/or exposed by the backend over a protocol.
- A **frontend↔backend transport** is now required (Node process ↔ Python
  process). This is the central new architectural decision (see D18b).
- Python-side niceties move or get bridged: `tiktoken` token estimate (D6) is
  Python — either estimate on the Node side (`js-tiktoken`, a new npm dep) or
  stream counts from Python. `rickshaw_ai` pricing/context metadata (D10/D11)
  lives in Python and must be surfaced to the frontend via the protocol.
- **Packaging/distribution** changes: `rickshaw` is a pip console script today;
  it now needs Node + Python present (or a bundled Node runtime). Launch story
  TBD (see D18c).
- Orchestrator/providers/memory/`rickshaw_ai` (backend) are **unchanged in
  purpose** and stay in Python.

Many earlier UI decisions (D2–D16) still hold as *product* intent but their
*implementation* now targets Ink/React components instead of Textual widgets.

## ⚠️⚠️ SECOND PIVOT (Q19 final) — FULL TYPESCRIPT REWRITE

Ricky chose **C) Full TypeScript rewrite, retiring the Python core** (informed
choice made after a written pros/cons + a strong recommendation to keep Python).

- **Everything becomes TypeScript/Node.** No Python backend, no stdio/HTTP
  bridge. Ink for the TUI.
- The entire current Python codebase (`rickshaw/`, `rickshaw_ai/`, memory,
  orchestrator, providers, tool registry, prompt builder, worker, tests) is
  **retired / reimplemented in TS**.
- This is far larger than a TUI facelift — it's a ground-up reimplementation.
  Scope of THIS PRD must be pinned (Q20): full port vs frontend-first with core
  port scoped separately.
- Components that must be re-created in TS (each a design area): provider layer
  (OpenAI/Anthropic/Devin + OAuth PKCE/device, streaming, token/cost), semantic
  memory (store, embedder, ranker/MMR, dedupe, compaction), orchestrator turn
  loop, tool registry, prompt budgeting, settings/config, CLI entry.
- Ecosystem substitutions to decide: SQLite (better-sqlite3 / node:sqlite),
  vector search (chromadb JS client / alt), embeddings (js-tiktoken for tokens;
  TF-IDF reimplemented or a JS embedding lib), config format, test runner
  (vitest/jest), package manager, build/bundler, distribution (npm bin).

## ✅ RESOLUTION (Q19 settled) — STAY WITH PYTHON + TEXTUAL

After weighing Ink and a full TS rewrite, Ricky settled on **staying with
Python + Textual and enhancing the existing TUI**. Both pivots above are
**reverted** and kept only as a record of the discussion.

- **Foundation:** Python + Textual; enhance `rickshaw/tui.py` in place.
- The mature Python core (`rickshaw_ai`, memory, orchestrator, tool registry)
  is **kept and reused** — no rewrite, no cross-language bridge.
- All UI product decisions **D2–D16 stand** and are implemented as Textual
  widgets/CSS (welcome panel, bordered multi-line `TextArea` input, slash-command
  dropdown via overlay/`OptionList`, animated spinner+elapsed+live token count,
  persistent status bar, role-labeled messages, input history, keybindings, full
  custom theme). Textual is confirmed capable of all of these.
- **D18 constraints (revised):** framework = Textual (NOT Ink); **backward
  compatibility NOT required** (pre-users, dev-mode — free to change commands /
  settings schema / flows); prefer no new runtime deps (Textual/Rich/tiktoken
  already present); truecolor target with graceful 256-color degrade; min width
  ~80 cols (welcome panel & status bar adapt, don't break).
- Scope question Q20 (phasing) is **moot** — no rewrite; this remains a focused
  TUI enhancement of the existing app.

## Decisions log

### D1 — Reference target & north-star
- **Decision:** Blend Claude Code + Codex, prioritizing overall **visual polish**
  over 1:1 fidelity to either.
- **Alternatives:** (a) match Claude Code closely, (b) match Codex closely.
- **Rationale:** Ricky wants "the likes of" these harnesses, not a clone; a blend
  lets us cherry-pick the best-looking element from each (Claude's boxed input +
  welcome panel, Codex's lean status line/blocks) and tune a cohesive theme.
- **Implication:** Success is judged on cohesive polish, not a per-feature diff
  against one tool (resolves A1; A6 leans "subjective polish + element checklist").

### D2 — Welcome / splash panel
- **Decision:** Rounded **bordered welcome panel** rendered once on launch:
  logo/slogan, active provider · model · effort, cwd, and 2–3 quick tips
  (`/help`, `esc to interrupt`, `^c quit`).
- **Alternatives:** (b) lean polished header lines, (c) keep current banner.
- **Rationale:** Biggest single "feels like Claude Code" signal; sets tone.
  Rendered only at launch (and after `/clear`?) so it doesn't clutter transcript.
- **Open follow-ups:** exact contents when no provider is selected (J1); whether
  panel reappears after `/clear`.

### D3 — Input box
- **Decision:** Rounded **bordered multi-line input** with a `›` prompt glyph
  inside; `Enter` submits, `Shift+Enter` (and/or `Ctrl+J`) inserts a newline;
  grows a few lines then scrolls internally.
- **Alternatives:** (b) bordered single-line, (c) borderless + multiline only.
- **Rationale:** Matches both references; enables pasting/editing longer prompts.
- **KEY IMPLEMENTATION COST (flag in Constraints):** Textual `Input` is
  single-line. Multi-line requires switching to `TextArea`, which changes how the
  step-based wizards (provider picker, `/settings`, `/provider add`, OAuth
  `/login`) currently consume `Input.Submitted`. Wizard input handling must be
  re-plumbed onto the new widget while preserving behavior (A3 backward-compat).
- **Open follow-ups:** newline binding (`Shift+Enter` vs `Ctrl+J` vs both);
  whether wizards stay single-line-style within the multiline widget.

### D4 — Slash-command menu
- **Decision:** **Dropdown popup** anchored above the input listing matching
  commands with descriptions (`/help — Show this help.`), live-filtered as you
  type, `↑/↓` to select, `Enter`/`Tab` to accept, `Esc` to dismiss.
- **Alternatives:** (b) keep ghost-text suggester + descriptions, (c) no change.
- **Rationale:** Most visible slash-command upgrade; matches both references.
  Descriptions already exist in `_COMMANDS`, so content is free.
- **Open follow-ups:** does the menu only trigger on leading `/`; behavior for
  commands with args (e.g. `/effort <level>`, `/model <name>`).

### D5 — Thinking / streaming indicator
- **Decision:** **Animated inline status** = spinner + elapsed seconds + **live
  token count** + "esc to interrupt", shown from submit until it collapses into
  the streaming reply, e.g. `⠋ Thinking… (3s · 412 tok · esc to interrupt)`.
- **Alternatives:** (b) spinner + elapsed only, (c) static hint.
- **Rationale:** Ricky wants the full "alive" Codex/Claude feel including token
  count.
- **KEY IMPLEMENTATION COST (flag in Constraints):** the streaming path
  (`Orchestrator.run_turn(on_delta=...)` → `provider.stream()`) currently yields
  **text chunks only, not usage**. A live token count requires either (a)
  local estimation of streamed tokens (e.g. `tiktoken`, already a dep) as a
  proxy, or (b) plumbing incremental usage through the provider/orchestrator.
  Need to decide which (follow-up question).
- **Open follow-ups:** spinner frames/refresh rate.

### D6 — Live token count source
- **Decision:** **Local `tiktoken` estimate** of the accumulated streamed text
  (client-side), labeled as an estimate; the **authoritative** total from real
  `usage` still shown on the existing end-of-turn meta line.
- **Alternatives:** (b) plumb real incremental usage through provider/orchestrator.
- **Rationale:** Gives the live ticking-counter feel with minimal blast radius;
  `tiktoken` is already a dependency and works for every provider. Real usage
  often only arrives at end-of-stream, so it wouldn't tick live anyway.

### D7 — Persistent status bar
- **Decision:** **Persistent, bottom-fixed, single-row** status bar showing:
  `provider | model | effort | context % | token usage | rough price estimate`.
  Users can **customize** which segments appear / their order.
- **Rationale (Ricky, verbatim intent):** always-on orientation; exactly 1 row of
  chrome to preserve the minimal look; user-editable.
- **NEW COMPLEXITY — price estimation:** requires a per-model **pricing table**
  ($ per 1M input/output tokens) applied to token usage.
  - **D8 decision:** **Bundled defaults + user overrides** — ship a pricing
    table for common OpenAI/Anthropic models; users can override/add rates in
    `~/.rickshaw/settings.json`. Show `—` when a model has no rate. Price is a
    rough estimate based on the D6 tiktoken count (labeled approximate).
  - **Rationale:** works out-of-the-box yet stays accurate for custom providers
    / newer models. Bundled rates may go stale (accepted; document as such).
- **D9 — customization mechanism:** **Config-driven segment list** in
  `~/.rickshaw/settings.json`, e.g.
  `"status_bar": ["provider","model","effort","context","tokens","price"]`.
  Users reorder/remove segments from a **fixed, documented vocabulary** of known
  segments. Default = all six in the order above. Unknown segment names are
  ignored with a warning.
  - **Alternatives:** (b) config + in-TUI `/statusbar` editor, (c) free-form
    template string with placeholders.
  - **Rationale:** delivers customization with least risk; matches how other
    settings persist. An in-TUI editor (B) is a clean future follow-up (note in
    Out of Scope).
- **D10 — "context %" basis:** compute against the active model's context window
  read from **`rickshaw_ai` model metadata**. Confirmed present:
  `rickshaw_ai.registry.ModelInfo.context_window: int` and builtins populate it
  (e.g. Anthropic models `ctx=200_000` in `rickshaw_ai/_builtins.py`). If a
  model's `context_window` is missing/0, **raise a warning** and render `—` for
  the context segment (per Ricky).
  - **Rationale:** authoritative per-model data already ships in `rickshaw_ai`;
    no new bundled window table needed.

- **CODEBASE FINDING that revises D8 (pricing):** `rickshaw_ai` **also already
  ships per-model pricing** — `ModelInfo.pricing = Pricing(input=$/1M,
  output=$/1M, cache_read=...)`, populated in `_builtins.py` (e.g.
  claude-sonnet-4 `pin=3, pout=15`). So the "bundled defaults" for D8 should be
  **`rickshaw_ai`'s existing pricing metadata**, not a brand-new `pricing.yaml`.
  User overrides (D8) layer on top via settings.
  - **D11 CONFIRMED:** use `rickshaw_ai` built-in pricing as default + user
    overrides in `~/.rickshaw/settings.json`. No separate bundled pricing file.
    (Supersedes the D8 "new bundled table" wording.)

- **D12 — segment basis (hybrid):** `token usage` and `price` are **cumulative
  session** running totals (spend/usage tracking); `context %` reflects the
  **current-turn context fill** (latest turn's context tokens ÷ window). Warn +
  `—` when data unavailable (D10).
  - **Rationale:** price/usage meters answer "how much this session"; a % most
    naturally means "how full is the window right now".

### D13 — Input history recall
- **Decision:** **Persistent** input history — ↑/↓ cycles previously submitted
  messages, saved to `~/.rickshaw/history` and restored across launches.
- **Alternatives:** (a) in-session only, (c) none.
- **Rationale:** standard in these harnesses; cheap.
- **Interaction note (multi-line input, D3):** ↑/↓ triggers history only when the
  cursor is on the first line (↑) / last line (↓) and no menu is open; otherwise
  arrows move between lines / within the slash menu. Must be spec'd explicitly.
- **Open follow-ups:** history size cap; whether slash commands are stored too.

### D14 — Turn / message styling
- **Decision:** **Role-labeled blocks** — a dim `you` / `rickshaw` label (or
  glyph) above each message and a subtle left gutter/indent for the assistant;
  keep hairline rules between turns.
- **Alternatives:** (a) keep minimal `›` + rules only, (c) tinted per-message
  boxes.
- **Rationale:** improves scannability without the heaviness of boxes, which
  fight the minimalist theme and wrap awkwardly in narrow terminals.
- **Open follow-ups:** exact labels/glyphs; whether the amber `›` marker stays.

### D15 — Keybindings
- **Decision (adopted as proposed):**
  - `Enter` submit; `Shift+Enter`/`Ctrl+J` newline (D3).
  - `Esc` — interrupt in-flight turn / dismiss slash menu / cancel wizard; if
    input has text and nothing is running, clear the input line.
  - `↑/↓` — slash-menu nav when menu open, else input history (D13).
  - `Ctrl+C` — **double-tap to quit** (first press: "press again to quit");
    a single `Ctrl+C` cancels a running turn if one is active.
  - `Ctrl+L` — clear transcript (kept).
- **Rationale:** blends both references; double-tap `Ctrl+C` avoids accidental
  quits while `Esc` stays the primary interrupt.
- **Note:** `Ctrl+C` double-tap replaces Textual's default immediate quit; needs
  explicit handling + a transient "press again" hint.

### D16 — Theme / palette
- **Decision (D16+D17):** **Full re-theme** to a **new Rickshaw identity inspired
  by both** Claude Code and Codex (its own palette, not a clone of either).
  Ship **one built-in theme** now; a theme-selection setting is a future add
  (Out of Scope).
- **Alternatives:** (a) keep monochrome+amber refine-only, (b) restrained accent
  on current identity; palette: Claude-warm / Codex-cool; config: 2–3 themes.
- **Rationale:** Ricky wants a distinct, polished Rickshaw look drawing on both
  references rather than mimicking one.
- **Implication:** replaces the near-monochrome+amber identity; exact palette
  defined below. Branding glyphs (`o--o`, `›`) kept, restyled to new accent.

### D21 — Concrete theme palette (adopted)
Single built-in dark theme (new Rickshaw identity, warm accent + cool neutrals):
- Background `#0f1113` · Surface/panels `#16181b` · Borders/rules `#2a2f36`
- Primary text `#e6e8ea` · Secondary/meta `#8b929c`
- Accent (user marker, focus, spinner, links) `#e0a86b`
- Assistant role label `#7fb0c9` · Warning `#d98a3d` · Error/degraded `#d16a5a`
  · Success `#7fae7f`
- Applied via Textual CSS across welcome panel, status bar, input frame, role
  labels, rules, spinner. 256-color terminals get nearest-color degrade.

### D22 — Interaction defaults (confirmed)
1. **Welcome panel after `/clear`:** re-render the panel (fresh-start feel).
2. **Newline key:** support **both** `Shift+Enter` and `Ctrl+J`; `Enter` submits.
3. **Slash menu with args:** dropdown lists/filters command *names*; once a
   command needing an arg is entered (e.g. `/effort `), an **interactive picker**
   of valid values is shown (e.g. `low | medium | high`; `/model` → available
   models) rather than only a hint.
4. **Narrow terminal (<~80 cols):** status bar drops lowest-priority segments
   first (price → tokens → context) to fit; welcome panel collapses to a compact
   1–2 line form. Never wrap-breaks.
5. **History contents:** input history stores plain messages **and** slash
   commands.

### D23 — Visual detail decisions (resolved former open items)
1. **Spinner:** line spinner `|/-\` @ ~8 fps.
2. **Role glyphs:** amber `›` before user message; `o--o` + dim `rickshaw` for
   assistant.
3. **Border style:** Textual `round` for welcome panel + input frame.
4. **History cap:** rolling 1,000 entries in `~/.rickshaw/history`.
