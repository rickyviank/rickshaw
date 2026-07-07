# PRD — Rickshaw TUI: a Claude-Code / Codex-grade terminal experience

Status: **for sign-off** · Owner: Ricky Vian · Author: Devin (spec only, no code)
Foundation: **Python + Textual** (enhance existing `rickshaw/tui.py`)

---

## 1. Problem Statement

Rickshaw already has a functional Textual TUI, but it is minimal and does not yet
"feel" like the modern agentic-CLI harnesses it aspires to (Claude Code, OpenAI
Codex CLI). The goal of this effort is to **polish the interactive TUI look &
feel** — launch experience, input box, slash-command discovery, live turn
feedback, orientation/status, message styling, keybindings, and theme — so
Rickshaw reads as a first-class harness **before** more capabilities are built on
top of it.

Scope is explicitly the **look & feel of the terminal UI**, not the agentic
coding workflow (file editing / shell tools). The Python core (`rickshaw_ai`,
semantic memory, orchestrator, tool registry) is unchanged in purpose and reused.

Why it matters: the TUI is the entire product surface today; its polish sets the
perceived quality bar and the ergonomic foundation for every future feature.

---

## 2. Proposed Solution

Enhance the existing Textual application (`rickshaw/tui.py`) — **not** a rewrite
and **not** a framework change. Textual is confirmed capable of every element
below. The upgrade is delivered as a cohesive visual + interaction pass:

**Architecture choices**
- **Framework:** remain on **Textual** (Python). No Ink, no TypeScript, no
  frontend/backend split. (See Decisions D18/D19 for the evaluation and why this
  was chosen over an Ink rewrite.)
- **Core reuse:** all turns continue to route through `Orchestrator.run_turn`;
  provider/model/effort switching, OAuth, memory, and degraded-mode handling are
  reused as-is. New UI reads model metadata (context window, pricing) from
  `rickshaw_ai` (`ModelInfo.context_window`, `ModelInfo.pricing`).
- **No new runtime dependencies** where avoidable — reuse Textual, Rich, and
  `tiktoken` (all already present). Any new dependency must follow the project's
  supply-chain rules; none is currently anticipated.
- **Backward compatibility is NOT required** (project is pre-release / dev-mode):
  commands, `settings.json` schema, and flows may change where it improves the
  experience. New settings keys are additive with sensible defaults.
- **Terminal support:** truecolor target; graceful nearest-color degrade on
  256-color terminals; assume min width ~80 cols — the welcome panel and status
  bar **adapt** below that rather than wrap-break.

**Key technical decisions (summary; full rationale in §5)**
- Multi-line input requires migrating from Textual `Input` to `TextArea`; all
  step-based wizards (provider picker, `/settings`, `/provider add`, OAuth
  `/login`) are re-plumbed onto the new widget.
- Live token count during streaming is a **local `tiktoken` estimate** (the
  streaming path carries text, not usage); the authoritative total from real
  `usage` still appears on the end-of-turn meta line.
- Status-bar `context %` uses `rickshaw_ai` per-model `context_window`; `price`
  uses `rickshaw_ai` per-model `pricing` as defaults, overridable in settings.

**New / changed UI elements**
1. Bordered **welcome panel** on launch (and after `/clear`).
2. Rounded **bordered multi-line input** with in-frame `›` glyph.
3. **Slash-command dropdown** menu (names + descriptions; arg pickers).
4. **Animated thinking/streaming indicator** (spinner + elapsed + live tokens).
5. **Persistent, customizable, single-row status bar**.
6. **Role-labeled** message blocks.
7. **Persistent input history** (↑/↓).
8. Updated **keybindings** (incl. `Ctrl+C` double-tap to quit).
9. A single **new built-in theme** (concrete palette in §5, D21).

---

## 3. User Journeys

### J1 — First launch (no provider configured)
1. User runs `rickshaw` with no provider in flags/env/`settings.json`.
2. The **welcome panel** renders: logo/slogan, `provider: (none)`, cwd, and quick
   tips (`/help`, `esc interrupt`, `^c quit`). In a narrow terminal it collapses
   to a compact 1–2 line form.
3. Because no provider is selected, the interactive **provider picker** starts
   immediately (existing behavior, restyled). OAuth-capable providers show
   `(oauth)`; selecting one triggers the existing OAuth login flow.
4. On success, the status bar populates (`provider · model · effort · …`) and the
   input is focused. → connects to J3/J4.

### J2 — Composing a multi-line prompt
1. User types in the bordered input. `Enter` submits; `Shift+Enter` or `Ctrl+J`
   inserts a newline. The box grows a few lines then scrolls internally.
2. Pasting multi-line text keeps newlines. `Esc` (nothing running, text present)
   clears the input line.
3. `↑` on the first line recalls previous history; `↓` on the last line moves
   forward through history (see J5). Within the text, `↑/↓` move the cursor
   between lines.

### J3 — Discovering & running a slash command
1. User types `/`. A **dropdown** appears above the input listing matching
   commands with descriptions, filtered live as they type.
2. `↑/↓` selects; `Enter`/`Tab` accepts; `Esc` dismisses.
3. For a command needing an argument (e.g. `/effort `, `/model `), an
   **interactive value picker** is shown (e.g. `low | medium | high`; `/model` →
   the provider's available models) rather than only a hint.
4. Selecting a value applies the command (reusing existing `_cmd_*` handlers) and
   the status bar updates immediately (e.g. new effort/model).

### J4 — A streaming turn (thinking → streaming → done), with interrupt
1. User submits a message. A hairline rule separates it from the prior turn; the
   message is prefixed with the amber `›` marker (D23.2).
2. An **animated indicator** appears: spinner + elapsed seconds + live token
   estimate + "esc to interrupt", e.g. `/ Thinking… (3s · 412 tok · esc to
   interrupt)` (line spinner `|/-\` @ ~8 fps, D23.1).
3. When the first token arrives, the indicator collapses and the reply streams
   into a Markdown block under an `o--o` + dim `rickshaw` label with an assistant
   left gutter/indent (D23.2).
4. The live token estimate ticks as text streams (local `tiktoken`).
5. On completion, the end-of-turn meta line shows the **authoritative** totals
   (real `usage` tokens · tool calls); the status bar's cumulative token/price
   totals update; `context %` reflects this turn's context fill.
6. **Interrupt:** at any point `Esc` (or a single `Ctrl+C` while a turn runs)
   cancels the turn; `(interrupted)` is shown and input re-enables.

### J5 — Recall and re-run a previous message
1. With the input empty (or cursor on first/last line), `↑/↓` cycles previously
   submitted entries (plain messages **and** slash commands), loaded from
   `~/.rickshaw/history` (persisted across launches).
2. User edits the recalled text and submits → normal turn (J4).

### J6 — Narrow terminal / resize
1. Below ~80 cols the **status bar** drops lowest-priority segments first in the
   order price → tokens → context, keeping provider · model · effort visible.
2. The **welcome panel** collapses to a compact 1–2 line form.
3. Nothing wrap-breaks or corrupts the layout; on resize back to wide, dropped
   segments reappear.

### J7 — Degraded / offline turn
1. If the provider is unreachable, the orchestrator returns a degraded result
   (existing behavior).
2. The transcript shows the existing **degraded banner** ("provider unreachable —
   showing local memory only"), themed with the new error/degraded color.
3. Status-bar segments that can't be computed (e.g. price/context when model
   metadata is missing) render `—`, and a warning is surfaced.

### J8 — Customizing the status bar
1. User edits `~/.rickshaw/settings.json` `status_bar` array (e.g. removes
   `price`, reorders segments) from the fixed vocabulary
   `{provider, model, effort, context, tokens, price}`.
2. On next launch the status bar reflects the chosen segments/order. Unknown
   segment names are ignored with a warning.

---

## 4. Constraints

- **C1 — Framework:** Textual only; no framework change, no rewrite.
- **C2 — Core reuse:** the Python core and `rickshaw_ai` are reused; UI reads
  `ModelInfo.context_window` and `ModelInfo.pricing` for status-bar segments.
- **C3 — Dependencies:** prefer no new runtime deps (Textual/Rich/`tiktoken`
  present). Any new dep follows supply-chain rules (prefer versions published
  ≥7 days, no floating ranges); none currently expected.
- **C4 — Backward compatibility:** NOT required (pre-users/dev-mode); commands,
  settings schema, and flows may change. New settings keys are additive.
- **C5 — Terminal support:** truecolor target; graceful 256-color degrade; min
  width ~80 cols with adaptive (not wrap-breaking) welcome panel & status bar.
- **C6 — Streaming reality:** the streaming path (`Orchestrator.run_turn(
  on_delta=...)` → `provider.stream()`) yields **text chunks only, not usage**;
  live token count must be estimated client-side, not assumed from the stream.
- **C7 — Wizard re-plumbing:** moving to a multi-line `TextArea` changes how the
  step-based wizards read input; they must be preserved behaviorally on the new
  widget.
- **C8 — Data availability:** when per-model metadata (context window / pricing)
  is missing, warn and render `—` rather than fail.

---

## 5. Decisions Log

> Each decision lists the choice, alternatives considered, and rationale. The
> living record (with the abandoned Ink/rewrite detour) is in `DRAFT.md`.

**D1 — Reference & north-star.** Blend Claude Code + Codex, prioritizing overall
visual polish over 1:1 fidelity. *Alternatives:* match one tool closely.
*Rationale:* Ricky wants "the likes of" these tools; a blend cherry-picks the
best of each into a cohesive look.

**D2 — Welcome panel.** Rounded bordered panel on launch (and re-rendered after
`/clear`, per D22.1): logo/slogan, provider · model · effort, cwd, quick tips.
*Alternatives:* lean header lines; keep the single banner. *Rationale:* biggest
single "feels like Claude Code" signal; rendered only on launch/clear so it
doesn't clutter the transcript.

**D3 — Input box.** Rounded bordered **multi-line** input with in-frame `›`
glyph; `Enter` submits, `Shift+Enter`/`Ctrl+J` newline (D22.2); grows then
scrolls. *Alternatives:* bordered single-line; borderless + multiline.
*Rationale:* matches both references, enables pasting/editing longer prompts.
*Cost:* requires `Input` → `TextArea` migration and wizard re-plumbing (C7).

**D4 — Slash-command menu.** Dropdown popup above the input: command +
description, live-filtered, `↑/↓` select, `Enter`/`Tab` accept, `Esc` dismiss;
arg commands show an interactive value picker (D22.3). *Alternatives:* keep
ghost-text suggester; no change. *Rationale:* most visible slash upgrade;
descriptions already exist in `_COMMANDS`.

**D5 — Thinking/streaming indicator.** Animated **line spinner `|/-\` @ ~8 fps**
(D23.1) + elapsed seconds + live token estimate + "esc to interrupt", collapsing
into the streamed reply.
*Alternatives:* spinner + elapsed only; static hint. *Rationale:* Ricky wants the
full "alive" feel including token count.

**D6 — Live token source.** Local **`tiktoken` estimate** of streamed text
(labeled approximate); authoritative real `usage` total still shown on the
end-of-turn meta line. *Alternatives:* plumb real incremental usage through the
provider/orchestrator. *Rationale:* streaming carries text not usage (C6);
`tiktoken` is already a dep and works for all providers; real usage often only
arrives at end-of-stream so it wouldn't tick live.

**D7 — Status bar.** Persistent, bottom-fixed, **single-row** bar:
`provider | model | effort | context % | token usage | rough price`, with
user-customizable segments. *Rationale (Ricky):* always-on orientation, exactly
1 row of chrome, user-editable.

**D8/D11 — Pricing source.** Use `rickshaw_ai`'s built-in per-model `pricing`
(`ModelInfo.pricing`) as defaults + user overrides in `settings.json`; `—` when
unknown; price is a rough estimate off the D6 token count. *Alternatives:* new
bundled `pricing.yaml`; user-supplied only. *Rationale:* pricing metadata already
ships in the code (finding during interview) — no new table needed; overrides
keep it accurate for custom/newer models. *Note:* bundled rates may drift;
documented as approximate.

**D9 — Status-bar customization.** Config-driven segment list in
`settings.json`, e.g. `"status_bar": ["provider","model","effort","context",
"tokens","price"]`, from a fixed documented vocabulary; default = all six in that
order; unknown names ignored with a warning. *Alternatives:* in-TUI `/statusbar`
editor; free-form template string. *Rationale:* least risk, matches existing
settings persistence; an in-TUI editor is a clean future add (Out of Scope).

**D10 — Context % basis.** Compute against the active model's `context_window`
from `rickshaw_ai` metadata (confirmed present, e.g. Anthropic `ctx=200_000`); if
missing/0, warn and render `—`. *Alternatives:* bundled window table; drop the
segment. *Rationale:* authoritative per-model data already ships.

**D12 — Segment basis (hybrid).** `token usage` and `price` are cumulative
session totals; `context %` reflects current-turn context fill. *Rationale:*
price/usage answer "how much this session"; a % most naturally means "how full is
the window now".

**D13 — Input history.** Persistent history in `~/.rickshaw/history`, ↑/↓ recall,
storing plain messages **and** slash commands (D22.5); **rolling cap of 1,000
entries** (D23.4). *Alternatives:* in-session
only; none. *Rationale:* standard, cheap. *Interaction (C7/D3):* ↑/↓ triggers
history only when cursor is on first/last line and no menu is open.

**D14 — Message styling.** Role-labeled blocks with brand glyphs (D23.2): the
user message is prefixed with the amber `›` marker; the assistant message carries
an `o--o` mark + dim `rickshaw` label with a subtle left gutter; keep hairline
rules. *Alternatives:* plain dim `you`/`rickshaw` text; glyph-only; keep `›` +
rules only; tinted boxes. *Rationale:* ties message styling to the existing brand
identity while keeping turns scannable.

**D15 — Keybindings.** `Enter` submit; `Shift+Enter`/`Ctrl+J` newline; `Esc`
interrupt turn / dismiss menu / cancel wizard / clear input line; `↑/↓` menu-nav
when open else history; **`Ctrl+C` double-tap to quit** (first press: "press
again to quit"), single `Ctrl+C` cancels a running turn; `Ctrl+L` clear
transcript. *Rationale:* blends both references; double-tap avoids accidental
quits. *Note:* overrides Textual's default immediate `Ctrl+C` quit.

**D16/D17/D21 — Theme.** Full re-theme to a **new Rickshaw identity** inspired by
both references; **one built-in theme** now (selection setting is future).
*Alternatives:* refine monochrome+amber; restrained accent; match one reference;
2–3 themes. *Rationale:* Ricky wants a distinct, polished look. **Concrete
palette (adopted):**
| Token | Hex | Use |
|---|---|---|
| Background | `#0f1113` | screen |
| Surface | `#16181b` | welcome panel, status bar |
| Border/rule | `#2a2f36` | frames, hairline rules |
| Primary text | `#e6e8ea` | body/assistant text |
| Secondary/meta | `#8b929c` | meta lines, hints |
| Accent | `#e0a86b` | user `›` marker, focus, spinner, links |
| Assistant label | `#7fb0c9` | `rickshaw` role label |
| Warning | `#d98a3d` | warnings |
| Error/degraded | `#d16a5a` | errors, degraded banner |
| Success | `#7fae7f` | confirmations |

Applied via Textual CSS; 256-color terminals get nearest-color degrade. Welcome
panel and input frame use Textual **`round`** border style (D23.3).

**D18/D19 — Foundation (Textual, not Ink).** During the interview Ricky explored
switching to **Ink** (the TS framework Claude Code/Codex use), including an
Ink-frontend + Python-backend split and even a full TS rewrite, then settled on
**staying with Python + Textual and enhancing the existing TUI**. *Alternatives
evaluated:* (a) Ink + Python core over stdio JSON-lines; (b) Ink + Python core
over local HTTP/WebSocket; (c) full TypeScript rewrite retiring the Python core.
*Rationale:* the desired look & feel is fully achievable in Textual; Ink's only
real payoff (same stack as Claude Code) would cost retiring or bridging the
mature Python core (`rickshaw_ai`, semantic memory, orchestrator) for no
user-visible gain, and would stall the actual UX work behind a port. Staying on
Textual keeps the engine intact and ships the polish fastest.

**D20 — PRD scope.** Moot given D19 (no rewrite) — this remains a focused
enhancement of the existing Textual app.

**D22 — Interaction defaults.** (1) Welcome panel re-renders after `/clear`.
(2) Newline = `Shift+Enter` **and** `Ctrl+J`. (3) Arg commands show an
interactive value picker. (4) Narrow-terminal status bar drops price → tokens →
context first; welcome panel goes compact; never wrap-breaks. (5) History stores
messages and slash commands.

**D23 — Visual detail decisions (formerly open items).**
1. **Spinner:** line spinner `|/-\` at ~8 fps. *Alternatives:* braille dots @
   ~10 fps; dot pulse @ ~4 fps.
2. **Role glyphs:** amber `›` before the user message; `o--o` mark + dim
   `rickshaw` label for the assistant. *Alternatives:* plain text labels;
   glyph-only.
3. **Border style:** Textual `round` for the welcome panel and input frame.
   *Alternatives:* `heavy`; `ascii`.
4. **History cap:** rolling **1,000** entries in `~/.rickshaw/history`.
   *Alternatives:* 200; unbounded.

---

## 6. Out of Scope

- **Agentic coding workflow** — file reading/editing, shell/tool execution,
  diffs, plan mode, `@`-file mentions. This PRD is look & feel only.
- **Ink / TypeScript rewrite** or any framework change (explicitly rejected, D19).
- **Multiple selectable themes / in-TUI theme switcher** — one built-in theme now
  (D16/D17); theme selection is a future add.
- **In-TUI status-bar editor** (`/statusbar` wizard) — customization is via
  `settings.json` only for now (D9).
- **Real incremental-usage plumbing** through providers/orchestrator — live count
  is a `tiktoken` estimate (D6); real totals remain end-of-turn only.
- **Streaming through the tool-call loop** — unchanged provider-side limitation.
- **New providers / memory/embedding algorithm changes** — backend behavior is
  reused as-is.
- **Remote/attach/multi-client** operation and any web UI — not pursued (a factor
  in rejecting the HTTP/WS transport option).
- **Windows-specific terminal quirks** beyond truecolor/256-color degrade — not
  specially targeted.

---

## 7. Open Items for Sign-off

**None.** All previously-open visual details are resolved in D23 (spinner = line
`|/-\` @ ~8 fps; role glyphs = amber `›` / `o--o` + `rickshaw`; border =
`round`; history cap = 1,000 rolling). PRD is fully specified and ready for
sign-off.
