# Self-Evaluating Lesson Content Generator

An agentic system that generates a beginner lesson on a topic, judges it against a hard
pass/fail rubric it does not get to negotiate with, and regenerates using the specific
failure reasons until it clears the bar (or exhausts its retry budget) — the
**generate → evaluate → regenerate** loop, built as an explicit, inspectable
[LangGraph](https://github.com/langchain-ai/langgraph) state machine.

Built for the take-home assessment topic: **Introduction to RAG (Retrieval-Augmented
Generation)**, for a learner who is a 12th-grade graduate from India with limited English
vocabulary and no technical background.

## Why this design

**Why LangGraph, not a `while` loop.** The brief asks to see the architecture and the
reasoning, not just a passing lesson. A hand-rolled loop hides its control flow inside
imperative code; a `StateGraph` makes it a diagram you can point at: `generate` and
`evaluate` are named nodes, and the retry bound is a structural property of the
conditional edge (`ship` / `regenerate` / `force_ship`), not a stray counter someone could
forget to check. It's also what makes the run trivially loggable — every node hands back
a plain `LessonState`, so the rejection log is just an accumulation of that state, not a
side-channel.

**Why the rubric is a mix of deterministic checks and LLM judges, not six LLM calls.**
Two of the six dimensions — jargon and readability — are exact, reproducible, and *free*:
a regex-based jargon scanner and a from-scratch Flesch Reading Ease score. They need no API
key at all, which is also why they're the two dimensions the test suite and the
"deliberate error" demo lean on — you can watch the evaluator catch a planted defect with
zero network calls. The other four (accurate & grounded, teaches by example, covers key
points, coherent flow) genuinely need semantic judgment a regex can't give you, so each is
its own narrow LLM call with a strict "binary pass/fail, no partial credit, give a
specific reason" instruction. One call per dimension, not one call judging everything,
so a single bad judgment doesn't take the whole evaluation down and every failure is
attributable to exactly one checkpoint.

**Why "accurate & grounded" is checked against a knowledge card, not vibes.** Asking an
LLM "is this accurate?" with no reference point is asking it to grade its own homework
against its own priors — exactly the failure mode RAG itself exists to fix. Instead,
`lesson_agent/topics.py` defines a short list of **verified facts** about RAG and a
checklist of **required key points**. The generator is told to only make claims consistent
with these; the evaluator checks the draft's claims against the same list. This also
means the "wrong fact" deliberate-error demo has a crisp, checkable failure mode instead
of a fuzzy one.

**Why the loop is guaranteed to terminate.** `max_retries` (default 2) bounds the number
of regenerations structurally: the conditional edge routes to `force_ship` — not another
`generate` — once the budget is spent. A lesson that still fails after every retry still
ships, but visibly: `status: force_shipped` in the rejection log, never a silent pass.

**Why memory is a flat JSON file, not a database.** `data/memory.json` accumulates one
record per attempt across *all* runs. Before generating, the system looks up this topic's
history and — if a dimension has failed 2+ times before — injects an explicit
"you have failed this before, for this reason, don't repeat it" instruction into the
generation prompt. That's the self-evolving piece: the prompt accumulates topic-specific
scar tissue across runs instead of making the same mistake every time. A JSON file is
enough here (single writer, small volume) and is trivial to open and show in the video.

**Why the LLM client is provider-agnostic.** `lesson_agent/llm_client.py` exposes one
`complete()` / `complete_json()` interface over Gemini, Groq, and xAI's Grok, selected by
an env var. Nothing else in the codebase imports a vendor SDK. This is also what makes the
system testable without any API key: `tests/fakes.py` implements the same interface with
canned, scripted responses, so the graph, rubric routing, and termination logic all have
real unit test coverage that runs in milliseconds with no network access.

**Why 429s are retried, not fatal.** A single attempt makes 1 generate call plus up to 4
LLM-judge calls, and a full run can make several attempts — enough to trip a free-tier
tokens-per-minute budget even with no other traffic on the key. `llm_client.py` retries on
HTTP 429, honoring the provider's own `Retry-After` header or its "try again in Xs" message,
with exponential backoff, instead of failing the whole run over a transient rate limit.

**Why there's a multi-model fallback chain behind the primary provider/model.** If the
active model fails for *any* reason — 429 rate limit, a retired/404 model id, a malformed
response — `complete()` first walks any other configured models in the *same* provider
family (`GROK_FALLBACK_MODELS` when `LLM_PROVIDER=grok`, `GEMINI_FALLBACK_MODELS` when it's
`gemini`), then always falls through to the Gemini fallback chain as the universal last
resort regardless of which provider was primary — each model draws from its own separate
free-tier quota (or is simply a different model id entirely), so one failing doesn't mean
they all will (see `LLMClient._attempt_chain`). Only once every configured model has failed
does the last error propagate. A free-tier 429 is almost always the whole TPM window blown,
not a one-off blip a retry would clear, and with other models available in the chain
there's no reason to wait at all — every model in the chain gets 0 retries, so a failure
moves to the next model immediately with no backoff sleep. The default model IDs
(`gemini-2.5-flash`, `grok-4.6`, and their fallback lists) were checked against each
vendor's docs in Sep 2026 — Gemini and xAI retire model IDs periodically (this project hit
exactly that with `gemini-2.5-flash-lite` going 404 mid-fallback), so revisit
`.env.example` if a request 404s on an unrecognized model name.

**Why generation has an explicit, provider-aware token budget.** `max_tokens` isn't a fixed
guess: `Settings.max_generation_tokens` (`lesson_agent/config.py`) resolves to whichever
provider/model is active — e.g. 8192 for `gemini-2.0-flash`, 16384 for Groq's
`qwen/qwen3.8-27b` — since requesting past a model's real output ceiling has no effect, and
the old unset default (a library fallback of 4096) was silently truncating lessons.
Override via `MAX_GENERATION_TOKENS` if you switch to a model with a higher cap. This
matters even more on Groq's reasoning models (`openai/gpt-oss-*`, `qwen/qwen3*`): they burn
part of that budget on invisible "thinking" tokens before writing anything visible, so
`llm_client.py` also sends `reasoning_effort` at the lowest value each family supports
(`"none"` for Qwen3, `"low"` for gpt-oss) to reclaim that budget for the actual lesson.

**Why the four LLM-judge calls run concurrently.** They're independent of each other
(`lesson_agent/rubric.py::evaluate_lesson`) — each is its own request/response round trip
with no shared state — so running them one after another only adds up their latencies for
no benefit. A `ThreadPoolExecutor` fires all four at once, collapsing a judging pass from
the sum of four calls to roughly the slowest single one, which matters on a rate-limited
free-tier key where a single call can itself take several seconds under 429 backoff.

**Why PDF export polls for the file instead of waiting on the browser to exit.** Some
Chrome builds finish writing the `--print-to-pdf` output within seconds but then hang
instead of exiting headless mode. Blocking on the subprocess (the obvious approach) would
misreport that as a timeout and discard an already-valid PDF. `output_writer.py::export_pdf`
instead watches the output file until its size stops changing, then terminates the browser
itself — export finishes in a few seconds instead of failing after a 120s timeout.

## Architecture

```
                 ┌─────────────┐
   topic ──────▶ │  generate   │◀────────────────────┐
                 └──────┬──────┘                      │
                        │ lesson draft                │ feedback = exact
                        ▼                              failure reasons
                 ┌─────────────┐                      │ from the last
                 │  evaluate   │                      │ evaluate() call
                 │ (6 hard     │                      │
                 │  checkpoints│──── any FAIL, ───────┘
                 │  no partial │      retries left
                 │  credit)    │
                 └──────┬──────┘
                        │
         all 6 PASS     │      any FAIL, retries exhausted
                ▼                         ▼
              ship (END)            force_ship (END, flagged)
```

Six checkpoints, evaluated every attempt (`lesson_agent/rubric.py`):

| # | Checkpoint | How it's checked |
|---|---|---|
| 1 | Accurate & grounded | LLM judge vs. a verified-facts knowledge card |
| 2 | Beginner-friendly language | Deterministic Flesch Reading Ease score |
| 3 | Teaches by example | LLM judge — needs a concrete example/analogy, not just definitions |
| 4 | No unexplained jargon | Deterministic glossary scan + nearby-explanation heuristic |
| 5 | Covers the key points | LLM judge vs. a required-points checklist |
| 6 | Coherent teaching flow | LLM judge — logical what→why→how ordering, no forward references |

## Repo layout

```
lesson_agent/
  config.py         # settings (provider, models, retry budget, per-provider max output tokens) from env
  llm_client.py      # Gemini + Groq + Grok behind one interface, with 429 retry/backoff and reasoning_effort handling
  topics.py          # per-topic knowledge card: verified facts, required points, glossary
  prompts.py         # generation system/user prompt templates
  rubric.py          # the 6 checkpoints (2 deterministic, 4 LLM-judge, run concurrently)
  memory.py          # cross-run JSON store + "recurring failure" guidance
  state.py           # LangGraph state schema
  graph.py           # the generate → evaluate → regenerate StateGraph
  test_hooks.py       # deliberate-error injection, for the demo + tests
  output_writer.py    # renders final state -> lesson.md + rejection_log.md; polls for PDF completion instead of waiting on Chrome to exit
main.py               # CLI entry point
tests/                 # unit tests using a fake, scripted LLM client (no API key needed)
outputs/<topic>/       # generated lesson.md, rejection_log.md, run_state.json per run
data/memory.json       # cross-run memory (created on first run)
```

## Setup

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then fill in your API key
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env    # then fill in your API key
```

**Windows (Command Prompt / cmd.exe):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
```

> **Tip:** After activating the virtual environment with one of the commands above, you can use plain `python` and `pip` commands for the rest of the session — no need for the full `.venv/bin/python` or `.venv\Scripts\python.exe` path.

Pick one provider in `.env` via `LLM_PROVIDER=gemini|groq|grok` (Groq and Grok are
different providers despite the similar name -- see the comment in `config.py`).
`MAX_GENERATION_TOKENS` is optional -- leave it unset to use the built-in default for
whichever provider/model you picked (see `Settings._default_max_generation_tokens` in
`config.py`), or set it to override that default.

## Run

> Activate your virtual environment first (see **Setup** above), then use plain `python`.
> If you prefer not to activate, use the platform-specific full path:
> - macOS/Linux: `.venv/bin/python main.py ...`
> - Windows: `.venv\Scripts\python.exe main.py ...`

```bash
# Full pipeline, using whichever provider is set in .env (LLM_PROVIDER=gemini|groq|grok)
python main.py --topic rag

# Force a specific provider for this run
python main.py --topic rag --provider groq

# Demo: force a deliberate defect into the first draft so the evaluator has
# something concrete to catch, then watch attempt 2 fix it
python main.py --topic rag --inject-error jargon
python main.py --topic rag --inject-error wrong_fact
python main.py --topic rag --inject-error readability

# Change the retry budget (default 2)
python main.py --topic rag --max-retries 1

# Export the final lesson as a PDF (requires --pdf flag)
python main.py --topic rag --max-retries 4 --pdf
```

Output lands in `outputs/introduction-to-rag-retrieval-augmented-generation/`:
`lesson.md` (the shipped lesson), `rejection_log.md` (what failed, why, and what changed
on each retry), and `run_state.json` (the same data as structured JSON).

## Tests

> Activate your virtual environment first (see **Setup** above), then run:

```bash
pip install pytest
python -m pytest tests/ -v
```

Or without activating, using the full path:

**macOS / Linux:**
```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/ -v
```

**Windows (PowerShell / cmd):**
```powershell
.venv\Scripts\pip.exe install pytest
.venv\Scripts\python.exe -m pytest tests/ -v
```

All 10 tests run against `tests/fakes.py::FakeLLMClient` — a scripted stand-in with the
same `complete`/`complete_json` interface as the real client — so the full suite needs
**no API key and makes no network calls**. Coverage includes: the two deterministic checks
in isolation, a deliberate jargon error caught and fixed on retry, a deliberate wrong-fact
error caught by the grounded-judge check, the loop terminating via `force_ship` when a
failure never clears (proving the retry budget is a hard bound, not best-effort), and the
output writer producing both deliverable files correctly.

## Adding a new topic

Add an entry to `KNOWLEDGE_CARDS` in `lesson_agent/topics.py`: a short list of verified
facts, the key points a beginner lesson must cover, and a glossary of terms that need
inline explanation if used. Everything else (prompts, rubric, graph, memory) is
topic-agnostic.

## Known trade-offs

- **Readability threshold (Flesch Reading Ease ≥ 55) is a single number.** It's a
  reasonable proxy for "not too dense," but it can't catch every way prose could be
  unclear to this specific audience (non-English-medium background) — it's a floor, not
  a substitute for the LLM-judge checks.
- **The jargon glossary is curated per-topic, not learned.** A term outside the glossary
  that's genuinely confusing would slip through checkpoint 4 (though it may still get
  caught by the "beginner-friendly language" or "coherent flow" judges).
- **`force_ship` still ships.** For a real production pipeline, exhausting retries should
  probably page a human instead of shipping — kept as auto-ship-but-flagged here so the
  loop is demoable end-to-end without a human in it, per the brief.
