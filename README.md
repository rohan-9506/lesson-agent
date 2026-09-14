# Self-Evaluating Lesson Content Generator

An agentic system that generates a beginner lesson on a topic, judges it against a hard
pass/fail rubric, and regenerates using the specific failure reasons until it clears the
bar (or exhausts its retry budget) — the **generate → evaluate → regenerate** loop, built
as a [LangGraph](https://github.com/langchain-ai/langgraph) state machine.

Built for topic: **Introduction to RAG (Retrieval-Augmented Generation)**, for a learner
who is a 12th-grade graduate from India with limited English vocabulary and no technical
background.

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

The four LLM-judge checkpoints run concurrently; the two deterministic ones need no API
key at all. `max_retries` (default 2) is a structural bound on the graph, not a counter —
once spent, the conditional edge routes to `force_ship` instead of another `generate`, so
a lesson that still fails ships visibly flagged rather than looping forever. Cross-run
memory (`data/memory.json`) tracks per-topic recurring failures and injects "don't repeat
this" guidance into future generation prompts. The LLM client (`llm_client.py`) is
provider-agnostic (Gemini / Groq / Grok behind one interface), with 429 retry/backoff and
a multi-model fallback chain.

## Repo layout

```
lesson_agent/
  config.py         # settings (provider, models, retry budget, max output tokens) from env
  llm_client.py      # Gemini + Groq + Grok behind one interface, with 429 retry/backoff
  topics.py          # per-topic knowledge card: verified facts, required points, glossary
  prompts.py         # generation system/user prompt templates
  rubric.py          # the 6 checkpoints (2 deterministic, 4 LLM-judge, run concurrently)
  memory.py          # cross-run JSON store + "recurring failure" guidance
  state.py           # LangGraph state schema
  graph.py           # the generate → evaluate → regenerate StateGraph
  test_hooks.py       # deliberate-error injection, for the demo + tests
  output_writer.py    # renders final state -> lesson.md + rejection_log.md, incl. PDF export
main.py               # CLI entry point
tests/                 # unit tests using a fake, scripted LLM client (no API key needed)
outputs/<topic>/       # generated lesson.md, rejection_log.md, run_state.json per run
data/memory.json       # cross-run memory (created on first run)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env           # then fill in your API key
```

Pick one provider in `.env` via `LLM_PROVIDER=gemini|groq|grok`. `MAX_GENERATION_TOKENS`
is optional — leave unset to use the built-in default for whichever provider/model you
picked, or set it to override.

## Run

```bash
# Full pipeline, using whichever provider is set in .env
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

# Export the final lesson as a PDF
python main.py --topic rag --max-retries 4 --pdf
```

Output lands in `outputs/introduction-to-rag-retrieval-augmented-generation/`:
`lesson.md` (the shipped lesson), `rejection_log.md` (what failed, why, and what changed
on each retry), and `run_state.json` (the same data as structured JSON).

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

All 10 tests run against `tests/fakes.py::FakeLLMClient` — a scripted stand-in with the
same interface as the real client — so the full suite needs **no API key and makes no
network calls**.

## Adding a new topic

Add an entry to `KNOWLEDGE_CARDS` in `lesson_agent/topics.py`: verified facts, required
key points, and a glossary of terms needing inline explanation. Everything else (prompts,
rubric, graph, memory) is topic-agnostic.

## Known trade-offs

- **Readability threshold (Flesch Reading Ease ≥ 55) is a single number** — a proxy for
  "not too dense," not a substitute for the LLM-judge checks.
- **The jargon glossary is curated per-topic, not learned** — a confusing term outside
  the glossary could slip through checkpoint 4.
- **`force_ship` still ships** — kept as auto-ship-but-flagged so the loop is demoable
  end-to-end without a human in it; a real pipeline would page a human instead.
