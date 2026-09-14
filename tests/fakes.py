"""A fake LLM client for tests: no network calls, fully deterministic.

Implements the same interface as lesson_agent.llm_client.LLMClient
(complete / complete_json) via duck typing, so it can be passed anywhere an
LLMClient is expected.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

# A simple, short-sentence lesson that satisfies the two deterministic
# checks (readability, jargon-explained) on its own, so tests can isolate
# whichever dimension they actually want to exercise via judge_script.
GOOD_LESSON_TEXT = """# Introduction to RAG

## What it is
RAG stands for Retrieval-Augmented Generation. It means the AI first finds
useful text, then writes an answer using that text.

## Why it matters
A language model (a computer program trained to write text) can forget facts
or make things up. RAG helps fix this. It lets the model look things up
first, like checking notes before answering.

## How it works
First, we cut documents into small pieces called chunks. Each chunk is
turned into an embedding (a list of numbers that captures its meaning). All
embeddings go into a vector database, which is a database made for finding
similar meanings fast.

When you ask a question, the system turns your question into an embedding
too. It compares it to the stored chunks using cosine similarity, a way to
measure how close two embeddings are. It fetches the top-k retrieval result,
meaning the best few matches.

For example, imagine asking "What is RAG?" The system finds the chunk that
explains RAG, adds it to the model's prompt (the instructions given to the
model), and the model writes an answer using that chunk.

This is different from fine-tuning, which means retraining the model itself
on new data. RAG does not change the model. It just gives it better notes to
read before answering.

## Recap
RAG finds relevant text first, then generates an answer using it. This
keeps answers fresh and grounded, without retraining the model.
"""


class FakeLLMClient:
    """judge_script: dict mapping a keyword found in the judge prompt to a
    list of {"passed": bool, "reason": str} dicts, consumed in order (one per
    call to that check). Once a list is exhausted, defaults to passed=True.
    """

    _CHECK_MARKERS = {
        "VERIFIED FACTS:": "accurate_and_grounded",
        "concrete example, analogy": "teaches_by_example",
        "REQUIRED KEY POINTS:": "covers_key_points",
        "coherent teaching flow": "coherent_teaching_flow",
    }

    def __init__(self, judge_script: dict[str, list[dict[str, Any]]] | None = None, lesson_text: str = GOOD_LESSON_TEXT):
        self.provider = "fake"
        self.judge_script = judge_script or {}
        self._call_counts: dict[str, int] = defaultdict(int)
        self.lesson_text = lesson_text
        self.generation_calls = 0

    def complete(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.6,
                 json_mode: bool = False, max_tokens: int = 4096) -> str:
        self.generation_calls += 1
        return self.lesson_text

    def complete_json(self, system_prompt: str, user_prompt: str, *, temperature: float = 0.0,
                       max_tokens: int = 2048) -> dict[str, Any]:
        check_name = None
        for marker, name in self._CHECK_MARKERS.items():
            if marker in user_prompt:
                check_name = name
                break
        if check_name is None:
            return {"passed": True, "reason": "no script matched; default pass"}

        script = self.judge_script.get(check_name, [])
        idx = self._call_counts[check_name]
        self._call_counts[check_name] += 1
        if idx < len(script):
            return script[idx]
        return {"passed": True, "reason": f"default pass ({check_name}, call {idx})"}
