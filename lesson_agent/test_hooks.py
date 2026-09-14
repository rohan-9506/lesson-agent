"""Deliberate-error injection, used for the assessment's required demo of
"your evaluator catching a deliberate error" and for deterministic tests.

This does NOT touch the evaluator at all -- it only corrupts the generator's
first-attempt output in a specific, known way, so we can show the evaluator
independently catching it on attempt 1 and the regenerate step fixing it on
attempt 2. Available modes:

  "jargon"       -- appends a sentence using an unexplained glossary term.
  "readability"  -- appends a long, dense, jargon-stacked sentence.
  "wrong_fact"   -- appends a sentence contradicting a grounding fact
                    (fine-tuning changes weights, not RAG).
"""
from __future__ import annotations

from .topics import KnowledgeCard


def corrupt_lesson(lesson_text: str, mode: str, card: KnowledgeCard) -> str:
    mode = mode.lower().strip()

    if mode == "jargon":
        # Pick a glossary term and use it with zero nearby explanation.
        term = "context window"
        addition = (
            f"\n\n## A Quick Technical Note\n"
            f"Under the hood, the ranking step relies on {term} to sort the candidates before they are "
            f"passed along."
        )
        return lesson_text + addition

    if mode == "readability":
        addition = (
            "\n\n## A Quick Technical Note\n"
            "Consequently, the aforementioned methodological paradigm necessitates the orchestration of "
            "multidimensional vector representations, which, notwithstanding their computational overhead, "
            "facilitate an epistemologically robust retrieval mechanism that substantially ameliorates the "
            "hallucinatory propensities intrinsic to autoregressive language modeling architectures."
        )
        return lesson_text + addition

    if mode == "wrong_fact":
        addition = (
            "\n\n## A Quick Technical Note\n"
            "In fact, RAG works by fine-tuning the model's weights on your documents every time you ask a "
            "question, so the model permanently learns your private data after just one query."
        )
        return lesson_text + addition

    raise ValueError(f"Unknown inject_error mode: {mode!r} (use 'jargon', 'readability', or 'wrong_fact')")
