"""The rubric: six hard pass/fail checkpoints, no partial credit.

Design choice: this is a deliberate mix of two *deterministic* checks and
four *LLM-judge* checks, not six LLM calls.

- Deterministic checks (jargon, readability) are exact, reproducible, free,
  and don't depend on an API key -- they can be (and are, in tests/) unit
  tested with zero network calls. They also happen to be the two dimensions
  easiest to sabotage on purpose, which is why the "deliberate error" demo
  in the assessment brief hooks into the jargon checker.
- LLM-judge checks (grounded, teaches-by-example, key-point coverage,
  coherent flow) need semantic understanding a regex can't give you. Each is
  a separate, narrow call with a strict "respond with exactly this JSON
  shape, binary pass/fail, no partial credit" instruction -- one dimension
  per call rather than one call trying to judge everything, so a single
  judge mistake doesn't take the whole evaluation down with it and each
  failure reason is attributable to one specific checkpoint.

Every check returns a CheckResult(passed, reason). The lesson ships only if
ALL checks pass.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from .config import SETTINGS
from .llm_client import LLMClient
from .topics import KnowledgeCard

READABILITY_MIN_FLESCH_EASE = 55.0  # "fairly easy" or better; tuned for a 12th-grade,
                                     # non-native-English-medium reader


@dataclass
class CheckResult:
    name: str
    passed: bool
    reason: str


# --------------------------------------------------------------------------- #
# Deterministic checks
# --------------------------------------------------------------------------- #

_EXPLANATION_CUES = re.compile(
    r"\b(which means|i\.e\.|in other words|meaning|means that|is a way of|"
    r"is basically|think of it as|simply put|in simple terms)\b",
    re.IGNORECASE,
)
_PARENTHETICAL_DEFN = re.compile(r"\([^)]{6,120}\)")


def check_jargon(lesson_text: str, glossary: dict[str, str]) -> CheckResult:
    """Fail if any glossary term is used without a nearby explanation.

    'Nearby' = the sentence containing the term, plus the one before/after,
    contains either an explanation cue phrase, a parenthetical gloss, or at
    least two content words drawn from that term's glossary definition.
    """
    sentences = re.split(r"(?<=[.!?])\s+", lesson_text)
    unexplained: list[str] = []

    for term, definition in glossary.items():
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        occurrences = [i for i, s in enumerate(sentences) if pattern.search(s)]
        if not occurrences:
            continue  # term never used -- nothing to check

        def_words = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", definition)}

        explained_somewhere = False
        for idx in occurrences:
            window = " ".join(sentences[max(0, idx - 1) : idx + 2])
            if _EXPLANATION_CUES.search(window) or _PARENTHETICAL_DEFN.search(window):
                explained_somewhere = True
                break
            window_words = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", window)}
            if len(window_words & def_words) >= 2:
                explained_somewhere = True
                break

        if not explained_somewhere:
            unexplained.append(term)

    if unexplained:
        return CheckResult(
            "no_unexplained_jargon",
            False,
            "Used without a nearby beginner-friendly explanation: " + ", ".join(sorted(unexplained)),
        )
    return CheckResult("no_unexplained_jargon", True, "All jargon terms used are explained in context.")


_VOWEL_GROUPS = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _count_syllables(word: str) -> int:
    """Lightweight heuristic syllable counter (vowel-group counting with the
    standard silent-'e' adjustment). Deliberately dependency-free: textstat's
    default syllable counter pulls in an NLTK corpus download, which is a
    network dependency this pipeline should not need just to grade prose."""
    word = word.lower().strip(".,!?;:\"'()")
    if not word:
        return 0
    groups = _VOWEL_GROUPS.findall(word)
    count = len(groups)
    if word.endswith("e") and not word.endswith("le") and count > 1:
        count -= 1
    return max(count, 1)


def _flesch_reading_ease(text: str) -> float:
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z']+", text)
    if not sentences or not words:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    n_sentences, n_words = len(sentences), len(words)
    return 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (syllables / n_words)


def check_readability(lesson_text: str) -> CheckResult:
    score = _flesch_reading_ease(lesson_text)
    if score >= READABILITY_MIN_FLESCH_EASE:
        return CheckResult(
            "beginner_friendly_language", True, f"Flesch Reading Ease {score:.1f} >= {READABILITY_MIN_FLESCH_EASE}."
        )
    return CheckResult(
        "beginner_friendly_language",
        False,
        f"Flesch Reading Ease {score:.1f} is below the {READABILITY_MIN_FLESCH_EASE} threshold "
        "-- sentences are too long/complex for a beginner with limited English vocabulary.",
    )


# --------------------------------------------------------------------------- #
# LLM-judge checks
# --------------------------------------------------------------------------- #

_JUDGE_SYSTEM = """You are a strict binary grader for beginner learning content. \
You output ONLY a JSON object: {{"passed": true|false, "reason": "<one or two sentences>"}}. \
No partial credit is allowed -- if the check is even partially unmet, passed must be false. \
Be specific and concrete in "reason": quote or paraphrase the exact part of the lesson that \
caused a fail, or that satisfied the check if it passed."""


def _run_judge(llm: LLMClient, instruction: str, lesson_text: str, name: str) -> CheckResult:
    user_prompt = f"{instruction}\n\n--- LESSON TEXT START ---\n{lesson_text}\n--- LESSON TEXT END ---"
    # Judge responses are tiny: {"passed": true|false, "reason": "one or two sentences"}.
    # Capping at 512 tokens avoids wasting the rate-limit budget on output tokens the
    # judge never uses, and also prevents json_validate_failed truncation errors that
    # reasoning models (openai/gpt-oss-*) produce when max_tokens is too large.
    data = llm.complete_json(_JUDGE_SYSTEM, user_prompt, temperature=0.0, max_tokens=512)
    passed = bool(data.get("passed", False))
    reason = str(data.get("reason", "")).strip() or "(model gave no reason)"
    return CheckResult(name, passed, reason)


def check_grounded(llm: LLMClient, lesson_text: str, card: KnowledgeCard) -> CheckResult:
    facts_block = "\n".join(f"- {f}" for f in card.grounding_facts)
    instruction = (
        "Below is a list of VERIFIED FACTS about the topic. Read the lesson and check whether every "
        "factual/technical claim it makes is either (a) consistent with these verified facts, or (b) safe, "
        "well-established general knowledge that does not contradict them. FAIL if the lesson states "
        "anything that contradicts a verified fact, or invents a specific technical detail (a number, a "
        "named tool behavior, a mechanism) that is not supported by the facts or common knowledge and could "
        "mislead a beginner into believing something false.\n\nVERIFIED FACTS:\n" + facts_block
    )
    return _run_judge(llm, instruction, lesson_text, "accurate_and_grounded")


def check_teaches_by_example(llm: LLMClient, lesson_text: str) -> CheckResult:
    instruction = (
        "Check whether the lesson teaches by example AND explains with enough depth for a reader with "
        "zero background and limited English vocabulary. It must include at least one concrete example, "
        "analogy, worked scenario, or walkthrough that makes the core concept tangible -- not just an "
        "abstract definition. FAIL if the lesson only defines terms abstractly with no concrete "
        "illustration a beginner could latch onto. Separately, FAIL if the lesson reads like a dictionary "
        "or glossary -- e.g. a list of one-line bullet definitions ('term - short phrase') for its major "
        "concepts, with no full-sentence elaboration connecting each idea to why it matters or how it "
        "relates to the concept before it. A beginner should be able to follow the reasoning in prose, not "
        "just memorize a list of terse definitions."
    )
    return _run_judge(llm, instruction, lesson_text, "teaches_by_example")


def check_key_points(llm: LLMClient, lesson_text: str, card: KnowledgeCard) -> CheckResult:
    points_block = "\n".join(f"{i+1}. {p}" for i, p in enumerate(card.required_points))
    instruction = (
        "Below is a checklist of key points a beginner lesson on this topic MUST cover, at least briefly. "
        "FAIL if the lesson is missing ANY one of them (no partial credit -- all must be present). List which "
        "point(s), if any, are missing in your reason.\n\nREQUIRED KEY POINTS:\n" + points_block
    )
    return _run_judge(llm, instruction, lesson_text, "covers_key_points")


def check_coherent_flow(llm: LLMClient, lesson_text: str) -> CheckResult:
    instruction = (
        "Check whether the lesson has a coherent teaching flow: it should move logically from 'what it is' "
        "to 'why it matters' to 'how it works', with an example, in an order a total beginner could follow "
        "without getting lost, and without circular or contradictory explanations. FAIL if the ordering is "
        "confusing, sections contradict each other, or a concept is used before it is introduced."
    )
    return _run_judge(llm, instruction, lesson_text, "coherent_teaching_flow")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

ALL_CHECK_NAMES = [
    "accurate_and_grounded",
    "beginner_friendly_language",
    "teaches_by_example",
    "no_unexplained_jargon",
    "covers_key_points",
    "coherent_teaching_flow",
]


@dataclass
class EvaluationResult:
    passed: bool
    checks: dict[str, CheckResult]

    def failed_checks(self) -> list[CheckResult]:
        return [c for c in self.checks.values() if not c.passed]

    def as_reason_map(self) -> dict[str, str]:
        return {c.name: c.reason for c in self.failed_checks()}


def evaluate_lesson(llm: LLMClient, lesson_text: str, card: KnowledgeCard) -> EvaluationResult:
    # The four LLM-judge checks are independent of each other (each is its own
    # request/response round trip against the provider), so running them
    # sequentially only adds up their latencies for no benefit. Firing them
    # concurrently collapses that to roughly the slowest single call instead
    # of the sum of all four -- worthwhile on providers with generous
    # per-minute budgets (Gemini, Grok). The two deterministic checks (jargon,
    # readability) are pure local computation, so they just run inline.
    #
    # Groq's free tier is the exception: its RPM/TPM budget is small enough
    # that firing all 4 judges at once just means all 4 trip the 429 limiter
    # together, and each one independently runs its own multi-attempt backoff
    # in LLMClient._post_with_retry -- 4x the contention and 4x the retry
    # noise, with no actual latency win since Groq serializes them via 429s
    # anyway. Run judges one at a time there instead.
    llm_checks = (
        lambda: check_grounded(llm, lesson_text, card),
        lambda: check_teaches_by_example(llm, lesson_text),
        lambda: check_key_points(llm, lesson_text, card),
        lambda: check_coherent_flow(llm, lesson_text),
    )
    max_workers = 1 if SETTINGS.provider == "groq" else len(llm_checks)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        llm_results = [f.result() for f in [pool.submit(fn) for fn in llm_checks]]

    checks: list[CheckResult] = [
        *llm_results,
        check_readability(lesson_text),
        check_jargon(lesson_text, card.jargon_glossary),
    ]
    checks_by_name = {c.name: c for c in checks}
    return EvaluationResult(passed=all(c.passed for c in checks), checks=checks_by_name)
