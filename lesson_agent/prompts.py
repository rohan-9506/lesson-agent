"""Prompt templates for the generation node."""
from __future__ import annotations

from .topics import KnowledgeCard

GENERATE_SYSTEM_PROMPT = """You are a curriculum author writing a BEGINNER lesson for learners with:
- a 12th-grade education from India
- limited English vocabulary
- a non-English-medium schooling background
- zero prior background in this topic, and a strong motivation to start an AI career

Write in plain, simple, short SENTENCES -- but do not confuse "simple" with "short lesson". \
A one-line bullet like "vector - a list of numbers that represents its meaning" is NOT enough \
for this reader: it is a dictionary entry, not a taught concept. For every new idea or term, \
write 2-4 full sentences of plain-language explanation before moving on -- say what it is, \
why it exists / what problem it solves, and connect it back to the idea before it, in \
ordinary conversational prose. Use short bullet lists only to summarize points you have \
already explained in prose, never as the sole explanation of a new concept. If you name a \
specific tool or product (e.g. a database or library), briefly say in plain words what it is \
for -- do not just drop the name and move on. If you must use a technical term, immediately \
explain it in a beginner-friendly way, in the same or next sentence -- never assume the \
reader already knows it. Use at least one concrete, fully worked-through example or analogy, \
explained step by step, not just named. Structure the lesson as clear sections in Markdown: \
"What it is", "Why it matters", "How it works" (with an example), and a short "Recap". \
Only state facts you are given below as verified, or safe, well-known background -- never \
invent specific technical details you are not sure of."""


def build_generation_prompt(
    card: KnowledgeCard,
    *,
    feedback: str | None = None,
    memory_guidance: str = "",
) -> str:
    facts_block = "\n".join(f"- {f}" for f in card.grounding_facts)
    points_block = "\n".join(f"- {p}" for p in card.required_points)

    parts = [
        f"Write a standalone beginner lesson on: {card.topic}",
        "\nVERIFIED FACTS you may draw on (do not contradict or go beyond these with invented specifics):",
        facts_block,
        "\nKEY POINTS the lesson MUST cover (all of them, at least briefly):",
        points_block,
    ]

    if memory_guidance:
        parts.append("\n" + memory_guidance)

    if feedback:
        parts.append(
            "\nThis is a REGENERATION. The previous draft was rejected by the evaluator for these "
            "specific reasons -- fix every one of them in this new draft, don't just reword around them:\n"
            + feedback
        )

    parts.append(
        "\nWrite the full lesson now in Markdown. Do not include any preamble like 'Here is the lesson' "
        "-- start directly with the title."
    )
    return "\n".join(parts)
