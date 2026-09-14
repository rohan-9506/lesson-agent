#!/usr/bin/env python3
"""CLI entry point.

    python main.py --topic rag
    python main.py --topic rag --provider grok
    python main.py --topic rag --inject-error jargon   # demo: evaluator catches a deliberate error
"""
from __future__ import annotations

import argparse
import sys

from lesson_agent.config import SETTINGS
from lesson_agent.graph import run_pipeline
from lesson_agent.llm_client import LLMClient, LLMError
from lesson_agent.output_writer import export_pdf, write_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-evaluating lesson content generator")
    parser.add_argument("--topic", default="rag", help="Topic key, e.g. 'rag' (see lesson_agent/topics.py)")
    parser.add_argument("--provider", choices=["gemini", "grok", "groq"], default=None, help="Override LLM_PROVIDER")
    parser.add_argument("--max-retries", type=int, default=SETTINGS.max_retries)
    parser.add_argument(
        "--inject-error",
        choices=["jargon", "readability", "wrong_fact"],
        default=None,
        help="Demo/test hook: force a specific defect into the first draft so the evaluator has "
        "something concrete to catch (see lesson_agent/test_hooks.py).",
    )
    parser.add_argument(
        "--pdf",
        action="store_true",
        help="Also export the shipped lesson as a PDF (best-effort, needs a local Chrome/Edge/Chromium).",
    )
    args = parser.parse_args()

    try:
        llm = LLMClient(provider=args.provider)
    except LLMError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1

    print(f"Provider: {llm.provider}")
    print(f"Topic: {args.topic}  |  max_retries: {args.max_retries}  |  inject_error: {args.inject_error}\n")

    try:
        final_state = run_pipeline(
            args.topic,
            llm=llm,
            max_retries=args.max_retries,
            inject_error=args.inject_error,
        )
    except LLMError as exc:
        print(f"LLM call failed: {exc}", file=sys.stderr)
        return 1

    lesson_path, log_path = write_outputs(final_state)

    print(f"Status: {final_state['status']}  (attempts: {final_state['attempt']})")
    print(f"Lesson written to:        {lesson_path}")
    print(f"Rejection log written to: {log_path}")

    if args.pdf:
        pdf_path = export_pdf(lesson_path)
        if pdf_path:
            print(f"PDF written to:           {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
