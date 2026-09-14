"""Shared state passed between LangGraph nodes."""
from __future__ import annotations

from typing import Optional, TypedDict

from .rubric import EvaluationResult


class LessonState(TypedDict, total=False):
    topic_key: str          # e.g. "rag" -- used to look up the KnowledgeCard
    topic_label: str        # human-readable topic name for the lesson title
    lesson_text: str        # current draft
    attempt: int            # 1-indexed generation attempt number
    max_retries: int        # regenerations allowed after the first attempt
    evaluation: Optional[EvaluationResult]
    rejection_log: list[dict]   # accumulated across attempts: {attempt, failed, reasons}
    status: str              # "generating" | "evaluating" | "passed" | "force_shipped"
    inject_error: Optional[str]  # test hook: force a specific failure mode on attempt 1
