"""Persistent, cross-run memory.

This is what makes the system "self-evolving" rather than a stateless
function call: every run's outcome (which rubric dimensions failed, why, and
what changed on retry) is appended to a JSON store on disk. Before generating,
we look up this topic's history and surface any *recurring* failure pattern
(a dimension that has failed more than once historically) as an explicit
extra instruction injected into the generation prompt -- "you have failed
this check before for this reason, do not repeat it." Over repeated runs the
prompt accumulates topic-specific scar tissue instead of making the same
mistake every time.

Storage is a plain JSON file rather than a database: the assessment doesn't
need concurrent writers, and a JSON file is trivial to inspect/diff in the
repo and in the video walkthrough.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import MEMORY_DB_PATH


@dataclass
class RunRecord:
    timestamp: str
    topic: str
    attempt: int
    passed: bool
    failed_dimensions: list[str]
    reasons: dict[str, str]  # dimension -> reason
    change_summary: str  # what was changed vs. the previous attempt (empty on attempt 1)


@dataclass
class MemoryStore:
    path: Path = field(default_factory=lambda: MEMORY_DB_PATH)

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _save(self, records: list[dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def record_run(self, record: RunRecord) -> None:
        records = self._load()
        records.append(asdict(record))
        self._save(records)

    def history_for_topic(self, topic: str) -> list[dict[str, Any]]:
        return [r for r in self._load() if r["topic"] == topic]

    def recurring_failure_guidance(self, topic: str, min_occurrences: int = 2) -> str:
        """Summarize dimensions that have failed >= min_occurrences times historically
        for this topic, across ALL prior runs (not just the current one), as a short
        block of extra instructions to prepend to the generation prompt.
        """
        history = self.history_for_topic(topic)
        dim_reasons: dict[str, list[str]] = {}
        counts: Counter[str] = Counter()
        for rec in history:
            for dim in rec["failed_dimensions"]:
                counts[dim] += 1
                reason = rec["reasons"].get(dim, "")
                if reason:
                    dim_reasons.setdefault(dim, []).append(reason)

        recurring = {d: c for d, c in counts.items() if c >= min_occurrences}
        if not recurring:
            return ""

        lines = [
            "LEARNED GUIDANCE FROM PAST RUNS ON THIS TOPIC "
            "(these checks have failed repeatedly before -- pay deliberate extra attention):"
        ]
        for dim, count in sorted(recurring.items(), key=lambda kv: -kv[1]):
            example_reason = dim_reasons.get(dim, ["(no reason recorded)"])[-1]
            lines.append(f"- '{dim}' has failed {count}x before. Most recent reason: {example_reason}")
        return "\n".join(lines)

    def all_records(self) -> list[dict[str, Any]]:
        return self._load()


MEMORY = MemoryStore()
