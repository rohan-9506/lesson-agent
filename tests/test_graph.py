"""Tests for the generate -> evaluate -> regenerate LangGraph loop, using
FakeLLMClient so these run with zero API key and zero network calls."""
import json
from pathlib import Path

import pytest

from lesson_agent.graph import run_pipeline
from lesson_agent.memory import MemoryStore
from lesson_agent.output_writer import write_outputs
from tests.fakes import FakeLLMClient


@pytest.fixture(autouse=True)
def isolated_memory(tmp_path, monkeypatch):
    """Point the module-level MEMORY store at a scratch file per test so runs
    don't leak learned guidance between tests (or into the real data/memory.json)."""
    scratch_path = tmp_path / "memory.json"
    import lesson_agent.memory as memory_mod

    fresh_store = MemoryStore(path=scratch_path)
    monkeypatch.setattr(memory_mod, "MEMORY", fresh_store)
    monkeypatch.setattr("lesson_agent.graph.MEMORY", fresh_store)
    yield fresh_store


def test_deliberate_jargon_error_is_caught_then_fixed_on_retry():
    """This mirrors the assessment's required demo: inject a deliberate
    defect, show the evaluator catch it by name, then show the regenerate
    step produce a passing lesson."""
    fake = FakeLLMClient(judge_script={})  # all LLM-judge checks default to pass
    final_state = run_pipeline("rag", llm=fake, max_retries=2, inject_error="jargon")

    assert final_state["status"] == "passed"
    assert final_state["attempt"] == 2  # failed once, fixed on the very next attempt

    first_attempt_log = final_state["rejection_log"][0]
    assert first_attempt_log["passed"] is False
    assert "no_unexplained_jargon" in first_attempt_log["failed"]
    assert "context window" in first_attempt_log["reasons"]["no_unexplained_jargon"]

    second_attempt_log = final_state["rejection_log"][1]
    assert second_attempt_log["passed"] is True


def test_deliberate_wrong_fact_is_caught_by_grounded_judge_check():
    fake = FakeLLMClient(
        judge_script={
            # Attempt 1: the corrupted "wrong_fact" text should make the grounded
            # judge fail. Attempt 2 (uncorrupted) passes.
            "accurate_and_grounded": [
                {"passed": False, "reason": "Claims RAG permanently fine-tunes the model on every query; contradicts verified facts."},
                {"passed": True, "reason": "Consistent with verified facts."},
            ]
        }
    )
    final_state = run_pipeline("rag", llm=fake, max_retries=2, inject_error="wrong_fact")

    assert final_state["attempt"] == 2
    assert final_state["status"] == "passed"
    assert "accurate_and_grounded" in final_state["rejection_log"][0]["failed"]


def test_loop_always_terminates_via_force_ship_when_retries_exhausted():
    """If a dimension NEVER passes, the loop must still terminate (max_retries
    is a hard structural bound, not best-effort)."""
    always_fail = {"passed": False, "reason": "persistently missing a required key point"}
    fake = FakeLLMClient(
        judge_script={
            "covers_key_points": [always_fail, always_fail, always_fail, always_fail],
        }
    )
    final_state = run_pipeline("rag", llm=fake, max_retries=1)

    # max_retries=1 => attempt 1 (fail) + attempt 2 (fail, retries exhausted) => force ship
    assert final_state["attempt"] == 2
    assert final_state["status"] == "force_shipped"
    assert len(final_state["rejection_log"]) == 2
    assert all(not entry["passed"] for entry in final_state["rejection_log"])


def test_clean_generation_passes_on_first_attempt():
    fake = FakeLLMClient()
    final_state = run_pipeline("rag", llm=fake, max_retries=2)

    assert final_state["attempt"] == 1
    assert final_state["status"] == "passed"
    assert fake.generation_calls == 1


def test_output_writer_produces_lesson_and_rejection_log(tmp_path, monkeypatch):
    import lesson_agent.output_writer as ow

    monkeypatch.setattr(ow, "OUTPUTS_DIR", tmp_path)

    fake = FakeLLMClient(judge_script={})
    final_state = run_pipeline("rag", llm=fake, max_retries=2, inject_error="jargon")
    lesson_path, log_path = write_outputs(final_state)

    assert lesson_path.exists()
    assert log_path.exists()
    assert "RAG" in lesson_path.read_text(encoding="utf-8")
    log_text = log_path.read_text(encoding="utf-8")
    assert "no_unexplained_jargon" in log_text
    assert "FAIL" in log_text and "PASS" in log_text
