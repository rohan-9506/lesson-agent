"""Renders the final graph state into the two deliverables: the lesson
document and the rejection log."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from .config import OUTPUTS_DIR
from .state import LessonState


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def write_outputs(state: LessonState) -> tuple[Path, Path]:
    slug = _slug(state["topic_label"])
    run_dir = OUTPUTS_DIR / slug
    run_dir.mkdir(parents=True, exist_ok=True)

    lesson_path = run_dir / "lesson.md"
    lesson_path.write_text(state["lesson_text"], encoding="utf-8")

    log_path = run_dir / "rejection_log.md"
    log_path.write_text(_render_rejection_log(state), encoding="utf-8")

    raw_path = run_dir / "run_state.json"
    raw_path.write_text(
        json.dumps(
            {
                "topic_label": state["topic_label"],
                "status": state["status"],
                "final_attempt": state["attempt"],
                "rejection_log": state["rejection_log"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return lesson_path, log_path


# --------------------------------------------------------------------------- #
# Optional PDF export
# --------------------------------------------------------------------------- #

# Common install locations for a Chromium-family browser, checked in order.
# Used as a fallback when the browser isn't on PATH (e.g. Edge on Windows).
_BROWSER_CANDIDATES = [
    "msedge",
    "chrome",
    "google-chrome",
    "chromium",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
]


def _find_browser() -> Optional[str]:
    for candidate in _BROWSER_CANDIDATES:
        if shutil.which(candidate):
            return shutil.which(candidate)
        if Path(candidate).exists():
            return candidate
    return None


def export_pdf(lesson_path: Path) -> Optional[Path]:
    """Render lesson.md to lesson.pdf next to it, via a headless Chromium-family
    browser's print-to-pdf. Best-effort: returns None (and prints why) instead
    of failing the run if no supported browser is installed, since this is a
    convenience export, not a deliverable the pipeline depends on.
    """
    import markdown

    browser = _find_browser()
    if not browser:
        print("PDF export skipped: no Chrome/Edge/Chromium found on this machine.")
        return None

    html_path = lesson_path.with_suffix(".html").resolve()
    pdf_path = lesson_path.with_suffix(".pdf").resolve()

    body = markdown.markdown(lesson_path.read_text(encoding="utf-8"), extensions=["fenced_code", "tables"])
    html_path.write_text(
        "<html><head><meta charset='utf-8'>"
        "<style>"
        "body{font-family:Arial,sans-serif;max-width:760px;margin:40px auto;"
        "font-size:18px;line-height:1.7;color:#1a1a1a}"
        "h1{font-size:1.8em}h2{font-size:1.4em;margin-top:1.4em}h3{font-size:1.15em}"
        "li{margin-bottom:0.4em}"
        "code{background:#f2f2f2;padding:1px 4px;font-size:0.95em}"
        "pre{background:#f2f2f2;padding:8px;overflow-x:auto}"
        "</style></head>"
        f"<body>{body}</body></html>",
        encoding="utf-8",
    )

    # Chromium's headless print-to-pdf needs an absolute path or a file:// URI --
    # a bare relative path gets parsed as a URL with the first segment as host.
    # A scratch --user-data-dir avoids poking the browser's real profile
    # (sync/SmartScreen network calls that only add noise and latency here).
    # Extra flags required for reliable headless operation on macOS:
    #   --no-sandbox / --disable-dev-shm-usage: avoid permission + shared-memory
    #     issues in sandboxed/CI environments.
    #   --run-all-compositor-stages-before-draw + --virtual-time-budget: tell the
    #     headless renderer to treat the page as fully loaded immediately, so it
    #     doesn't hang waiting for fonts or timers before printing.
    #
    # We don't wait for the process to exit (subprocess.run/.wait()): recent
    # Chrome builds (observed on 152.x, both --headless and --headless=new)
    # write a complete --print-to-pdf file within a few seconds but then hang
    # indefinitely instead of exiting, so blocking on exit reports a false
    # "timed out" failure and throws away a perfectly good PDF. Instead we
    # poll for the output file to appear and stop growing, then kill the
    # browser ourselves -- the PDF is already flushed to disk by that point.
    with tempfile.TemporaryDirectory(prefix="lesson-agent-pdf-") as profile_dir:
        proc = subprocess.Popen(
            [
                browser,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
                "--run-all-compositor-stages-before-draw",
                "--virtual-time-budget=5000",
                "--print-to-pdf-no-header",
                f"--user-data-dir={profile_dir}",
                f"--print-to-pdf={pdf_path}",
                html_path.as_uri(),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 60.0
            last_size = -1
            stable_since: Optional[float] = None
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break  # process exited on its own -- fall through to the checks below
                if pdf_path.exists():
                    size = pdf_path.stat().st_size
                    if size > 0 and size == last_size:
                        # Unchanged for two consecutive checks: Chrome has
                        # finished flushing the file even though the process
                        # itself is still sitting there.
                        stable_since = stable_since or time.monotonic()
                        if time.monotonic() - stable_since >= 1.0:
                            break
                    else:
                        stable_since = None
                    last_size = size
                time.sleep(0.5)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)

    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        stderr = proc.stderr.read().decode(errors="replace")[:500] if proc.stderr else ""
        print(f"PDF export failed: Chrome never produced a PDF within 60s. {stderr}")
        return None
    return pdf_path



def _render_rejection_log(state: LessonState) -> str:
    lines = [f"# Rejection Log -- {state['topic_label']}", ""]
    status_label = {
        "passed": "PASSED",
        "force_shipped": "FORCE-SHIPPED (max retries exhausted, still failing)",
    }.get(state["status"], state["status"].upper())
    lines.append(f"**Final status:** {status_label} after {state['attempt']} attempt(s).\n")

    for entry in state["rejection_log"]:
        lines.append(f"## Attempt {entry['attempt']}")
        if entry["passed"]:
            lines.append("- Result: **PASS** -- all six checkpoints passed.\n")
            continue
        lines.append(f"- Result: **FAIL** -- failed checkpoint(s): {', '.join(entry['failed'])}")
        for dim, reason in entry["reasons"].items():
            lines.append(f"  - **{dim}**: {reason}")
        lines.append("")

    # What changed between attempts, in plain terms.
    if len(state["rejection_log"]) > 1:
        lines.append("## What changed on retry")
        for i in range(1, len(state["rejection_log"])):
            prev_fail = set(state["rejection_log"][i - 1]["failed"])
            lines.append(
                f"- Attempt {i} -> {i+1}: regeneration was given the failure reasons for "
                f"{', '.join(sorted(prev_fail))} and told to fix them directly, not reword around them."
            )

    return "\n".join(lines)
