"""Central configuration for the lesson-generation agent."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT_DIR / "outputs"
DATA_DIR = ROOT_DIR / "data"
MEMORY_DB_PATH = DATA_DIR / "memory.json"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """Runtime settings, overridable via env vars or CLI flags."""

    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini"))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    # gemini-2.0-flash (the old default) was shut down June 2026 -- gemini-2.5-flash
    # is the current stable, free-tier-eligible general-purpose model (checked
    # against ai.google.dev/gemini-api/docs/models, Sep 2026).
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    # xAI Grok (api.x.ai) -- OpenAI-compatible chat completions API.
    # grok-2-latest (the old default) is long retired; grok-4.6 is xAI's current
    # flagship (checked against docs.x.ai/developers/models, Sep 2026).
    grok_api_key: str = field(default_factory=lambda: os.getenv("GROK_API_KEY", ""))
    grok_model: str = field(default_factory=lambda: os.getenv("GROK_MODEL", "grok-4.6"))
    # Groq (console.groq.com) -- a *different* provider from xAI's Grok, also
    # OpenAI-compatible but a different host/model family. Kept distinct on
    # purpose since it's easy to confuse the two by name.
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = field(default_factory=lambda: os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"))

    # When the active provider's rate limit doesn't clear within its retry
    # budget (see LLMClient.RateLimitExhausted), LLMClient first walks any
    # other models of the *same* provider/family (grok_fallback_models when
    # the active provider is grok, gemini_fallback_models when it's gemini),
    # then always falls through to gemini_fallback_models as the universal
    # last resort regardless of the primary provider. Each entry is a
    # separate model with its own free-tier quota, so one being exhausted
    # doesn't mean the rest are. Lists checked against each vendor's docs Sep
    # 2026; update them as models are added/retired.
    #
    # Ordered with the 2.x models first: they can fully disable "thinking"
    # (LLMClient._call_gemini sets thinkingBudget=0 for them), which the 3.x
    # models cannot do at all, making 2.x the safer bet for the small
    # structured judge responses that are easiest to truncate.
    # NOTE: gemini-2.5-flash-lite 404s as of Sep 2026 ("no longer available to
    # new users" per Google's own error body, which points at
    # gemini-3.5-flash-lite instead) -- omitted here for that reason. A model
    # 404ing no longer aborts the whole chain (complete() treats any failure
    # as "try the next model"), but there's no reason to keep a known-dead
    # entry in the default list.
    gemini_fallback_models: list[str] = field(
        default_factory=lambda: [
            m.strip()
            for m in os.getenv(
                "GEMINI_FALLBACK_MODELS",
                "gemini-2.5-flash,gemini-3.5-flash,gemini-3.5-flash-lite,"
                "gemini-3.1-flash-lite,gemini-3.8-flash,gemini-3.7-flash,gemini-3.6-flash",
            ).split(",")
            if m.strip()
        ]
    )
    grok_fallback_models: list[str] = field(
        default_factory=lambda: [
            m.strip()
            for m in os.getenv(
                "GROK_FALLBACK_MODELS",
                "grok-4.5,grok-4.3,grok-4.20-0309-non-reasoning",
            ).split(",")
            if m.strip()
        ]
    )

    # The loop always terminates: generate once, then at most this many
    # feedback-driven regenerations (so at most MAX_RETRIES + 1 generations total).
    max_retries: int = 2

    temperature_generate: float = 0.6
    temperature_evaluate: float = 0.0  # deterministic-as-possible judging

    # Output budget for the lesson-generation call. The prompt asks for several
    # dense sections (What it is / Why it matters / How it works / Recap) plus
    # a worked example across every required point, which routinely runs past
    # a few thousand tokens -- the previous unset default of 4096 (the
    # `complete()` fallback) was cutting lessons off mid-section.
    # Resolved in __post_init__ since the right ceiling depends on which
    # provider/model is active; MAX_GENERATION_TOKENS in the environment
    # always wins over the computed default.
    max_generation_tokens: int = field(default=0)

    def __post_init__(self) -> None:
        override = os.getenv("MAX_GENERATION_TOKENS")
        self.max_generation_tokens = int(override) if override else self._default_max_generation_tokens()

    def _default_max_generation_tokens(self) -> int:
        # Hard output-token ceilings per provider/model, from each vendor's
        # docs (checked Sep 2026) -- requesting above these has no effect, so
        # there's no point defaulting higher even though it wouldn't error.
        if self.provider == "gemini":
            return 8192  # gemini-2.5-flash and the 3.x flash family share this practical ceiling
        if self.provider == "groq":
            if self.groq_model.startswith("openai/gpt-oss"):
                return 16384  # ceiling is 65536; 16384 keeps lessons a practical length
            if self.groq_model.startswith("qwen/qwen3"):
                return 16384  # qwen3.8-27b's actual max-output-token ceiling
            return 8192  # unrecognized groq model -- conservative fallback
        return 8192  # grok (xAI) -- conservative default; not all Grok model ceilings are documented


SETTINGS = Settings()
