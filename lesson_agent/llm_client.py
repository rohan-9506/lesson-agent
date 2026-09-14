"""Provider-agnostic LLM client.

Supports Google Gemini and xAI Grok behind one interface so the rest of the
system (generation, evaluation) never talks to a vendor SDK directly. Swapping
providers is a config change, not a code change -- this is what the brief's
"any model" note asks for, and it's also what makes the system testable
without a live key (see tests/, which inject a FakeLLMClient with the same
interface).
"""
from __future__ import annotations

import json
import re
import sys
import time
from typing import Any, Optional

import requests

from .config import SETTINGS


class LLMError(RuntimeError):
    """Raised when a provider call fails or returns something we can't use."""


class RateLimitExhausted(LLMError):
    """Raised when a provider's 429s didn't clear within the retry budget.

    Distinct from LLMError so `complete()` can tell "this model/provider is
    out of quota, try the next one" apart from a real failure (bad request,
    auth, malformed response) that a fallback attempt wouldn't fix either.
    """


class LLMClient:
    """Thin wrapper exposing a single `complete()` method regardless of vendor."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or SETTINGS.provider).lower().strip()
        if self.provider not in {"gemini", "grok", "groq"}:
            raise LLMError(f"Unsupported provider: {self.provider!r} (use 'gemini', 'grok', or 'groq')")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.6,
        json_mode: bool = False,
        max_tokens: int = 4096,
    ) -> str:
        """Return the raw text completion. If json_mode, caller expects valid JSON text.

        Tries the configured provider/model first. On failure -- rate limit
        exhausted, or anything else (a retired/404 model id, a malformed
        response, ...) -- walks any other configured models for that same
        provider (e.g. grok-4.6 -> grok-4.5 -> grok-4.3), then falls through
        to the Gemini fallback chain as the universal last resort. Models get
        retired or renamed without notice (see .env.example), so treating any
        single model's failure as fatal would make the whole pipeline as
        fragile as its least-maintained model id; only once every configured
        model has failed does the last error propagate. See _attempt_chain().
        """
        last_exc: Optional[LLMError] = None
        for provider, model, is_fallback in self._attempt_chain():
            try:
                text = self._dispatch(provider, model, system_prompt, user_prompt, temperature, json_mode, max_tokens)
            except LLMError as exc:
                last_exc = exc
                reason = str(exc).splitlines()[0][:150]
                print(f"[llm] {provider}/{model} failed ({reason}), trying next fallback...", file=sys.stderr)
                continue
            if is_fallback:
                print(f"[llm] recovered using fallback {provider}/{model}", file=sys.stderr)
            return text
        assert last_exc is not None  # _attempt_chain() always yields at least the primary
        raise last_exc

    def _attempt_chain(self) -> list[tuple[str, str, bool]]:
        """Ordered (provider, model, is_fallback) attempts.

        First exhausts every model configured for the active provider's own
        family (e.g. grok-4.6 -> grok-4.5 -> grok-4.3), then -- unless the
        provider already *is* gemini, whose own family was just exhausted --
        falls through to the Gemini fallback chain as the universal last
        resort.
        """
        own_models = self._own_family_models()
        chain = [(self.provider, own_models[0], False)]
        chain += [(self.provider, model, True) for model in own_models[1:]]
        if self.provider != "gemini":
            chain += [("gemini", model, True) for model in SETTINGS.gemini_fallback_models]
        return chain

    def _own_family_models(self) -> list[str]:
        """Primary model for the active provider, followed by its configured siblings."""
        if self.provider == "gemini":
            primary, siblings = SETTINGS.gemini_model, SETTINGS.gemini_fallback_models
        elif self.provider == "grok":
            primary, siblings = SETTINGS.grok_model, SETTINGS.grok_fallback_models
        else:
            return [SETTINGS.groq_model]  # groq has no configured sibling models
        return [primary] + [m for m in siblings if m != primary]

    def _dispatch(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        json_mode: bool,
        max_tokens: int,
    ) -> str:
        # A 429 on free-tier tiers is almost always the whole TPM window being
        # blown, not a one-off blip that a retry would clear -- and with a
        # fallback chain of other models available, there's no reason to wait
        # at all. 0 retries means _post_with_retry makes exactly one request
        # and, on a 429, returns immediately (no sleep) so complete() can move
        # to the next model in the chain right away.
        retries = 0
        if provider == "gemini":
            return self._call_gemini(
                system_prompt, user_prompt, temperature, json_mode, max_tokens, model=model, max_retries=retries
            )
        if provider == "groq":
            return self._call_groq(
                system_prompt, user_prompt, temperature, json_mode, max_tokens, model=model, max_retries=retries
            )
        return self._call_grok(
            system_prompt, user_prompt, temperature, json_mode, max_tokens, model=model, max_retries=retries
        )

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Call the model and parse strict JSON out of the response.

        Models occasionally wrap JSON in markdown fences or add stray prose
        despite instructions; we defensively extract the first {...} block
        before parsing so a formatting slip doesn't crash the whole pipeline.
        """
        raw = self.complete(system_prompt, user_prompt, temperature=temperature, json_mode=True, max_tokens=max_tokens)
        return self._extract_json(raw)

    # ------------------------------------------------------------------ #
    # Provider implementations
    # ------------------------------------------------------------------ #
    def _call_gemini(
        self, system_prompt, user_prompt, temperature, json_mode, max_tokens, model: Optional[str] = None, max_retries: int = 10
    ) -> str:
        if not SETTINGS.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set (see .env.example).")

        model = model or SETTINGS.gemini_model
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={SETTINGS.gemini_api_key}"
        )

        # Gemini 2.0/2.5 models default to "thinking" (internal reasoning
        # tokens that eat into maxOutputTokens before the visible response is
        # written) but can turn it off via thinkingBudget=0. Gemini 3.x
        # cannot disable thinking at all (checked against
        # ai.google.dev/gemini-api/docs/generate-content/thinking, Sep 2026)
        # -- so for those models, pad the budget instead, or a small
        # structured response (like a judge's {"passed": ..., "reason": ...})
        # can get cut off mid-JSON before it ever reaches the visible part,
        # the same failure mode this project already works around for Groq's
        # reasoning models.
        is_gemini_3 = model.startswith("gemini-3")
        if is_gemini_3:
            max_tokens += 3000

        generation_config: dict[str, Any] = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if not is_gemini_3:
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}

        payload: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": generation_config,
        }
        if json_mode:
            payload["generationConfig"]["response_mime_type"] = "application/json"

        resp = self._post_with_retry(url, {}, payload, max_retries=max_retries)
        if resp.status_code == 429:
            raise RateLimitExhausted(f"Gemini ({model}) rate limit didn't clear: {resp.text[:500]}")
        if resp.status_code != 200:
            raise LLMError(f"Gemini API error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Gemini response shape: {data}") from exc

    def _call_grok(
        self, system_prompt, user_prompt, temperature, json_mode, max_tokens, model: Optional[str] = None, max_retries: int = 10
    ) -> str:
        if not SETTINGS.grok_api_key:
            raise LLMError("GROK_API_KEY is not set (see .env.example).")

        model = model or SETTINGS.grok_model
        url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {SETTINGS.grok_api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        resp = self._post_with_retry(url, headers, payload, max_retries=max_retries)
        if resp.status_code == 429:
            raise RateLimitExhausted(f"Grok ({model}) rate limit didn't clear: {resp.text[:500]}")
        if resp.status_code != 200:
            raise LLMError(f"Grok API error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Grok response shape: {data}") from exc

    def _call_groq(
        self, system_prompt, user_prompt, temperature, json_mode, max_tokens, model: Optional[str] = None, max_retries: int = 10
    ) -> str:
        if not SETTINGS.groq_api_key:
            raise LLMError("GROQ_API_KEY is not set (see .env.example).")

        model = model or SETTINGS.groq_model
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {SETTINGS.groq_api_key}",
            "Content-Type": "application/json",
        }

        # Groq's reasoning models use internal "thinking" tokens that eat into
        # max_tokens before the visible output is produced -- if unchecked, a
        # lesson (or judge verdict) can come back short or empty because the
        # model spent most/all of its budget "thinking" before writing anything
        # visible. When combined with json_mode, Groq's server-side JSON
        # validator additionally rejects the response with 400
        # json_validate_failed if the output is empty or truncated mid-JSON.
        # Workaround: disable json_mode for these models and let _extract_json
        # handle parsing -- it already deals with markdown fences and plain
        # JSON text -- and dial reasoning down as far as each family allows.
        # Different model families expose different reasoning_effort values
        # (checked against console.groq.com/docs/reasoning): gpt-oss only
        # supports low/medium/high, so "low" is its floor; the Qwen3 family
        # additionally supports "none", which fully disables thinking mode --
        # worth using here since lesson-writing and pass/fail judging don't
        # need chain-of-thought, only the visible output.
        if model.startswith("openai/gpt-oss"):
            reasoning_effort: Optional[str] = "low"
        elif model.startswith("qwen/qwen3"):
            reasoning_effort = "none"
        else:
            reasoning_effort = None
        is_reasoning_model = reasoning_effort is not None
        use_json_mode = json_mode and not is_reasoning_model

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if use_json_mode:
            payload["response_format"] = {"type": "json_object"}
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort


        delay = 5.0
        resp = self._post_with_retry(url, headers, payload, max_retries=max_retries)
        for attempt in range(6):
            # 400 json_validate_failed: reasoning model was truncated mid-JSON -- retry.
            if resp.status_code == 400 and "json_validate_failed" in resp.text:
                print(
                    f"[groq] json_validate_failed (attempt {attempt + 1}/6), retrying immediately...",
                    file=sys.stderr,
                )
                resp = self._post_with_retry(url, headers, payload, max_retries=max_retries)
                continue
            if resp.status_code != 200:
                break
            content = ""
            try:
                content = resp.json()["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, ValueError):
                pass
            # Empty content (200 but blank): reasoning model returned nothing —
            # thinking tokens consumed the full budget, or a soft rate-limit
            # manifested as a silent empty response. Back off and retry.
            if not content.strip():
                if attempt < 5:
                    print(
                        f"[groq] empty response (attempt {attempt + 1}/6), "
                        f"waiting {delay:.1f}s before retry...",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                    delay = min(delay * 1.5, 30.0)
                    resp = self._post_with_retry(url, headers, payload, max_retries=max_retries)
                    continue
                else:
                    raise RateLimitExhausted(
                        f"Groq ({model}) returned empty content after 6 retries -- "
                        "the free-tier TPM budget is likely exhausted."
                    )
            break

        if resp.status_code == 429:
            raise RateLimitExhausted(f"Groq ({model}) rate limit didn't clear: {resp.text[:500]}")
        if resp.status_code != 200:
            raise LLMError(f"Groq API error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected Groq response shape: {data}") from exc


    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _post_with_retry(
        url: str, headers: dict[str, str], payload: dict[str, Any], *, max_retries: int = 10
    ) -> requests.Response:
        """POST, retrying on 429 with the provider's own suggested backoff.

        Free-tier token-per-minute budgets are small enough that a single
        multi-call pipeline run (1 generate + up to 4 rubric judges, per
        attempt) can trip them even with no other traffic on the key. Rather
        than surface that as a hard failure, honor `Retry-After` (or the
        "try again in Xs" hint providers put in the 429 body) and retry.
        A minimum 5s wait is enforced regardless of what the header says,
        because "try again in 1.5s" often understates the true window reset
        time on free-tier accounts with multiple parallel quotas.
        """
        delay = 5.0
        resp = requests.post(url, headers=headers, json=payload, timeout=90)
        for attempt in range(max_retries):
            if resp.status_code != 429:
                return resp
            suggested = LLMClient._parse_retry_after(resp) or delay
            wait = min(max(suggested, 5.0), 65.0)  # at least 5s, but don't trust a runaway header value
            print(
                f"[{url.split('/')[2]}] 429 rate limited (attempt {attempt + 1}/{max_retries}), "
                f"waiting {wait:.1f}s before retry...",
                file=sys.stderr,
            )
            time.sleep(wait)
            delay = min(delay * 1.5, 60.0)
            resp = requests.post(url, headers=headers, json=payload, timeout=90)
        return resp


    @staticmethod
    def _parse_retry_after(resp: requests.Response) -> Optional[float]:
        header = resp.headers.get("Retry-After")
        if header:
            try:
                return float(header)
            except ValueError:
                pass
        # Groq (and other OpenAI-compatible providers) report exactly how long until
        # the tripped quota refills via these headers, e.g. "49.86s" or "1m2.3s" --
        # far more reliable than the generic Retry-After/body-text guesses above,
        # which is what was causing repeated retries into a still-exhausted budget.
        for header_name in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
            value = resp.headers.get(header_name)
            if value:
                parsed = LLMClient._parse_groq_duration(value)
                if parsed is not None:
                    return parsed
        match = re.search(r"try again in ([\d.]+)s", resp.text)
        if match:
            return float(match.group(1))
        return None

    @staticmethod
    def _parse_groq_duration(value: str) -> Optional[float]:
        """Parse Groq's rate-limit reset duration strings, e.g. '49.86s', '1m2.3s', '57m36s'."""
        match = re.match(r"^(?:(\d+)m)?(\d+(?:\.\d+)?)s$", value.strip())
        if not match:
            return None
        minutes = float(match.group(1)) if match.group(1) else 0.0
        seconds = float(match.group(2))
        return minutes * 60 + seconds

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        text = raw.strip()
        # Strip ```json ... ``` or ``` ... ``` fences if present.
        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fall back to grabbing the first balanced-looking {...} span.
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError as exc:
                    raise LLMError(f"Could not parse JSON from model output: {text[:500]}") from exc
            raise LLMError(f"Could not parse JSON from model output: {text[:500]}")
