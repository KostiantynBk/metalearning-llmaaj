"""WP2 — Gemini API client for Gemma 4 (google-genai). Never raises for API errors."""
from __future__ import annotations

from .contracts import LLMClient, LLMResult, Settings


def make_client(settings: Settings) -> LLMClient:
    """FakeLLMClient() if settings.fake_llm else GemmaClient(settings)."""
    raise NotImplementedError


def preflight(client: LLMClient) -> tuple[bool, str]:
    """One tiny synthetic call (no task data). Returns (ok, detail incl. error_kind)."""
    raise NotImplementedError


class GemmaClient:
    """Implements contracts.LLMClient.

    - Model per role: settings.model_assessor / model_proposer / model_baseline.
    - Assessor & baseline: temperature 0, thinking_level "minimal"; proposer: thinking_level "high".
    - system_instruction, response_mime_type="application/json" and thinking_config are each
      probed: if the API rejects one (400 naming it), retry once without it and remember.
    - Client-side limiter: settings.rpm_limit requests/minute across threads; bounded concurrency.
    - 429 rate limit / 5xx: wait (Retry-After or exponential backoff with jitter) without
      consuming content attempts, bounded by settings.deadline_epoch -> error_kind "deadline".
      429 whose message indicates a per-day quota -> error_kind "daily_quota" immediately.
    - Disk cache in settings.cache_dir keyed by contracts.stable_hash(model, role, system, user,
      config); only ok results are cached; cached=True on hits.
    - Appends one JSON line per call to settings.usage_log (role, model, tokens, latency,
      error_kind, cached). Never log the API key or full prompts.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

    def generate(self, *, role: str, system: str, user: str,
                 json_expected: bool = True, max_output_tokens: int = 2048) -> LLMResult:
        raise NotImplementedError
