"""Gemma transport, quota pacing and exact-request cache. No prompts are logged."""
from __future__ import annotations

import json
import math
import os
import random
import re
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict, replace
from pathlib import Path

from .contracts import LLMClient, LLMResult, Settings, stable_hash


def make_client(settings: Settings) -> LLMClient:
    """FakeLLMClient() if settings.fake_llm else GemmaClient(settings)."""
    if settings.fake_llm:
        from .fake_client import FakeLLMClient
        return FakeLLMClient()
    return GemmaClient(settings)


def preflight(client: LLMClient) -> tuple[bool, str]:
    """One tiny synthetic call; never includes task data."""
    result = client.generate(role="assessor", system="Return only JSON.",
                             user='Return {"ok":true}.', max_output_tokens=32)
    return result.ok, "ok" if result.ok else (result.error_kind or "other")


class GemmaClient:
    """Thread-safe client with per-model feature discovery and sliding-window quotas.

    Test transport signature: transport(model=..., contents=..., config=dict).
    It returns an SDK-shaped response (text, usage_metadata), a dict with those
    keys, or LLMResult, and may raise exceptions with code/message attributes.
    Quotas are conservative and shared across models within this client.
    """

    def __init__(self, settings: Settings, *, transport=None):
        self.settings = settings
        self._transport = transport
        self._local = threading.local()
        self._lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max(1, settings.max_concurrency))
        self._window = deque()
        self._flags = {}

    @property
    def capability_flags(self):
        with self._lock:
            return {model: dict(flags) for model, flags in self._flags.items()}

    def _model_flags(self, model):
        with self._lock:
            return dict(self._flags.setdefault(model, {
                "system_instruction": True, "response_mime_type": True,
                "thinking_config": True}))

    def _sleep(self, delay):
        remaining = self.settings.seconds_left()
        if remaining <= 0:
            return False
        time.sleep(min(delay, remaining))
        return self.settings.seconds_left() > 0

    def _reserve(self, tokens):
        """Reserve a mutable token entry, replaced by observed input usage later."""
        while self.settings.seconds_left() > 0:
            with self._lock:
                now = time.monotonic()
                while self._window and now - self._window[0][0] >= 60:
                    self._window.popleft()
                rpm = max(1, self.settings.rpm_limit)
                tpm = max(1, self.settings.tpm_limit)
                fits = not self._window or sum(e[1] for e in self._window) + tokens <= tpm
                if len(self._window) < rpm and fits:
                    entry = [now, tokens]
                    self._window.append(entry)
                    return entry
                delay = max(0.001, 60 - (now - self._window[0][0]))
            if not self._sleep(delay):
                return None
        return None

    def _call(self, model, contents, config):
        if self._transport is not None:
            return self._transport(model=model, contents=contents, config=config)
        # Lazy imports: neither importing this module nor fake tests need the SDK.
        from google import genai
        from google.genai import types
        if not hasattr(self._local, "client"):
            self._local.client = genai.Client(http_options=types.HttpOptions(timeout=120000))
        cfg = dict(config)
        if "thinking_config" in cfg:
            cfg["thinking_config"] = types.ThinkingConfig(**cfg["thinking_config"])
        return self._local.client.models.generate_content(
            model=model, contents=contents, config=types.GenerateContentConfig(**cfg))

    @staticmethod
    def _field(obj, key, default=None):
        return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)

    @staticmethod
    def _classify(exc):
        code = getattr(exc, "code", None)
        try:
            code = int(code)
        except (TypeError, ValueError):
            pass
        message = str(getattr(exc, "message", str(exc)))
        low = message.lower()
        if code == 429:
            kind = "daily_quota" if re.search(r"per\s*day|daily", low) else "rate_limit"
        elif code in (500, 502, 503, 504):
            kind = "unavailable"
        elif code in (401, 403):
            kind = "auth"
        elif code == 400:
            kind = "bad_request"
        elif isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower() or "timed out" in low:
            kind = "timeout"
        else:
            kind = "other"
        return kind, message

    @staticmethod
    def _rejected_feature(message, flags):
        low = message.lower()
        features = (("system_instruction", ("system instruction", "system_instruction", "systeminstruction", "developer instruction")),
                    ("response_mime_type", ("response_mime_type", "response mime type", "responsemimetype", "json mode")),
                    ("thinking_config", ("thinking",)))
        for feature, names in features:
            if flags[feature] and any(name in low for name in names):
                return feature
        return None

    @staticmethod
    def _retry_delay(message, attempt):
        match = re.search(r'retry[_ -]?delay[\s\"\':=]*([0-9]+(?:\.[0-9]+)?)\s*s?', message, re.I)
        if not match:
            match = re.search(r'retry[_ -]?delay[\s\"\':={]*[\"\']?seconds[\"\'\s:]+([0-9]+(?:\.[0-9]+)?)', message, re.I)
        return float(match.group(1)) if match else min(60.0, 2.0 ** min(attempt + 1, 6) + random.uniform(0, 0.5))

    def _cache_read(self, key):
        if not self.settings.cache_dir:
            return None
        try:
            data = json.loads((Path(self.settings.cache_dir) / (key + ".json")).read_text(encoding="utf-8"))
            if data.get("ok") is True and isinstance(data.get("text"), str):
                return replace(LLMResult(**data), cached=True)
        except (OSError, ValueError, TypeError):
            pass
        return None

    def _cache_write(self, key, result):
        if not self.settings.cache_dir:
            return
        tmp = None
        try:
            folder = Path(self.settings.cache_dir)
            folder.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder, delete=False) as f:
                tmp = f.name
                json.dump(asdict(result), f, ensure_ascii=False)
            os.replace(tmp, folder / (key + ".json"))
        except OSError:
            pass  # A broken optional cache must not invalidate an API result.
        finally:
            if tmp and os.path.exists(tmp):
                os.unlink(tmp)

    def _log(self, role, model, result):
        if not self.settings.usage_log:
            return
        record = {k: getattr(result, k) for k in (
            "ok", "error_kind", "input_tokens", "output_tokens", "latency_s", "cached")}
        record.update(role=role, model=model)
        try:
            with self._log_lock:
                path = Path(self.settings.usage_log)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
        except OSError:
            pass

    def generate(self, *, role: str, system: str, user: str,
                 json_expected: bool = True, max_output_tokens: int = 2048) -> LLMResult:
        start = time.monotonic()
        model = getattr(self.settings, "model_" + role, None)
        if model is None:
            return LLMResult(ok=False, error_kind="bad_request", error="Unknown role")
        # Include initial flags in cache identity; store fallback result under both
        # initial and final flags so a fresh process can replay without discovery.
        flags = self._model_flags(model)
        def key_for(f):
            return stable_hash(model, role, system, user, json_expected, max_output_tokens, f)
        original_key = key_for(flags)
        cached = self._cache_read(original_key)
        if cached:
            cached.latency_s = time.monotonic() - start
            self._log(role, model, cached)
            return cached
        while self.settings.seconds_left() > 0:
            if self._slots.acquire(timeout=min(0.1, self.settings.seconds_left())):
                break
        else:
            result = LLMResult(ok=False, error_kind="deadline")
            self._log(role, model, result)
            return result
        try:
            result = self._generate_locked(model, role, system, user, json_expected, max_output_tokens)
        finally:
            self._slots.release()
        result.latency_s = time.monotonic() - start
        if result.ok:
            self._cache_write(original_key, result)
            self._cache_write(key_for(self._model_flags(model)), result)
        self._log(role, model, result)
        return result

    def _generate_locked(self, model, role, system, user, json_expected, max_output_tokens):
        attempt = 0
        total_input = total_output = 0
        while self.settings.seconds_left() > 0:
            flags = self._model_flags(model)
            config = {"temperature": 0.0, "max_output_tokens": max_output_tokens}
            contents = user
            if system:
                if flags["system_instruction"]:
                    config["system_instruction"] = system
                else:
                    contents = "SYSTEM INSTRUCTIONS:\n" + system + "\n\nINPUT:\n" + user
            if json_expected and flags["response_mime_type"]:
                config["response_mime_type"] = "application/json"
            if flags["thinking_config"]:
                config["thinking_config"] = {"thinking_level": "high" if role == "proposer" else "minimal"}
            estimate = math.ceil(len(system + contents) / 2.5) if "system_instruction" in config else math.ceil(len(contents) / 2.5)
            entry = self._reserve(estimate)
            if entry is None:
                break
            try:
                response = self._call(model, contents, config)
                if isinstance(response, LLMResult):
                    result = replace(response)
                else:
                    usage = self._field(response, "usage_metadata")
                    inp = self._field(usage, "prompt_token_count")
                    out = self._field(usage, "candidates_token_count")
                    text = self._field(response, "text", "") or ""
                    result = LLMResult(ok=bool(text), text=text, input_tokens=inp or 0,
                                       output_tokens=out or 0, error_kind="" if text else "other",
                                       error="" if text else "Empty model response")
                if result.input_tokens:
                    with self._lock:
                        entry[1] = result.input_tokens
                total_input += result.input_tokens
                total_output += result.output_tokens
                if result.ok or result.error_kind not in ("rate_limit", "unavailable", "timeout"):
                    result.input_tokens, result.output_tokens = total_input, total_output
                    return result
                kind, message = result.error_kind, result.error
            except Exception as exc:
                kind, message = self._classify(exc)
            if kind == "bad_request":
                feature = self._rejected_feature(message, flags)
                if feature:
                    with self._lock:
                        self._flags[model][feature] = False
                    continue
            if kind not in ("rate_limit", "unavailable", "timeout"):
                # Do not propagate SDK exception messages, which may contain URLs,
                # credentials or snippets of the submitted prompt.
                return LLMResult(ok=False, error_kind=kind, error="Gemma request failed: " + kind,
                                 input_tokens=total_input, output_tokens=total_output)
            if not self._sleep(self._retry_delay(message, attempt)):
                break
            attempt += 1
        return LLMResult(ok=False, error_kind="deadline", input_tokens=total_input, output_tokens=total_output)
