"""All transports are synthetic. No SDK, API key or network needed."""
import builtins
import importlib
import importlib.util
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from rulejudge.contracts import LLMResult, Settings
from rulejudge.fake_client import FakeLLMClient
from rulejudge.gemma_client import GemmaClient, make_client, preflight


class APIError(Exception):
    def __init__(self, code, message):
        self.code, self.message = code, message
        super().__init__(message)


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class TestGemmaClient(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.patches = [patch("rulejudge.gemma_client.time.time", self.clock.time),
                        patch("rulejudge.gemma_client.time.monotonic", self.clock.time),
                        patch("rulejudge.gemma_client.time.sleep", self.clock.sleep),
                        patch("rulejudge.gemma_client.random.uniform", return_value=0)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.settings = Settings(deadline_epoch=1600)
        self.calls = []

    def client(self, responses):
        pending = iter(responses)
        def transport(**kw):
            self.calls.append(kw)
            out = next(pending)
            if isinstance(out, Exception):
                raise out
            return out
        return GemmaClient(self.settings, transport=transport)

    def call(self, client, **kwargs):
        args = dict(role="assessor", system="system", user="user")
        args.update(kwargs)
        return client.generate(**args)

    def test_fake_and_preflight(self):
        self.assertIsInstance(make_client(Settings(fake_llm=True)), FakeLLMClient)
        self.assertIsInstance(make_client(Settings()), GemmaClient)
        self.assertEqual(preflight(FakeLLMClient()), (True, "ok"))
        self.assertEqual(preflight(FakeLLMClient(responder=lambda *a: LLMResult(False, error_kind="auth"))),
                         (False, "auth"))

    def test_import_without_google_sdk(self):
        original = builtins.__import__
        def no_google(name, *args, **kwargs):
            if name == "google" or name.startswith("google."):
                raise ImportError("SDK deliberately unavailable")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", no_google):
            module = importlib.reload(importlib.import_module("rulejudge.gemma_client"))
            result = self.call(module.GemmaClient(self.settings, transport=lambda **kw: LLMResult(True, text="{}")))
        self.assertTrue(result.ok)

    def test_smoke_env_loader_uses_mock_file_only(self):
        spec = importlib.util.spec_from_file_location("smoke_gemma_test", Path(__file__).resolve().parents[1] / "scripts" / "smoke_gemma.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for assignment in ('GEMINI_API_KEY="synthetic"', "export GEMINI_API_KEY='synthetic'",
                           'GEMINI_API_KEY=synthetic # comment'):
            with self.subTest(assignment=assignment), patch.dict(os.environ, {}, clear=True):
                with patch.object(Path, "read_text", return_value="# synthetic only\n" + assignment):
                    module._load_api_key(Path("not-a-real-repo"))
                self.assertEqual(os.environ["GEMINI_API_KEY"], "synthetic")
        with patch.dict(os.environ, {"GEMINI_API_KEY": "existing"}, clear=True):
            with patch.object(Path, "read_text", side_effect=AssertionError("must not read")):
                module._load_api_key(Path("not-a-real-repo"))
            self.assertEqual(os.environ["GEMINI_API_KEY"], "existing")

    def test_json_mode_wording_and_retry_delay_object(self):
        client = self.client([APIError(400, "JSON mode is not enabled for this model"),
                              APIError(429, 'retryDelay: {"seconds": 4}'), LLMResult(True, text="{}")])
        self.assertTrue(self.call(client).ok)
        self.assertNotIn("response_mime_type", self.calls[-1]["config"])
        self.assertEqual(self.clock.sleeps, [4])

    def test_rate_limit_then_success_honors_delay(self):
        client = self.client([APIError(429, 'retryDelay: "3s"'), LLMResult(True, text="{}")])
        self.assertTrue(self.call(client).ok)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.clock.sleeps, [3])

    def test_daily_quota_and_auth_never_retry(self):
        for code, message, kind in [(429, "quota PerDay", "daily_quota"),
                                    (429, "quota per day", "daily_quota"),
                                    (429, "Daily limit", "daily_quota"),
                                    (401, "key?secret", "auth"), (403, "forbidden", "auth")]:
            with self.subTest(message=message):
                before = len(self.calls)
                result = self.call(self.client([APIError(code, message)]))
                self.assertEqual(result.error_kind, kind)
                self.assertNotIn("secret", result.error)
                self.assertEqual(len(self.calls), before + 1)
        self.assertFalse(self.clock.sleeps)

    def test_feature_fallbacks_and_per_model_flags(self):
        client = self.client([APIError(400, "developer instruction not enabled"),
                              APIError(400, "response_mime_type unsupported"),
                              APIError(400, "thinking unsupported"),
                              LLMResult(True, text="{}"), LLMResult(True, text="{}")])
        self.assertTrue(self.call(client).ok)
        final = self.calls[-1]
        self.assertEqual(final["contents"], "SYSTEM INSTRUCTIONS:\nsystem\n\nINPUT:\nuser")
        self.assertNotIn("system_instruction", final["config"])
        self.assertNotIn("response_mime_type", final["config"])
        self.assertNotIn("thinking_config", final["config"])
        self.settings.model_proposer = "different-model"
        self.assertTrue(self.call(client, role="proposer").ok)
        self.assertEqual(self.calls[-1]["config"]["thinking_config"], {"thinking_level": "high"})
        self.assertTrue(client.capability_flags["different-model"]["system_instruction"])

    def test_unknown_400_does_not_retry(self):
        result = self.call(self.client([APIError(400, "Invalid model")]))
        self.assertEqual(result.error_kind, "bad_request")
        self.assertEqual(len(self.calls), 1)

    def test_cache_roundtrip_and_usage_log(self):
        with tempfile.TemporaryDirectory() as folder:
            self.settings.cache_dir = str(Path(folder) / "cache")
            self.settings.usage_log = str(Path(folder) / "usage.jsonl")
            client = self.client([LLMResult(True, text="{}", input_tokens=12, output_tokens=3)])
            self.assertFalse(self.call(client).cached)
            cached = self.call(client)
            self.assertTrue(cached.cached)
            self.assertEqual(cached.input_tokens, 12)
            self.assertEqual(len(self.calls), 1)
            self.assertTrue(self.call(self.client([])).cached)
            rows = [json.loads(s) for s in Path(self.settings.usage_log).read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(rows[-1]["cached"])
            self.assertEqual(set(rows[0]), {"role", "model", "ok", "error_kind", "input_tokens",
                                             "output_tokens", "latency_s", "cached"})

    def test_fallback_cache_replays_in_fresh_client(self):
        with tempfile.TemporaryDirectory() as folder:
            self.settings.cache_dir = folder
            client = self.client([APIError(400, "system_instruction unsupported"), LLMResult(True, text="{}")])
            self.assertTrue(self.call(client).ok)
            self.assertTrue(self.call(self.client([])).cached)
            self.assertTrue(self.call(client).cached)
            self.assertEqual(len(self.calls), 2)

    def test_cache_identity_includes_role_options_and_prompt(self):
        with tempfile.TemporaryDirectory() as folder:
            self.settings.cache_dir = folder
            client = self.client([LLMResult(True, text="{}") for _ in range(5)])
            for kwargs in ({}, {"role": "baseline"}, {"json_expected": False},
                           {"max_output_tokens": 5}, {"user": "different"}):
                self.assertTrue(self.call(client, **kwargs).ok)
            self.assertEqual(len(self.calls), 5)

    def test_failures_not_cached_and_corrupt_cache_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            self.settings.cache_dir = folder
            client = self.client([APIError(403, "no"), LLMResult(True, text="{}"), LLMResult(True, text="{}")])
            self.assertFalse(self.call(client).ok)
            self.assertFalse(list(Path(folder).glob("*.json")))
            self.assertTrue(self.call(client).ok)
            for path in Path(folder).glob("*.json"):
                path.write_text("garbage")
            self.assertTrue(self.call(client).ok)
            self.assertEqual(len(self.calls), 3)

    def test_deadline_before_call(self):
        self.settings.deadline_epoch = 999
        result = self.call(self.client([]))
        self.assertEqual(result.error_kind, "deadline")
        self.assertFalse(self.calls)

    def test_retry_deadline(self):
        self.settings.deadline_epoch = 1001
        result = self.call(self.client([APIError(503, "unavailable")]))
        self.assertEqual(result.error_kind, "deadline")
        self.assertEqual(self.clock.sleeps, [1])

    def test_timeout_and_server_backoff(self):
        result = self.call(self.client([TimeoutError(), APIError(502, "bad gateway"),
                                       APIError(504, "timeout"), LLMResult(True, text="{}")]))
        self.assertTrue(result.ok)
        self.assertEqual(self.clock.sleeps, [2, 4, 8])

    def test_rpm_sliding_window(self):
        self.settings.rpm_limit = 2
        client = self.client([LLMResult(True, text="{}") for _ in range(3)])
        for _ in range(3):
            self.assertTrue(self.call(client).ok)
        self.assertEqual(self.clock.sleeps, [60])

    def test_tpm_estimate_and_observed_correction(self):
        self.settings.tpm_limit = 10
        client = self.client([{"text": "{}", "usage_metadata": {"prompt_token_count": 8, "candidates_token_count": 2}},
                              LLMResult(True, text="{}")])
        result = self.call(client, system="abcde", user="abcde")  # estimate 4, actual 8
        self.assertEqual(result.input_tokens, 8)
        self.assertEqual(result.output_tokens, 2)
        self.assertTrue(self.call(client, system="abcde", user="abcde").ok)
        self.assertEqual(self.clock.sleeps, [60])

    def test_oversize_request_goes_alone(self):
        self.settings.tpm_limit = 4
        client = self.client([LLMResult(True, text="{}") for _ in range(3)])
        self.call(client, system="", user="x")
        self.call(client, system="", user="x" * 20)
        self.call(client, system="", user="x")
        self.assertEqual(self.clock.sleeps, [60, 60])

    def test_no_system_probe_and_json_optional(self):
        client = self.client([LLMResult(True, text="{}")])
        self.call(client, system="", json_expected=False)
        self.assertNotIn("system_instruction", self.calls[0]["config"])
        self.assertNotIn("response_mime_type", self.calls[0]["config"])
        self.assertEqual(self.calls[0]["config"]["temperature"], 0)

    def test_empty_response_and_unknown_role(self):
        self.assertEqual(self.call(self.client([{"text": None}])).error_kind, "other")
        self.assertEqual(self.call(self.client([]), role="unknown").error_kind, "bad_request")

    def test_thread_safe_log_and_concurrency_bound(self):
        active = maximum = 0
        lock = threading.Lock()
        gate = threading.Barrier(2)
        self.settings.max_concurrency = 2
        def transport(**kwargs):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            gate.wait(timeout=5)
            with lock:
                active -= 1
            return LLMResult(True, text="{}")
        with tempfile.TemporaryDirectory() as folder:
            self.settings.usage_log = str(Path(folder) / "usage.jsonl")
            client = GemmaClient(self.settings, transport=transport)
            with ThreadPoolExecutor(4) as pool:
                results = list(pool.map(lambda i: self.call(client, user=str(i)), range(4)))
            self.assertTrue(all(r.ok for r in results))
            self.assertEqual(maximum, 2)
            self.assertEqual(len(Path(self.settings.usage_log).read_text().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
