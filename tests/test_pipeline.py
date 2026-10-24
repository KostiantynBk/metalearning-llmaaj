"""WP1 integration tests using synthetic items and mocked unmerged work packages."""
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from rulejudge import pipeline
from rulejudge.artifacts import ArtifactStore
from rulejudge.contracts import (Assessment, Decision, HeadParams, Rulebook, Settings,
                                item_text_for_hash, stable_hash)
from rulejudge.fake_client import FakeLLMClient
from tests.fixtures import make_item, toy_dataset, toy_rulebook


def _decide(item, assessment, head, priors):
    prior = priors["by_topic"].get(item.topic, priors["global"])
    if head.kind == "prior":
        return Decision(item.item_id, prior, "prior_mode")
    if assessment is None or assessment.status == "failed":
        return Decision(item.item_id, prior, "fallback_prior")
    triggered = [rid for rid, value in assessment.support.items()
                 if type(value) is int and value >= head.tau and rid not in head.disabled]
    if triggered:
        return Decision(item.item_id, "bad", "model", triggered)
    return Decision(item.item_id, "good", "model")


def _metric(items, predictions):
    groups = {}
    for item in items:
        n, correct = groups.get(item.topic, (0, 0))
        groups[item.topic] = (n + 1, correct + (item.label == predictions[item.item_id]))
    return {"macro": sum(c / n for n, c in groups.values()) / len(groups) if groups else None,
            "by_topic": {t: {"n": n, "correct": c, "accuracy": c / n} for t, (n, c) in groups.items()},
            "n": len(items), "all_topics_present": len(groups) == 4}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.settings = Settings(input_dir=str(self.root / "input"),
                                 artifacts_dir=str(self.root / "artifacts"),
                                 output_dir=str(self.root / "output"))
        (self.root / "input").mkdir()
        self.train = toy_dataset()
        self.test = [make_item("test-good"), make_item("test-bad", assistant="WRONG-DATE")]
        self._write("train", self.train)
        self._write("test", self.test, labels=False)
        self.rulebook = toy_rulebook()
        self.rulebook.rules[0].provenance_ids = [self.train[0].item_id]
        self.client = FakeLLMClient(default='{"ok":true}')
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {"RJ_STRICT": "0", "RJ_CACHE_DIR": ""}))
        self.mocks = {}
        for target, kwargs in {
            "gemma_client.make_client": {"return_value": self.client},
            "gemma_client.preflight": {"side_effect": lambda client: (client.generate(role="assessor", system="synthetic", user="probe").ok, "ok")},
            "split.split_fs": {"return_value": ([i.item_id for i in self.train[:-4]], [i.item_id for i in self.train[-4:]])},
            "induction.induce_rulebook": {"return_value": (self.rulebook, {"calls": 1})},
            "assessor.assess_many": {"side_effect": self._assess},
            "decision.fit_fitted_head": {"return_value": HeadParams("fitted", 2, [], "F")},
            "decision.select_head": {"side_effect": lambda heads, *args: (heads[0], {"reason": "tie"})},
            "decision.apply_floor": {"side_effect": lambda head, *args: (head, {"applied": False})},
            "decision.refit_final": {"side_effect": lambda head, *args: head},
            "decision.decide": {"side_effect": _decide},
            "decision.topic_macro_accuracy": {"side_effect": _metric},
            "decision.rule_stats": {"return_value": {}},
            "explain.select_examples": {"return_value": {}},
            "explain.render_explanation": {"return_value": "# Learned explanation\n"},
            "explain.render_prior_only_explanation": {"return_value": "# Prior explanation\n"},
        }.items():
            self.mocks[target] = self.stack.enter_context(patch("rulejudge.pipeline." + target, **kwargs))
        self.stderr = io.StringIO()
        self.stack.enter_context(redirect_stderr(self.stderr))

    def _write(self, name, items, labels=True):
        rows = []
        for item in items:
            row = asdict(item)
            if not labels:
                row.pop("label")
            rows.append(json.dumps(row))
        (self.root / "input" / (name + ".jsonl")).write_text("\n".join(rows) + "\n", encoding="utf-8")

    def _assess(self, items, rulebook, client, settings, on_progress=None):
        if on_progress is not None:
            # Completeness must hold BEFORE the very first test assessment.
            self.assertEqual(self._predictions(), {"test-good": "good", "test-bad": "good"})
        out = {}
        for item in items:
            value = 3 if "WRONG-DATE" in item.conversation[-1]["content"] else 0
            out[item.item_id] = Assessment(item.item_id, "ok", {"R01": value})
            if on_progress:
                on_progress(dict(out))
                self.assertEqual(set(self._predictions()), {i.item_id for i in items})
        return out

    def _predictions(self):
        path = self.root / "output" / "predictions.csv"
        self.assertTrue(path.exists())
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, ["item_id", "label"])
            return {r["item_id"]: r["label"] for r in reader}

    def _adapt(self):
        self.assertEqual(pipeline.run_adapt(self.settings, self.client), 0)
        return ArtifactStore(self.settings.artifacts_dir).load_runtime()

    def test_end_to_end_fake_client_and_training_report(self):
        rt = self._adapt()
        self.assertEqual(rt["mode"], "learned")
        self.assertTrue(self.client.calls)
        path = Path(self.settings.artifacts_dir) / rt["manifest"]["bundle"]
        runtime_rules = json.loads((path / "runtime" / "rulebook.json").read_text())
        self.assertNotIn("provenance_ids", runtime_rules["rules"][0])
        jury_rules = json.loads((path / "jury" / "rulebook_with_provenance.json").read_text())
        self.assertEqual(jury_rules["rules"][0]["provenance_ids"], [self.train[0].item_id])
        report = self.mocks["explain.render_explanation"].call_args.args[0]
        expected = {"run_info", "rulebook", "head", "selection", "rule_stats", "examples", "metrics",
                    "fallbacks", "induction_notes", "hashes", "items_by_id", "assessments"}
        self.assertEqual(set(report), expected)
        self.assertEqual(report["selection"]["reason"], "tie")
        self.assertEqual(report["selection"]["selection_report"], {"reason": "tie"})
        saved_selection = json.loads((path / "jury" / "selection.json").read_text())
        self.assertEqual(saved_selection["reason"], "tie")
        fingerprint = stable_hash([item_text_for_hash(i) + i.label for i in self.train])
        self.assertEqual(report["run_info"]["train_fingerprint"], fingerprint)
        self.assertTrue(report["run_info"]["run_id"].endswith(fingerprint[:6]))
        for name in ("rulebook.json", "head.json"):
            self.assertEqual(report["hashes"][name], hashlib.sha256((path / "runtime" / name).read_bytes()).hexdigest())
        self.settings.model_assessor = "another-env-model"
        self.assertEqual(pipeline.run_eval(self.settings, self.client), 0)
        self.assertEqual(self._predictions(), {"test-good": "good", "test-bad": "bad"})
        self.assertEqual(self.settings.model_assessor, rt["assessor_config"]["model"])
        self.assertIn('"model": 2', self.stderr.getvalue())
        self.assertEqual(self.settings.eval_source_counts, {"model": 2})

    def test_prior_bundle_committed_before_first_model_call(self):
        def preflight(client):
            rt = ArtifactStore(self.settings.artifacts_dir).load_runtime()
            self.assertEqual(rt["mode"], "prior_only")
            self.assertTrue(self.settings.cache_dir)
            self.assertTrue(self.settings.usage_log)
            return False, "daily_quota"
        self.mocks["gemma_client.preflight"].side_effect = preflight
        rt = self._adapt()
        self.assertEqual(rt["mode"], "prior_only")
        self.assertEqual(pipeline.run_eval(self.settings, self.client), 0)
        self.assertEqual(self._predictions(), {"test-good": "good", "test-bad": "good"})
        self.mocks["assessor.assess_many"].assert_not_called()
        self.assertEqual(self.settings.eval_source_counts, {"prior_mode": 2})
        self.assertIn("daily_quota", self.stderr.getvalue())

    def test_eval_failed_preflight_keeps_valid_prior_csv(self):
        self._adapt()
        def preflight(client):
            self.assertEqual(self._predictions(), {"test-good": "good", "test-bad": "good"})
            return False, "auth"
        self.mocks["gemma_client.preflight"].side_effect = preflight
        self.assertEqual(pipeline.run_eval(self.settings, self.client), 0)
        self.assertEqual(self._predictions()["test-bad"], "good")
        self.assertEqual(self.settings.eval_source_counts, {"fallback_prior": 2})
        self.assertIn("auth", self.stderr.getvalue())

    def test_explanation_failure_retains_prior_bundle(self):
        self.mocks["explain.render_explanation"].side_effect = RuntimeError("bad explanation")
        self.assertEqual(self._adapt()["mode"], "prior_only")
        self.assertIn("bad explanation", self.stderr.getvalue())

    def test_prior_explanation_failure_uses_five_line_fallback(self):
        self.mocks["explain.render_prior_only_explanation"].side_effect = RuntimeError("bad renderer")
        self.mocks["gemma_client.preflight"].side_effect = lambda c: (False, "quota")
        rt = self._adapt()
        path = Path(self.settings.artifacts_dir) / rt["manifest"]["bundle"] / "jury" / "explanation.md"
        text = path.read_text(encoding="utf-8")
        self.assertEqual(len(text.splitlines()), 5)
        self.assertIn("No rules were learned", text)

    def test_no_rules_degrades(self):
        self.mocks["induction.induce_rulebook"].return_value = (Rulebook([]), {})
        self.assertEqual(self._adapt()["mode"], "prior_only")
        self.mocks["assessor.assess_many"].assert_not_called()

    def test_exception_after_eval_checkpoint_preserves_completed_labels(self):
        self._adapt()
        def fail_after_progress(items, rulebook, client, settings, on_progress=None):
            on_progress({"test-bad": Assessment("test-bad", "ok", {"R01": 3})})
            raise RuntimeError("simulated outage")
        self.mocks["assessor.assess_many"].side_effect = fail_after_progress
        self.assertEqual(pipeline.run_eval(self.settings, self.client), 0)
        self.assertEqual(self._predictions(), {"test-good": "good", "test-bad": "bad"})
        self.assertIn("simulated outage", self.stderr.getvalue())
        self.assertEqual(self.settings.eval_source_counts, {"model": 1, "fallback_prior": 1})

    def test_eval_source_counts_reset_and_track_missing_assessments(self):
        self._adapt()
        self.settings.eval_source_counts = {"stale": 999}
        self.mocks["assessor.assess_many"].side_effect = lambda *args, **kwargs: {
            "test-bad": Assessment("test-bad", "ok", {"R01": 3})}
        self.assertEqual(pipeline.run_eval(self.settings, self.client), 0)
        self.assertEqual(self.settings.eval_source_counts, {"model": 1, "fallback_prior": 1})
        self.assertEqual(self._predictions(), {"test-good": "good", "test-bad": "bad"})
        self.mocks["gemma_client.preflight"].side_effect = lambda client: (False, "auth")
        self.assertEqual(pipeline.run_eval(self.settings, self.client), 0)
        self.assertEqual(self.settings.eval_source_counts, {"fallback_prior": 2})
        (self.root / "input" / "test.jsonl").write_text("bad", encoding="utf-8")
        self.assertEqual(pipeline.run_eval(self.settings, self.client), 2)
        self.assertEqual(self.settings.eval_source_counts, {})

    def test_cache_override_supported_in_both_stages(self):
        cache = str(self.root / "shared-cache")
        with patch.dict(os.environ, {"RJ_CACHE_DIR": cache}):
            self._adapt()
            self.assertEqual(self.settings.cache_dir, cache)
            self.assertEqual(pipeline.run_eval(self.settings, self.client), 0)
            self.assertEqual(self.settings.cache_dir, cache)
            self.assertFalse(Path(self.settings.usage_log).is_relative_to(Path(self.settings.artifacts_dir)))

    def test_eval_prior_head_skips_client_even_in_learned_mode(self):
        self.mocks["decision.apply_floor"].side_effect = lambda *args: (HeadParams("prior"), {"applied": True})
        self.assertEqual(self._adapt()["head"].kind, "prior")
        calls = self.mocks["gemma_client.preflight"].call_count
        self.assertEqual(pipeline.run_eval(self.settings, self.client), 0)
        self.assertEqual(calls, self.mocks["gemma_client.preflight"].call_count)
        self.assertEqual(self._predictions()["test-bad"], "good")
        self.assertEqual(self.settings.eval_source_counts, {"prior_mode": 2})

    def test_invalid_input_or_artifact_returns_two(self):
        self.assertEqual(pipeline.run_eval(self.settings, self.client), 2)
        (self.root / "input" / "train.jsonl").write_text("bad", encoding="utf-8")
        self.assertEqual(pipeline.run_adapt(self.settings, self.client), 2)

    def test_strict_reraises_after_checkpoint_and_invalid_input(self):
        self.mocks["induction.induce_rulebook"].side_effect = RuntimeError("induction failed")
        with patch.dict(os.environ, {"RJ_STRICT": "1"}):
            with self.assertRaisesRegex(RuntimeError, "induction failed"):
                pipeline.run_adapt(self.settings, self.client)
        self.assertEqual(ArtifactStore(self.settings.artifacts_dir).load_runtime()["mode"], "prior_only")
        self.mocks["induction.induce_rulebook"].side_effect = None
        self._adapt()
        self.mocks["assessor.assess_many"].side_effect = RuntimeError("eval failed")
        with patch.dict(os.environ, {"RJ_STRICT": "1"}):
            with self.assertRaisesRegex(RuntimeError, "eval failed"):
                pipeline.run_eval(self.settings, self.client)
        self.assertEqual(set(self._predictions()), {"test-good", "test-bad"})
        (self.root / "input" / "test.jsonl").write_text("bad", encoding="utf-8")
        with patch.dict(os.environ, {"RJ_STRICT": "1"}):
            with self.assertRaises(ValueError):
                pipeline.run_eval(self.settings, self.client)

    def test_cli_uses_fake_flag_and_environment_directories(self):
        env = {"RJ_INPUT_DIR": self.settings.input_dir, "RJ_ARTIFACTS_DIR": self.settings.artifacts_dir,
               "RJ_OUTPUT_DIR": self.settings.output_dir, "RJ_FAKE_LLM": "1"}
        with patch.dict(os.environ, env):
            self.assertEqual(pipeline.main(["adapt"]), 0)
            made_settings = self.mocks["gemma_client.make_client"].call_args.args[0]
            self.assertTrue(made_settings.fake_llm)
            self.assertEqual(made_settings.input_dir, self.settings.input_dir)
            self.assertEqual(pipeline.main(["eval"]), 0)
        self.assertEqual(self._predictions(), {"test-good": "good", "test-bad": "bad"})

    def test_python_module_entrypoint_without_network(self):
        # Stubs may degrade adapt on this branch; real merged WPs use FakeLLMClient.
        # Either case must produce a valid priors-first CLI bundle and complete CSV.
        env = dict(os.environ, RJ_INPUT_DIR=self.settings.input_dir,
                   RJ_ARTIFACTS_DIR=self.settings.artifacts_dir, RJ_OUTPUT_DIR=self.settings.output_dir,
                   RJ_FAKE_LLM="1", RJ_STRICT="0", RJ_TIME_BUDGET_S="5")
        for command in ("adapt", "eval"):
            result = subprocess.run([sys.executable, "-m", "rulejudge", command],
                                    cwd=Path(__file__).resolve().parents[1], env=env,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(self._predictions()), {"test-good", "test-bad"})

    def test_cli_dispatch_help_usage_and_invalid_environment(self):
        with patch("rulejudge.pipeline.run_adapt", return_value=0) as adapt:
            self.assertEqual(pipeline.main(["adapt"]), 0)
            adapt.assert_called_once()
        with patch("rulejudge.pipeline.run_eval", return_value=0) as evaluate:
            self.assertEqual(pipeline.main(["eval"]), 0)
            evaluate.assert_called_once()
        self.assertEqual(pipeline.main([]), 2)
        self.assertEqual(pipeline.main(["other"]), 2)
        with patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(pipeline.main(["--help"]), 0)
        with patch.dict(os.environ, {"RJ_RPM": "not-an-int"}):
            self.assertEqual(pipeline.main(["adapt"]), 2)
