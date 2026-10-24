import contextlib
import inspect
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts import run_public
from rulejudge import pipeline
from rulejudge.contracts import TOPICS, HeadParams, item_from_dict, to_jsonable
from tests.fixtures import toy_dataset

from rulejudge.baseline import build_baseline_request, run_baseline
from rulejudge.contracts import LLMResult, Settings
from rulejudge.fake_client import FakeLLMClient
from rulejudge.render import render_item
from tests.fixtures import make_item


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.priors = {"by_topic": {"credit": "bad"}, "global": "good"}
        self.item = make_item("secret-item-id", label="bad")

    def test_prompt_uses_canonical_rendering_and_no_gold_or_id(self):
        system, user = build_baseline_request(self.item)
        self.assertEqual(user, render_item(self.item))
        self.assertNotIn(self.item.item_id, system + user)
        self.assertNotIn('"label": "bad"', user)
        self.assertEqual(build_baseline_request(make_item(self.item.item_id, label="good")),
                         (system, user))
        self.assertIn('Return ONLY JSON {"label": "good"|"bad"}', system)

    def test_parses_json_fences_and_surrounding_prose(self):
        for text, label in [('{"label":"good"}', "good"),
                            ('```json\n{"label":"bad"}\n```', "bad"),
                            ('Answer: {"label":"good"} done.', "good")]:
            with self.subTest(text=text):
                client = FakeLLMClient(default=text)
                result = run_baseline([self.item], client, Settings(), self.priors)
                self.assertEqual(result[self.item.item_id], {"label": label, "source": "model"})
                self.assertEqual(len(client.calls), 1)
                self.assertEqual(client.calls[0]["role"], "baseline")

    def test_bad_schema_and_labels_fall_back_after_two_attempts(self):
        for text in ['garbage', '[]', '{"label":"GOOD"}', '{"label":null}',
                     '{"label":1}', '{"label":["bad"]}', '{"verdict":"bad"}']:
            with self.subTest(text=text):
                client = FakeLLMClient(default=text)
                result = run_baseline([self.item], client, Settings(), self.priors)
                self.assertEqual(result[self.item.item_id], {"label": "bad", "source": "fallback"})
                self.assertEqual(len(client.calls), 2)

    def test_retry_can_recover(self):
        answers = iter(['not json', '{"label":"good"}'])
        client = FakeLLMClient(lambda *_: next(answers))
        result = run_baseline([self.item], client, Settings(), self.priors)
        self.assertEqual(result[self.item.item_id]["source"], "model")
        self.assertIn("previous reply", client.calls[1]["user"])

    def test_api_errors_retry_then_use_global_prior(self):
        client = FakeLLMClient(lambda *_: LLMResult(False, error_kind="unavailable"))
        item = make_item("s1", topic="savings")
        result = run_baseline([item], client, Settings(), self.priors)
        self.assertEqual(result["s1"], {"label": "good", "source": "fallback"})
        self.assertEqual(len(client.calls), 2)

    def test_terminal_errors_stop_all_remaining_calls(self):
        for kind in ("auth", "daily_quota", "deadline"):
            with self.subTest(kind=kind):
                client = FakeLLMClient(lambda *_: LLMResult(False, error_kind=kind))
                result = run_baseline([self.item, make_item("s1", "savings")],
                                      client, Settings(), self.priors)
                self.assertEqual(len(client.calls), 1)
                self.assertEqual(result["s1"]["source"], "fallback")

    def test_deadline_uses_prior_without_calls(self):
        client = FakeLLMClient()
        settings = Settings(deadline_epoch=time.time() - 1)
        result = run_baseline([self.item], client, settings, self.priors)
        self.assertEqual(result[self.item.item_id]["source"], "fallback")
        self.assertEqual(client.calls, [])

    def test_empty_input_and_invalid_prior(self):
        client = FakeLLMClient()
        self.assertEqual(run_baseline([], client, Settings(), {}), {})
        with self.assertRaises(ValueError):
            run_baseline([self.item], client, Settings(), {"global": "invalid"})


def _pipeline_implemented():
    return all("raise NotImplementedError" not in inspect.getsource(func)
               for func in (pipeline.run_adapt, pipeline.run_eval))


class PublicRunnerTests(unittest.TestCase):
    def test_subsets_are_deterministic_and_retain_topics(self):
        rows = [to_jsonable(item) for item in toy_dataset(8, 4)]
        first = run_public._stratified_subset(rows, 16, train=True)
        second = run_public._stratified_subset(list(reversed(rows)), 16, train=True)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual({row["topic"] for row in first}, set(TOPICS))
        for topic in TOPICS:
            self.assertEqual({row["label"] for row in first if row["topic"] == topic}, {"good", "bad"})
        with self.assertRaises(ValueError):
            run_public._stratified_subset(rows, 3, train=True)

    def test_test_subset_does_not_depend_on_labels(self):
        rows = [to_jsonable(item) for item in toy_dataset(8, 4)]
        before = run_public._stratified_subset(rows, 9, train=False)
        changed = [{**row, "label": "bad" if row["label"] == "good" else "good"} for row in rows]
        after = run_public._stratified_subset(changed, 9, train=False)
        self.assertEqual([row["item_id"] for row in before], [row["item_id"] for row in after])
        # Removal of test gold altogether also leaves selection unchanged.
        no_labels = [{k: v for k, v in row.items() if k != "label"} for row in rows]
        unlabeled = run_public._stratified_subset(no_labels, 9, train=False)
        self.assertEqual([row["item_id"] for row in before], [row["item_id"] for row in unlabeled])

    def test_quota_allocation_sizes_and_minima(self):
        for count in range(1, 18):
            result = run_public._allocate({"a": 1, "b": 6, "c": 10}, count)
            self.assertEqual(sum(result.values()), count)
            self.assertLessEqual(result["a"], 1)
            if count >= 3:
                self.assertTrue(all(n >= 1 for n in result.values()))

    def test_env_parser_uses_plain_values_and_preserves_existing_exports(self):
        # This is a synthetic settings file, NOT a .env or a real credential.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic-settings.txt"
            path.write_text('# comment\nGEMINI_API_KEY="synthetic-placeholder"\n'
                            'export RJ_RPM=12 # inline comment\nRJ_TPM=999\n'
                            'PATH=do-not-import\nRJ_UNFINISHED="unterminated\n'
                            'MALFORMED\nRJ_LITERAL=$(not-executed)\n', encoding="utf-8")
            target = {"RJ_RPM": "25"}
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                run_public._load_env_file(path, target)
            self.assertEqual(captured.getvalue(), "")
            self.assertEqual(target["GEMINI_API_KEY"], "synthetic-placeholder")
            self.assertEqual(target["RJ_RPM"], "25")
            self.assertEqual(target["RJ_TPM"], "999")
            self.assertEqual(target["RJ_LITERAL"], "$(not-executed)")
            self.assertNotIn("PATH", target)
            self.assertNotIn("RJ_UNFINISHED", target)
            run_public._load_env_file(Path(tmp) / "missing-file", target)

    def test_paired_counts_cover_all_four_outcomes(self):
        gold = [{"item_id": str(n), "topic": "credit", "label": "good"} for n in range(4)]
        ours = {str(n): "good" if n < 2 else "bad" for n in range(4)}
        baseline = {str(n): "good" if n % 2 == 0 else "bad" for n in range(4)}
        counts = run_public._paired_counts(gold, ours, baseline)
        self.assertEqual(counts["credit"], {"both_right": 1, "only_ours": 1,
                                           "only_baseline": 1, "both_wrong": 1})
        self.assertEqual(sum(counts["savings"].values()), 0)

    def test_counting_client_separates_cache_and_phase(self):
        results = iter([LLMResult(True, "{}", input_tokens=10, output_tokens=2),
                        LLMResult(True, "{}", input_tokens=10, output_tokens=2, cached=True),
                        LLMResult(False, error_kind="auth")])
        client = run_public._CountingClient(FakeLLMClient(lambda *_: next(results)))
        for _ in range(2):
            client.generate(role="assessor", system="x", user="y")
        client.phase = "baseline"
        client.generate(role="baseline", system="x", user="y")
        count = client.counts["adapt"]["assessor"]
        self.assertEqual((count["calls"], count["cached_calls"], count["input_tokens"],
                          count["uncached_input_tokens"]), (2, 1, 20, 10))
        self.assertEqual(client.counts["baseline"]["baseline"]["failed_calls"], 1)

    def test_eval_counts_never_infer_source_from_predictions(self):
        runtime = {"mode": "learned", "head": HeadParams("fixed")}
        settings = Settings()
        self.assertFalse(run_public._eval_counts(settings, runtime, 8)["available"])
        settings.eval_source_counts = {"model": 6, "fallback_prior": 2}
        self.assertEqual(run_public._eval_counts(settings, runtime, 8)["fallback_prior"], 2)
        del settings.eval_source_counts
        runtime["head"] = HeadParams("prior")
        self.assertEqual(run_public._eval_counts(settings, runtime, 8)["prior_mode"], 8)

    @staticmethod
    def _datasets(root):
        rows = [to_jsonable(item) for item in toy_dataset(5, 3)]
        train = root / "synthetic-train.jsonl"
        test = root / "synthetic-test.jsonl"
        run_public.write_jsonl(train, rows)
        run_public.write_jsonl(test, [{**row, "item_id": "test-" + row["item_id"]} for row in rows])
        return train, test

    def test_runner_report_with_mocked_pipeline_before_wp1_merge(self):
        def load_items(path, require_labels):
            return [item_from_dict(row) for row in run_public.load_dataset(path, require_labels)]

        def priors(items):
            return {"global": "good", "by_topic": dict.fromkeys(TOPICS, "good")}

        def write_predictions(path, labels, expected):
            self.assertEqual(set(labels), set(expected))
            with Path(path).open("w", encoding="utf-8", newline="") as stream:
                stream.write("item_id,label\n")
                for iid in expected:
                    stream.write(f"{iid},{labels[iid]}\n")

        def adapt(settings, client):
            result = client.generate(role="proposer", system="consolidate", user="PAIR 1")
            self.assertTrue(result.ok)
            return 0

        def evaluate_stage(settings, client):
            items = load_items(Path(settings.input_dir) / "test.jsonl", False)
            self.assertTrue(all(item.label is None for item in items))
            write_predictions(Path(settings.output_dir) / "predictions.csv",
                              {item.item_id: "good" for item in items}, [item.item_id for item in items])
            settings.eval_source_counts = {"prior_mode": len(items)}
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, test = self._datasets(root)
            out = root / "run"
            with patch.dict(os.environ, {}, clear=True), \
                 patch.object(run_public, "load_items", side_effect=load_items), \
                 patch.object(run_public, "fit_priors", side_effect=priors), \
                 patch.object(run_public, "prior_label", return_value="good"), \
                 patch.object(run_public, "write_predictions_atomic", side_effect=write_predictions), \
                 patch.object(pipeline, "run_adapt", side_effect=adapt), \
                 patch.object(pipeline, "run_eval", side_effect=evaluate_stage), \
                 patch.object(run_public.ArtifactStore, "load_runtime", return_value={
                     "mode": "prior_only", "head": HeadParams("prior")}), \
                 patch.object(run_public, "_load_env_file") as dotenv:
                captured = io.StringIO()
                with contextlib.redirect_stdout(captured):
                    code = run_public.main(["--fake", "--train", str(train), "--test", str(test),
                                            "--out", str(out), "--limit-train", "16", "--limit-test", "8"])
                self.assertEqual(code, 0)
                dotenv.assert_not_called()
                self.assertNotIn("RJ_FAKE_LLM", os.environ)
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertEqual((report["n_train"], report["n_test"]), (16, 8))
            self.assertFalse(report["benchmark_eligible"])
            self.assertEqual(set(report["scores"]), {"ours", "same_model_baseline", "majority_prior"})
            self.assertEqual(report["scores"]["same_model_baseline"]["macro_topic_accuracy"], 1.0)
            self.assertIn("SMOKE ONLY", (out / "report.md").read_text(encoding="utf-8"))
            self.assertTrue(all("label" not in row for row in run_public.load_dataset(out / "input/test.jsonl")))
            self.assertEqual(sum(sum(row.values()) for row in report["paired_by_topic"].values()), 8)

    @unittest.skipUnless(_pipeline_implemented(), "WP1 pipeline has not been merged")
    def test_fake_public_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            train, test = self._datasets(root)
            out = root / "run"
            with contextlib.redirect_stdout(io.StringIO()):
                code = run_public.main(["--fake", "--train", str(train), "--test", str(test),
                                        "--out", str(out), "--limit-train", "20", "--limit-test", "8"])
            self.assertEqual(code, 0)
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["benchmark_eligible"])
            self.assertEqual(set(report["scores"]), {"ours", "same_model_baseline", "majority_prior"})
            self.assertEqual(len(run_public.load_labels(out / "output/predictions.csv")), 8)
            self.assertTrue((out / "artifacts/manifest.json").is_file())
            self.assertEqual(report["fallback_counts"]["baseline"]["fallback"], 0)


if __name__ == "__main__":
    unittest.main()
