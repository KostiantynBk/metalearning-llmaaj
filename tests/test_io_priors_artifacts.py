"""WP1 synthetic tests: no API, Docker, credentials or task-data copies."""
import csv
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from rulejudge.artifacts import ArtifactStore
from rulejudge.contracts import HeadParams, Rule, Rulebook, to_jsonable
from rulejudge.io_contract import load_items, write_predictions_atomic
from rulejudge.priors import fit_priors, prior_label
from tests.fixtures import make_item


class IOTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _rows(self, rows):
        path = self.root / "items.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return str(path)

    def test_round_trip_and_bom(self):
        row = asdict(make_item("id", label="good", user="¿Cuándo?"))
        path = self._rows([row])
        Path(path).write_text("\ufeff" + Path(path).read_text(encoding="utf-8"), encoding="utf-8")
        self.assertEqual(load_items(path, True), [make_item("id", label="good", user="¿Cuándo?")])

    def test_invalid_json_and_empty_rejected(self):
        path = self.root / "bad.jsonl"
        for content in ("{broken", "\n", "[]", "null"):
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_items(str(path), True)

    def test_duplicate_ids_and_content_validation(self):
        row = asdict(make_item("id", label="good"))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_items(self._rows([row, row]), True)
        invalid = [dict(row, item_id=" "), dict(row, topic="other"), dict(row, context=[]),
                   dict(row, conversation=[]), dict(row, label="GOOD"),
                   dict(row, conversation=[{"role": "assistant", "content": ""}]),
                   dict(row, conversation=[{"role": "other", "content": "x"}]),
                   dict(row, conversation=[{"role": "user", "content": "x", "extra": 1}]),
                   dict(row, unexpected="x")]
        for bad in invalid:
            with self.subTest(row=bad), self.assertRaises(ValueError):
                load_items(self._rows([bad]), True)
        del row["label"]
        with self.assertRaisesRegex(ValueError, "labels"):
            load_items(self._rows([row]), True)
        self.assertIsNone(load_items(self._rows([row]), False)[0].label)

    def test_csv_exact_header_order_and_atomic_replace(self):
        path = self.root / "new" / "predictions.csv"
        write_predictions_atomic(str(path), {"b": "bad", "a": "good"}, ["a", "b"])
        self.assertEqual(path.read_text(), "item_id,label\na,good\nb,bad\n")
        old = path.read_bytes()
        real_replace = __import__("os").replace
        def check_replace(src, dst):
            self.assertEqual(path.read_bytes(), old)
            self.assertEqual(Path(src).parent, path.parent)
            with open(src, newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)
            real_replace(src, dst)
        with patch("rulejudge.io_contract.os.replace", side_effect=check_replace):
            write_predictions_atomic(str(path), {"a": "bad", "b": "good"}, ["a", "b"])
        self.assertEqual(path.read_text(), "item_id,label\na,bad\nb,good\n")

    def test_invalid_predictions_preserve_existing_csv(self):
        path = self.root / "predictions.csv"
        write_predictions_atomic(str(path), {"a": "good"}, ["a"])
        old = path.read_bytes()
        for labels, ids in [({}, ["a"]), ({"a": "no"}, ["a"]), ({"a": "good"}, ["a", "a"])]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                write_predictions_atomic(str(path), labels, ids)
            self.assertEqual(path.read_bytes(), old)
        with patch("rulejudge.io_contract.os.replace", side_effect=OSError("simulated")):
            with self.assertRaises(OSError):
                write_predictions_atomic(str(path), {"a": "bad"}, ["a"])
        self.assertEqual(path.read_bytes(), old)
        self.assertEqual(list(self.root.iterdir()), [path])


class PriorTests(unittest.TestCase):
    def test_majority_tie_and_missing_topics(self):
        items = [make_item("a", "credit", "bad"), make_item("b", "credit", "good"),
                 make_item("c", "savings", "bad"), make_item("d", "savings", "bad")]
        p = fit_priors(items)
        self.assertEqual(p["global"], "bad")
        self.assertEqual(p["counts"]["credit"], {"good": 1, "bad": 1})
        self.assertEqual(prior_label(p, "credit"), "good")
        self.assertEqual(prior_label(p, "spending"), "bad")
        self.assertEqual(prior_label(p, "unknown"), "bad")
        self.assertEqual(fit_priors([])["global"], "good")


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = ArtifactStore(str(self.root))
        self.runtime = {"rulebook.json": to_jsonable(Rulebook([])),
                        "head.json": to_jsonable(HeadParams("prior")),
                        "priors.json": fit_priors([make_item("a", label="good")]),
                        "assessor_config.json": {"template": None}}
        self.meta = {"run_id": "test", "train_fingerprint": "123"}

    def _bundle(self, mode="prior_only", runtime=None):
        return self.store.write_bundle(mode, runtime or self.runtime, {"explanation.md": "No rules\n"}, self.meta)

    def test_commit_round_trip_and_no_switch_before_commit(self):
        first = self._bundle()
        self.assertFalse((self.root / "manifest.json").exists())
        self.store.commit(first)
        rt = self.store.load_runtime()
        self.assertEqual(rt["head"], HeadParams("prior"))
        self.assertEqual(rt["mode"], "prior_only")
        self.assertEqual(rt["priors"], self.runtime["priors.json"])
        second = self._bundle()
        self.assertNotEqual(first, second)
        self.assertTrue(self.store.load_runtime()["manifest"]["bundle"].endswith(first))
        self.store.commit(second)
        self.assertTrue(self.store.load_runtime()["manifest"]["bundle"].endswith(second))

    def test_learned_round_trip(self):
        self.runtime["rulebook.json"] = to_jsonable(Rulebook([Rule("R01", "Match context")]))
        self.runtime["head.json"] = to_jsonable(HeadParams("fitted", 3, ["R02"], "F+S"))
        self.store.commit(self._bundle("learned"))
        rt = self.store.load_runtime()
        self.assertEqual(rt["rulebook"].rules[0].rule_id, "R01")
        self.assertEqual(rt["head"].tau, 3)

    def test_runtime_sha_mismatch_detected_and_jury_not_read(self):
        bid = self._bundle()
        self.store.commit(bid)
        jury = self.root / "bundles" / bid / "jury" / "explanation.md"
        jury.unlink()  # Simulate a runtime-only deployment, not a source-file deletion.
        self.assertEqual(self.store.load_runtime()["mode"], "prior_only")
        path = self.root / "bundles" / bid / "runtime" / "head.json"
        path.write_text('{}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.store.load_runtime()

    def test_invalid_bundle_and_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.store.load_runtime()
        with self.assertRaises(ValueError):
            self.store.write_bundle("prior_only", {}, {"explanation.md": "x"}, self.meta)
        with self.assertRaises(ValueError):
            self.store.write_bundle("prior_only", self.runtime, {"explanation.md": "x", "../../x": "x"}, self.meta)
        with self.assertRaises(ValueError):
            self.store.commit("../x")
        bid = self._bundle()
        self.store.commit(bid)
        path = self.root / "manifest.json"
        m = json.loads(path.read_text())
        m["bundle"] = "../outside"
        path.write_text(json.dumps(m), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.store.load_runtime()

    def test_failed_manifest_switch_preserves_previous(self):
        first = self._bundle()
        self.store.commit(first)
        second = self._bundle()
        with patch("rulejudge.artifacts.os.replace", side_effect=OSError("simulated")):
            with self.assertRaises(ValueError):
                self.store.commit(second)
        self.assertTrue(self.store.load_runtime()["manifest"]["bundle"].endswith(first))
