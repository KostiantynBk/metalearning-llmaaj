import importlib
import json
import unittest

from rulejudge.contracts import (Assessment, Evidence, Settings, assessment_from_dict, canonical_json,
                                 rulebook_from_dict, stable_hash, to_jsonable)
from rulejudge.fake_client import FakeLLMClient
from rulejudge.jsonutil import extract_json
from rulejudge.render import render_item, resolve_pointer
from tests.fixtures import make_item, toy_dataset, toy_rulebook

MODULES = ["contracts", "jsonutil", "render", "fake_client", "io_contract", "priors", "artifacts", "pipeline",
           "gemma_client", "assessor", "induction", "split", "decision", "explain", "baseline"]


class SkeletonTest(unittest.TestCase):
    def test_all_modules_import(self):
        for name in MODULES:
            importlib.import_module(f"rulejudge.{name}")

    def test_round_trips(self):
        rb = toy_rulebook()
        self.assertEqual(rulebook_from_dict(json.loads(canonical_json(rb))), rb)
        a = Assessment("x", "ok", {"R01": 2, "R02": "na"}, [Evidence("R01", 2, ["msg 2"], "wrong date")])
        self.assertEqual(assessment_from_dict(to_jsonable(a)), a)
        self.assertEqual(stable_hash("a", 1), stable_hash("a", 1))
        self.assertNotEqual(stable_hash("a", 1), stable_hash("a", 2))

    def test_settings_from_env(self):
        s = Settings.from_env({"RJ_RPM": "30", "RJ_FAKE_LLM": "1", "RJ_INPUT_DIR": "/tmp/in"})
        self.assertEqual((s.rpm_limit, s.fake_llm, s.input_dir), (30, True, "/tmp/in"))

    def test_helpers(self):
        it = make_item("i1")
        self.assertNotIn("i1", render_item(it))
        self.assertTrue(resolve_pointer(it, "context.credit_statement.due_date"))
        self.assertEqual(extract_json('```json\n{"label": "bad"}\n```'), {"label": "bad"})
        fake = FakeLLMClient(lambda role, system, user: '{"ok": true}')
        self.assertTrue(fake.generate(role="assessor", system="s", user="u").ok)
        self.assertEqual(len(toy_dataset()), 20)


if __name__ == "__main__":
    unittest.main()
