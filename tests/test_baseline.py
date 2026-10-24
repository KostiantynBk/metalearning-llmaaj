import time
import unittest

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


if __name__ == "__main__":
    unittest.main()
