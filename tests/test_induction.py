"""Synthetic, network-free tests for bounded contrast induction."""
from __future__ import annotations

import json
import re
import time
import unittest
from dataclasses import asdict

from rulejudge.contracts import MAX_RULES, TOPICS, Item, LLMResult, Rule, Rulebook, Settings
from rulejudge.fake_client import FakeLLMClient
from rulejudge.induction import (
    STAGE1_SYSTEM, STAGE2_SYSTEM, _consolidation_batches, _rubric_text,
    content_tokens, contrast_pairs, induce_rulebook, jaccard, validate_rulebook,
)
from tests.fixtures import make_item, toy_dataset


def candidate(pair=1, text="Dates must match the provided context.", **kwargs):
    return dict(pair=pair, text=text, applies_when="A date is requested.",
                violation="The answer contradicts a context date.", boundary="Accurate paraphrases are allowed.",
                topic_specific=False, confidence="high", **kwargs)


def pair_items(topic="credit"):
    return [make_item(f"synthetic-{topic}-bad", topic, "bad", assistant="Incorrect due date."),
            make_item(f"synthetic-{topic}-good", topic, "good", assistant="Correct due date.")]


def happy_responder(role, system, user):
    if system == STAGE1_SYSTEM:
        return json.dumps({"candidates": [candidate(int(k)) for k in re.findall(r"PAIR (\d+) — BAD", user)],
                           "no_clear_rule": []})
    return json.dumps({"rules": [{"text": "Dates must match the provided context.", "scope": "global",
                                 "applies_when": "A date is requested.",
                                 "violation": "The answer contradicts a context date.",
                                 "boundary": "Accurate paraphrases are allowed.",
                                 "merged_candidates": re.findall(r'"id":"(C\d+)"', user)}],
                       "notes": ["Unseen requirements may not be covered."]})


class PairingTests(unittest.TestCase):
    def test_tokens_use_values_not_keys_roles_ids_topics_or_labels(self):
        item = Item("secret-item-id", "credit", {"not_a_token": {"other_key": ["Hola MÉXICO 15 a", "12ab"]},
                                                    "numeric": 12345},
                    [{"role": "assistant", "content": "Спасибо dinero 4 x"}], "bad")
        self.assertEqual(content_tokens(item), {"hola", "méxico", "12ab", "спасибо", "dinero"})

    def test_jaccard(self):
        self.assertEqual(jaccard(set(), set()), 0.0)
        self.assertEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3)
        self.assertEqual(jaccard({"a"}, {"a"}), 1.0)

    def test_same_topic_nearest_and_deterministic_ties(self):
        items = [make_item("bad-nearest", "credit", "bad", user="apple pear", assistant="", context={}),
                 make_item("good-z", "credit", "good", user="apple pear", assistant="", context={}),
                 make_item("good-a", "credit", "good", user="apple pear", assistant="", context={}),
                 make_item("good-unrelated", "credit", "good", user="orange lemon", assistant="", context={}),
                 make_item("other-topic", "savings", "good", user="apple pear", assistant="", context={})]
        result = contrast_pairs(items)
        self.assertEqual([(b.item_id, g.item_id, score) for b, g, score in result],
                         [("bad-nearest", "good-a", 1.0)])
        self.assertEqual(contrast_pairs(list(reversed(items))), result)

    def test_bad_without_good_and_unlabelled_items_are_not_paired(self):
        items = [make_item("unmatched-bad", "spending", "bad"), make_item("test-item", "spending"),
                 make_item("good-other", "credit", "good")]
        self.assertEqual(contrast_pairs(items), [])


class ValidationTests(unittest.TestCase):
    def test_scope_ids_dedupe_and_nonmutation(self):
        source = Rulebook([Rule("old", "Dates must match the provided context.", "invalid"),
                           Rule("old2", "Dates must match the provided context.", "credit"),
                           Rule("old3", ""), Rule("old4", "Ask for missing income data.", "credit")])
        snapshot = asdict(source)
        clean, warnings = validate_rulebook(source, [])
        self.assertEqual([(r.rule_id, r.scope) for r in clean.rules], [("R01", "global"), ("R02", "credit")])
        self.assertEqual(asdict(source), snapshot)
        self.assertTrue(any("duplicate" in w for w in warnings))
        self.assertTrue(any("empty" in w for w in warnings))
        self.assertTrue(any("scope" in w for w in warnings))

    def test_near_duplicate_jaccard(self):
        clean, _ = validate_rulebook(Rulebook([Rule("a", "Give accurate dates amounts rates and balances."),
                                             Rule("b", "Give accurate dates amounts rates and balances now.")]), [])
        self.assertEqual(len(clean.rules), 1)

    def test_customer_names_ids_and_name_boundaries(self):
        item = make_item("train-id-private")
        source = Rulebook([Rule("a", "Tell Ana the correct date."),
                           Rule("b", "Use 0123456789abcdef0123456789abcdef as the customer."),
                           Rule("c", "Use train-id-private as a criterion."),
                           Rule("d", "Analyse dates using context."),
                           Rule("e", "Give accurate values.", boundary="Except when ANA asks.")])
        clean, warnings = validate_rulebook(source, [item])
        self.assertEqual([r.text for r in clean.rules], ["Analyse dates using context."])
        self.assertEqual(sum("customer name" in w for w in warnings), 4)

    def test_field_caps_at_word_boundary(self):
        phrase = "word " * 100
        source = Rulebook([Rule("a", phrase, applies_when=phrase, violation=phrase, boundary=phrase)])
        clean, warnings = validate_rulebook(source, [])
        for field, limit in (("text", 160), ("applies_when", 120), ("violation", 140), ("boundary", 120)):
            value = getattr(clean.rules[0], field)
            self.assertLessEqual(len(value), limit)
            self.assertTrue(value.endswith("word…"), value)
        self.assertTrue(warnings)

    def test_empty_and_malformed_fields(self):
        source = Rulebook([Rule("a", None), Rule("b", "   "),
                           Rule("c", "Give accurate dates.", [], None, 12, {})])
        clean, _ = validate_rulebook(source, [])
        self.assertEqual(len(clean.rules), 1)
        self.assertEqual(clean.rules[0].scope, "global")
        self.assertEqual(clean.rules[0].applies_when, "")
        self.assertEqual(clean.rules[0].violation, "")
        self.assertEqual(clean.rules[0].boundary, "")

    def test_rule_count_cap(self):
        rules = [Rule(str(i), f"Unique criterion{chr(65 + i)}.") for i in range(20)]
        clean, warnings = validate_rulebook(Rulebook(rules), [])
        self.assertEqual(len(clean.rules), MAX_RULES)
        self.assertEqual([r.rule_id for r in clean.rules], [f"R{i:02d}" for i in range(1, MAX_RULES + 1)])
        self.assertTrue(any("MAX_RULES" in w for w in warnings))

    def test_total_rubric_drops_boundaries_before_rules(self):
        rules = [Rule(str(i), f"Criterion{chr(65 + i)} " + "requirement " * 12,
                      applies_when="condition " * 16, violation="contradiction " * 15,
                      boundary="exception " * 15) for i in range(12)]
        clean, warnings = validate_rulebook(Rulebook(rules), [])
        self.assertLessEqual(len(_rubric_text(clean.rules)), 3500)
        self.assertTrue(any("boundary removed" in w for w in warnings))
        self.assertTrue(any("Lowest-priority rule" in w for w in warnings))
        self.assertTrue(all(not r.boundary for r in clean.rules))
        self.assertTrue(clean.rules[0].text.startswith("CriterionA"))

    def test_provenance_is_preserved_sorted_and_unique(self):
        clean, _ = validate_rulebook(Rulebook([Rule("a", "Give correct amounts.", provenance_ids=["z", "a", "z"])]), [])
        self.assertEqual(clean.rules[0].provenance_ids, ["a", "z"])


class InductionTests(unittest.TestCase):
    def test_happy_path_and_private_provenance(self):
        items = pair_items() + [make_item("unused-good-conversation", label="good", assistant="An accepted alternate answer.")]
        client = FakeLLMClient(happy_responder)
        book, log = induce_rulebook(items, client, Settings())
        self.assertEqual(log["calls"], 2)
        self.assertEqual(len(book.rules), 1)
        self.assertEqual(book.rules[0].rule_id, "R01")
        self.assertEqual(book.rules[0].provenance_ids, sorted(i.item_id for i in items[:2]))
        self.assertEqual(log["candidates"][0]["frequency"], 1)
        self.assertIn("An accepted alternate answer.", client.calls[1]["user"])
        self.assertIn("expert accepted these", client.calls[1]["user"])
        self.assertIn("Unseen requirements may not be covered.", book.notes)
        for call in client.calls:
            self.assertEqual(call["role"], "proposer")
            for item in items:
                self.assertNotIn(item.item_id, call["system"] + call["user"])
        self.assertIn("PAIR 1 — BAD", client.calls[0]["user"])
        self.assertIn("PAIR 1 — GOOD", client.calls[0]["user"])

    def test_deduped_candidate_frequency_and_merged_provenance(self):
        items = [make_item(f"bad-case-{n}", label="bad") for n in range(3)] + [make_item("good-partner", label="good")]
        book, log = induce_rulebook(items, FakeLLMClient(happy_responder), Settings())
        self.assertEqual(len(log["candidates"]), 1)
        self.assertEqual(log["candidates"][0]["frequency"], 3)
        self.assertEqual(book.rules[0].provenance_ids, sorted(i.item_id for i in items))

    def test_batches_are_same_topic_at_most_four_and_bounded(self):
        items = toy_dataset(per_topic_good=1, per_topic_bad=9)
        client = FakeLLMClient(happy_responder)
        induce_rulebook(items, client, Settings())
        stage1 = [c for c in client.calls if c["system"] == STAGE1_SYSTEM]
        self.assertEqual(len(stage1), 12)
        for call in stage1:
            self.assertLessEqual(len(call["user"]), 9000)
            self.assertLessEqual(len(re.findall(r"PAIR \d+ — BAD", call["user"])), 4)
            self.assertEqual(len(set(re.findall(r"Topic: (\w+)", call["user"]))), 1)
        self.assertEqual(sum(len(re.findall(r"PAIR \d+ — BAD", c["user"])) for c in stage1), 36)

    def test_oversized_pair_is_excerpted_with_warning(self):
        items = [make_item("long-case-bad", label="bad", assistant="large content " * 2000),
                 make_item("long-case-good", label="good", assistant="accepted content " * 2000)]
        client = FakeLLMClient(happy_responder)
        book, log = induce_rulebook(items, client, Settings())
        self.assertTrue(book.rules)
        self.assertLessEqual(len(client.calls[0]["user"]), 9000)
        self.assertTrue(any("excerpted" in w for w in log["warnings"]))

    def test_two_bad_consolidations_fall_back(self):
        def responder(role, system, user):
            return happy_responder(role, system, user) if system == STAGE1_SYSTEM else "not JSON"
        client = FakeLLMClient(responder)
        book, log = induce_rulebook(pair_items(), client, Settings())
        self.assertEqual(log["calls"], 3)
        self.assertEqual(book.rules[0].text, "Dates must match the provided context.")
        self.assertIn("Previous response was invalid", client.calls[2]["user"])
        self.assertTrue(any("fallback" in w for w in log["warnings"]))

    def test_unsafe_consolidation_retries_then_uses_clean_candidates(self):
        def responder(role, system, user):
            if system == STAGE1_SYSTEM:
                return happy_responder(role, system, user)
            return json.dumps({"rules": [{"text": "Tell Ana the correct date.",
                                           "merged_candidates": ["C1"]}]})
        client = FakeLLMClient(responder)
        book, log = induce_rulebook(pair_items(), client, Settings())
        self.assertEqual(log["calls"], 3)
        self.assertTrue(book.rules)
        self.assertNotIn("Ana", book.rules[0].text)
        self.assertTrue(any("rubric validation" in warning for warning in log["warnings"]))

    def test_explicit_empty_consolidated_rubric_is_not_a_parse_failure(self):
        def responder(role, system, user):
            if system == STAGE1_SYSTEM:
                return happy_responder(role, system, user)
            return '{"rules":[],"notes":["No generalizable difference was found."]}'
        book, log = induce_rulebook(pair_items(), FakeLLMClient(responder), Settings())
        self.assertFalse(book.rules)
        self.assertEqual(log["calls"], 2)
        self.assertIn("No generalizable difference was found.", book.notes)

    def test_cross_topic_fallback_deduplication_merges_provenance(self):
        items = pair_items("credit") + pair_items("savings")
        def responder(role, system, user):
            return happy_responder(role, system, user) if system == STAGE1_SYSTEM else "invalid"
        book, log = induce_rulebook(items, FakeLLMClient(responder), Settings())
        self.assertEqual(len(book.rules), 1)
        self.assertEqual(book.rules[0].scope, "global")
        self.assertEqual(book.rules[0].provenance_ids, sorted(i.item_id for i in items))
        self.assertEqual(log["calls"], 4)

    def test_unused_good_samples_are_at_most_four_per_topic(self):
        items = pair_items() + [make_item(f"unused-private-{n}", label="good", assistant=f"Accepted alternative {n}.")
                                for n in range(9)]
        client = FakeLLMClient(happy_responder)
        induce_rulebook(items, client, Settings())
        user = client.calls[1]["user"]
        self.assertEqual(user.count("ACCEPTED credit "), 4)
        self.assertNotIn("Accepted alternative 4.", user)
        self.assertNotIn("Accepted alternative 8.", user)

    def test_consolidation_second_attempt_succeeds(self):
        counter = 0
        def responder(role, system, user):
            nonlocal counter
            if system == STAGE2_SYSTEM:
                counter += 1
                if counter == 1:
                    return "{}"
            return happy_responder(role, system, user)
        book, log = induce_rulebook(pair_items(), FakeLLMClient(responder), Settings())
        self.assertEqual(log["calls"], 3)
        self.assertTrue(book.rules)
        self.assertFalse(any("fallback" in w for w in log["warnings"]))

    def test_fallback_ranks_confidence_then_frequency(self):
        items = [make_item(f"rank-bad-{i}", label="bad") for i in range(4)] + [make_item("rank-good", label="good")]
        def responder(role, system, user):
            if system == STAGE2_SYSTEM:
                return "invalid"
            candidates = []
            for number in range(1, 5):
                c = candidate(number, "Always preserve truthful context dates.")
                c["confidence"] = "medium"
                candidates.append(c)
            c = candidate(1, "Ask for missing customer income before recommending loans.")
            c["confidence"] = "high"
            candidates.append(c)
            c = candidate(2, "Never recommend unlawful tax evasion.")
            c["confidence"] = "medium"
            candidates.append(c)
            return json.dumps({"candidates": candidates})
        book, log = induce_rulebook(items, FakeLLMClient(responder), Settings())
        self.assertTrue(book.rules[0].text.startswith("Ask for missing"))
        self.assertTrue(book.rules[1].text.startswith("Always preserve"))
        self.assertTrue(book.rules[2].text.startswith("Never recommend"))

    def test_call_budgets_zero_one_two(self):
        for budget in (0, 1, 2):
            with self.subTest(budget=budget):
                client = FakeLLMClient(happy_responder)
                book, log = induce_rulebook(pair_items(), client, Settings(max_induction_calls=budget))
                self.assertEqual(log["calls"], budget)
                self.assertEqual(len(client.calls), budget)
                self.assertEqual(bool(book.rules), bool(budget))

    def test_stage1_stops_at_budget_and_falls_back(self):
        items = toy_dataset(per_topic_good=1, per_topic_bad=8)
        client = FakeLLMClient(happy_responder)
        book, log = induce_rulebook(items, client, Settings(max_induction_calls=1))
        self.assertEqual(len(client.calls), 1)
        self.assertTrue(book.rules)
        self.assertTrue(any("call budget" in w for w in log["warnings"]))

    def test_no_pairs_or_candidates_makes_no_consolidation_call(self):
        client = FakeLLMClient(happy_responder)
        book, log = induce_rulebook([make_item("orphan-case", label="bad")], client, Settings())
        self.assertFalse(book.rules)
        self.assertFalse(client.calls)
        self.assertTrue(any("no same-topic" in w for w in log["warnings"]))
        client = FakeLLMClient(default='{"candidates":[],"no_clear_rule":[1]}')
        book, log = induce_rulebook(pair_items(), client, Settings())
        self.assertEqual(len(client.calls), 1)
        self.assertFalse(book.rules)
        self.assertIn("No clear contrast reported for 1 pairs.", book.notes)

    def test_terminal_errors_stop_without_futile_retries(self):
        for kind in ("auth", "daily_quota", "deadline"):
            with self.subTest(kind=kind):
                client = FakeLLMClient(lambda r, s, u: LLMResult(False, error_kind=kind))
                book, log = induce_rulebook(toy_dataset(), client, Settings())
                self.assertEqual(len(client.calls), 1)
                self.assertFalse(book.rules)

    def test_expired_deadline(self):
        client = FakeLLMClient(happy_responder)
        book, log = induce_rulebook(pair_items(), client, Settings(deadline_epoch=time.time() - 1))
        self.assertFalse(client.calls)
        self.assertEqual(log["calls"], 0)
        self.assertFalse(book.rules)
        self.assertTrue(any("deadline" in w for w in log["warnings"]))

    def test_malformed_candidates_and_unknown_provenance_are_not_trusted(self):
        def responder(role, system, user):
            if system == STAGE1_SYSTEM:
                return json.dumps({"candidates": [candidate(99), candidate(True), {"pair": 1},
                                                   "not an object", candidate(1)]})
            return json.dumps({"rules": [{"text": "Invented rule.", "merged_candidates": ["C99"]}]})
        book, log = induce_rulebook(pair_items(), FakeLLMClient(responder), Settings())
        self.assertEqual(len(book.rules), 1)
        self.assertNotIn("Invented", book.rules[0].text)
        self.assertEqual(log["calls"], 3)

    def test_malformed_response_and_exception_degrade_cleanly(self):
        def raises(role, system, user):
            raise RuntimeError("Do not retain exception details containing private input.")
        client = FakeLLMClient(raises)
        book, log = induce_rulebook(pair_items(), client, Settings())
        self.assertFalse(book.rules)
        self.assertEqual(log["calls"], 1)
        self.assertNotIn("private input", str(log))
        for reply in ("garbage", "[]", '{"candidates":{}}'):
            book, _ = induce_rulebook(pair_items(), FakeLLMClient(default=reply), Settings())
            self.assertFalse(book.rules)

    def test_customer_specific_candidate_is_not_echoed_to_consolidation(self):
        def responder(role, system, user):
            return json.dumps({"candidates": [candidate(1, "Tell Ana the date.")]})
        client = FakeLLMClient(responder)
        book, _ = induce_rulebook(pair_items(), client, Settings())
        self.assertFalse(book.rules)
        self.assertEqual(len(client.calls), 1)

    def test_context_id_values_are_redacted_but_not_unrelated_substrings(self):
        items = [make_item("redacted-bad-ref", label="bad", context={"ref": "redacted-bad-ref"}),
                 make_item("redacted-good-ref", label="good")]
        client = FakeLLMClient(happy_responder)
        book, _ = induce_rulebook(items, client, Settings())
        self.assertTrue(book.rules)
        self.assertIn("[training reference]", client.calls[0]["user"])
        self.assertNotIn("redacted-bad-ref", str(client.calls))

    def test_split_consolidation_requests_and_good_sample_caps(self):
        items = []
        accepted = {}
        candidates = []
        for topic in TOPICS:
            accepted[topic] = [make_item(f"accepted-{topic}-{n}", topic, "good", assistant="accepted " * 300)
                               for n in range(4)]
            items.extend(accepted[topic])
            for number in range(12):
                candidates.append({"id": f"C{len(candidates) + 1}", "topic": topic,
                                   "text": "requirement " * 12, "applies_when": "condition " * 10,
                                   "violation": "failure " * 16, "boundary": "exception " * 10,
                                   "confidence": "high", "frequency": 1, "topic_specific": False,
                                   "provenance_ids": ["local-only-id"], "pairs": [number], "pair": number})
        warnings = []
        batches = _consolidation_batches(candidates, accepted, items, warnings)
        self.assertEqual(len(batches), 4)
        self.assertTrue(warnings)
        self.assertEqual(sum(len(chunk) for chunk, _ in batches), len(candidates))
        for chunk, user in batches:
            self.assertLessEqual(len(user), 20000)
            self.assertEqual(len({c["topic"] for c in chunk}), 1)
            self.assertEqual(user.count("ACCEPTED "), 4)
            self.assertNotIn("local-only-id", user)
            for item in items:
                self.assertNotIn(item.item_id, user)


if __name__ == "__main__":
    unittest.main()


class GeneralRuleTests(unittest.TestCase):
    def test_prepends_fixed_rule_and_respects_cap(self):
        from rulejudge.contracts import MAX_RULES
        from rulejudge.induction import GENERAL_RULE_ID, add_general_rule
        learned = [Rule(f"R{i:02d}", f"Rule {i}") for i in range(1, MAX_RULES + 1)]
        out = add_general_rule(Rulebook(learned, ["note"]))
        self.assertEqual(out.rules[0].rule_id, GENERAL_RULE_ID)
        self.assertEqual(out.rules[0].scope, "global")
        self.assertEqual(out.rules[0].origin, "fixed_general_check")
        self.assertEqual(len(out.rules), MAX_RULES)
        self.assertEqual(out.rules[-1].rule_id, f"R{MAX_RULES - 1:02d}")
        self.assertEqual(out.notes[0], "note")
        self.assertIn("dropped", out.notes[-1])

    def test_idempotent_and_empty(self):
        from rulejudge.induction import add_general_rule
        once = add_general_rule(Rulebook([]))
        self.assertEqual([r.rule_id for r in once.rules], ["R00"])
        twice = add_general_rule(once)
        self.assertEqual([r.rule_id for r in twice.rules], ["R00"])
        self.assertEqual(twice.notes, [])
