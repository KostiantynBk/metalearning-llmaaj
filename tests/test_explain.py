"""Offline WP6 tests; reports are constructed by hand, not via WP5 stubs."""
from __future__ import annotations

import copy
import unittest

from rulejudge.contracts import Assessment, Evidence, HeadParams, Rule, Rulebook, TOPICS
from rulejudge.explain import (
    _consistency_issues, render_explanation, render_prior_only_explanation, select_examples,
)
from tests.fixtures import make_item, toy_dataset, toy_rulebook


def make_report():
    items = toy_dataset()
    book = toy_rulebook()
    head = HeadParams("fitted", tau=2, disabled=["R02"], fit_on="F+S")
    assessments = {}
    for item in items:
        support = {"R01": 3 if item.label == "bad" else 0,
                   "R02": "na" if item.label == "bad" else 3}
        if item.topic == "credit":
            support["R03"] = 1 if item.item_id == "cre-g1" else "na"
        if item.item_id == "cre-g2":
            support["R01"] = 2
        if item.item_id == "fin-b1":
            support["R01"] = 1
        if item.item_id == "sav-b1":
            support["R01"] = "na"
        status = "failed" if item.item_id == "spe-b1" else "partial" if item.item_id == "cre-b1" else "ok"
        evidence = [Evidence(rid, value, ["msg 2"], "Real toy assessor rationale.")
                    for rid, value in support.items() if type(value) is int and value >= 1]
        assessments[item.item_id] = Assessment(item.item_id, status, support, evidence,
                                                requested_rule_ids=list(support))
    report = {
        "run_info": {"run_id": "toy-run", "created_utc": "2026-10-24T12:20:00Z",
                     "train_fingerprint": "toy-fingerprint", "n_items": len(items),
                     "counts": {t: {"good": 3, "bad": 2} for t in TOPICS},
                     "models": {"assessor": "toy-assessor", "proposer": "toy-proposer"},
                     "language_note": "Synthetic English examples."},
        "rulebook": book, "head": head,
        "selection": {"candidates": [
            {"kind": "fixed", "tau": 2, "disabled": [], "F_macro": .5, "S_macro": .5},
            {"kind": "fitted", "tau": 2, "disabled": ["R02"], "F_macro": .8, "S_macro": .75}],
            "chosen": "fitted", "floor": {"switched": False}, "f_size": 16, "s_size": 4},
        "rule_stats": {
            "R01": {"applicable": 18, "triggered": 6, "triggered_bad": 5, "triggered_good": 1,
                    "sole_triggered": 6, "precision": 5 / 6, "disabled": False},
            "R02": {"applicable": 12, "triggered": 12, "triggered_bad": 0, "triggered_good": 12,
                    "sole_triggered": 0, "precision": 0, "disabled": True},
            "R03": {"applicable": 1, "triggered": 0, "triggered_bad": 0, "triggered_good": 0,
                    "sole_triggered": 0, "precision": None, "disabled": False}},
        "metrics": {"F": {"macro": .8}, "S": {"macro": .75},
                    "S_prior": {"macro": .5}, "all_final": {"macro": .8}},
        "fallbacks": {"failed": 1, "partial": 1}, "induction_notes": ["Toy rule coverage is incomplete."],
        "hashes": {"rulebook.json": "rulebook-hash", "head.json": "head-hash", "assessor_template": "template-hash"},
        "items_by_id": {item.item_id: item for item in items}, "assessments": assessments,
    }
    report["examples"] = select_examples(items, assessments, head, book)
    return report


class SelectExamplesTests(unittest.TestCase):
    def setUp(self):
        self.report = make_report()

    def test_categories_use_gold_labels_and_real_support(self):
        examples = self.report["examples"]
        self.assertEqual([r["item_id"] for r in examples["R01"]["supporting"]], ["cre-b0", "cre-b1"])
        self.assertTrue(all(r["label"] == "bad" and r["support"] == 3 and r["triggered"]
                            for r in examples["R01"]["supporting"]))
        alarm = examples["R01"]["false_alarm"][0]
        self.assertEqual((alarm["item_id"], alarm["label"], alarm["support"]), ("cre-g2", "good", 2))
        self.assertEqual(examples["R03"]["boundary_pass"][0]["item_id"], "cre-g1")
        self.assertEqual({r["item_id"] for r in examples["missed"]}, {"fin-b1", "sav-b1", "spe-b1"})
        self.assertEqual([r["item_id"] for r in examples["no_rule_applied"]], ["sav-b1"])
        self.assertEqual(_consistency_issues(self.report), [])

    def test_support_descending_and_id_ties(self):
        items = [make_item("z", label="bad"), make_item("a", label="bad"), make_item("b", label="bad")]
        assessments = {i.item_id: Assessment(i.item_id, "ok", {"R01": 2 if i.item_id == "a" else 3}) for i in items}
        rows = select_examples(items, assessments, HeadParams("fixed"), toy_rulebook())["R01"]["supporting"]
        self.assertEqual([r["item_id"] for r in rows], ["b", "z"])

    def test_boundary_prefers_support_one(self):
        items = [make_item("a", label="good"), make_item("z", label="good")]
        assessments = {i.item_id: Assessment(i.item_id, "ok", {"R01": 0 if i.item_id == "a" else 1}) for i in items}
        rows = select_examples(items, assessments, HeadParams("fixed"), toy_rulebook())["R01"]["boundary_pass"]
        self.assertEqual([r["item_id"] for r in rows], ["z", "a"])

    def test_disabled_rules_never_supply_trigger_examples(self):
        self.assertEqual(self.report["examples"]["R02"], {"supporting": [], "false_alarm": [], "boundary_pass": []})
        for row in self.report["examples"]["missed"]:
            self.assertNotIn("R02", row["triggered_rules"])

    def test_failed_support_never_triggers(self):
        rows = self.report["examples"]["missed"]
        row = next(r for r in rows if r["item_id"] == "spe-b1")
        self.assertFalse(row["triggered"])
        self.assertEqual(row["evidence"], [])

    def test_partial_trigger_is_valid_supporting_example(self):
        rows = self.report["examples"]["R01"]["supporting"]
        self.assertEqual(next(r for r in rows if r["item_id"] == "cre-b1")["assessment_status"], "partial")

    def test_deterministic_independent_of_input_order(self):
        r = self.report
        actual = select_examples(list(reversed(list(r["items_by_id"].values()))),
                                 dict(reversed(list(r["assessments"].items()))), r["head"], r["rulebook"])
        self.assertEqual(actual, r["examples"])

    def test_per_rule_cap_does_not_cap_global_counts(self):
        r = self.report
        examples = select_examples(r["items_by_id"].values(), r["assessments"], r["head"], r["rulebook"], per_rule=0)
        self.assertEqual(examples["R01"]["supporting"], [])
        self.assertEqual(len(examples["missed"]), 3)

    def test_empty_complete_assessment_counts_as_no_rule_applied(self):
        item = make_item("empty", label="good")
        assessment = Assessment("empty", "ok", {})
        examples = select_examples([item], {"empty": assessment}, HeadParams("fixed"), Rulebook([]))
        self.assertEqual([r["item_id"] for r in examples["no_rule_applied"]], ["empty"])

    def test_partial_all_na_is_not_claimed_complete(self):
        item = make_item("partial", label="good")
        assessment = Assessment("partial", "partial", {"R01": "na"})
        examples = select_examples([item], {"partial": assessment}, HeadParams("fixed"), toy_rulebook())
        self.assertEqual(examples["no_rule_applied"], [])

    def test_unlabelled_missing_and_wrong_scope_items_not_examples(self):
        items = [make_item("unlabelled"), make_item("missing", label="bad"),
                 make_item("savings", topic="savings", label="bad")]
        assessments = {"unlabelled": Assessment("unlabelled", "ok", {"R01": 3}),
                       "savings": Assessment("savings", "ok", {"R03": 3})}
        examples = select_examples(items, assessments, HeadParams("fixed"), toy_rulebook())
        self.assertEqual(examples["R01"]["supporting"], [])
        self.assertEqual(examples["R03"]["supporting"], [])

    def test_boolean_is_not_integer_support(self):
        item = make_item("bool", label="bad")
        assessments = {"bool": Assessment("bool", "ok", {"R01": True})}
        examples = select_examples([item], assessments, HeadParams("fixed", tau=1), toy_rulebook())
        self.assertEqual(examples["R01"]["supporting"], [])


class ExplanationTests(unittest.TestCase):
    def setUp(self):
        self.report = make_report()

    def test_all_five_jury_sections_and_run_identifiers(self):
        text = render_explanation(self.report)
        for heading in ("1. Decision rule and priorities", "2. Rules learned from this training set",
                        "3. Training evidence", "4. Boundaries and limitations", "5. Consistency with the implementation"):
            self.assertIn("## " + heading, text)
        for value in ("toy-run", "toy-fingerprint", "R01", "R02", "R03", "τ = 2", "Disabled rules: R02",
                      "rulebook-hash", "head-hash", "template-hash", "toy-assessor"):
            self.assertIn(value, text)
        self.assertIn("0.7500", text)
        self.assertIn("0.5000", text)
        self.assertIn("runtime/rulebook.json + head.json", text)

    def test_examples_state_real_label_rule_and_output(self):
        text = render_explanation(self.report)
        self.assertIn("Expert label: **BAD (provided)**", text)
        self.assertIn("Expert label: **GOOD (provided)**", text)
        self.assertIn("model-inferred from labels, not stated by the expert", text)
        self.assertIn("Pipeline output: bad — active veto: R01", text)
        self.assertIn("R01 support: 3; active trigger: yes", text)
        self.assertIn("R03 support: 1; active trigger: no", text)
        self.assertIn("Real toy assessor rationale.", text)
        self.assertEqual(_consistency_issues(self.report), [])

    def test_no_evidence_is_invented_when_categories_empty(self):
        text = render_explanation(self.report)
        self.assertIn("no supporting training example found", text)
        report = make_report()
        for assessment in report["assessments"].values():
            assessment.support = {r: "na" for r in assessment.support}
            assessment.evidence = []
        del report["examples"]
        text = render_explanation(report)
        self.assertIn("no boundary or false-alarm training example found", text)
        self.assertNotIn("Real toy assessor rationale.", text)

    def test_no_full_prompts_or_more_than_300_char_message_excerpts(self):
        item = make_item("long", label="bad", user="U" * 500, assistant="A" * 500)
        assessment = Assessment("long", "ok", {"R01": 3})
        r = self.report
        r["items_by_id"] = {"long": item}
        r["assessments"] = {"long": assessment}
        del r["examples"]
        text = render_explanation(r)
        user_line = next(line for line in text.splitlines() if line.startswith("> User: "))
        assistant_line = next(line for line in text.splitlines() if line.startswith("> Assistant: "))
        self.assertEqual(len(user_line.removeprefix("> User: ")), 300)
        self.assertEqual(len(assistant_line.removeprefix("> Assistant: ")), 300)
        self.assertNotIn("<<<MESSAGES", text)

    def test_untrusted_markdown_html_and_newlines_are_escaped(self):
        item = make_item("unsafe", label="bad", user="<script>\n**do this**|[link](https://evil.invalid)",
                         assistant="> outside\n# forged heading")
        r = self.report
        r["items_by_id"] = {"unsafe": item}
        r["assessments"] = {"unsafe": Assessment("unsafe", "ok", {"R01": 3})}
        del r["examples"]
        text = render_explanation(r)
        self.assertNotIn("<script>", text)
        self.assertNotIn("\n# forged", text)
        self.assertIn("&lt;script&gt;", text)
        self.assertIn("\\*\\*do this\\*\\*\\|\\[link\\]", text)

    def test_inconsistent_labels_and_trigger_status_are_detected_and_rejected(self):
        for key, value in (("label", "good"), ("support", 0), ("triggered", False),
                           ("excerpt", "Invented conversation"), ("evidence", [])):
            with self.subTest(key=key):
                report = copy.deepcopy(self.report)
                report["examples"]["R01"]["supporting"][0][key] = value
                self.assertTrue(_consistency_issues(report))
                with self.assertRaisesRegex(ValueError, "Inconsistent training evidence"):
                    render_explanation(report)

    def test_invented_item_and_wrong_category_are_rejected(self):
        report = copy.deepcopy(self.report)
        report["examples"]["R01"]["supporting"][0]["item_id"] = "invented"
        self.assertTrue(_consistency_issues(report))
        report = copy.deepcopy(self.report)
        report["examples"]["R01"]["supporting"] = report["examples"]["R01"]["false_alarm"]
        self.assertTrue(_consistency_issues(report))

    def test_disabled_examples_cannot_masquerade_as_active(self):
        report = copy.deepcopy(self.report)
        report["examples"]["R02"]["false_alarm"] = [{"item_id": "cre-g0"}]
        self.assertTrue(_consistency_issues(report))

    def test_counts_use_uncapped_assessments_not_selected_example_lists(self):
        self.report["examples"]["missed"] = []
        self.report["examples"]["no_rule_applied"] = []
        text = render_explanation(self.report)
        self.assertIn("- financial\\_position: 1", text)
        self.assertIn("- savings: 1", text)
        self.assertIn("- spending: 1", text)
        self.assertIn("No rule applied (complete all-na or empty assessments): 1 items", text)
        self.assertIn("False alarms: 1 distinct", text)

    def test_prior_head_never_claims_rule_veto_determines_predictions(self):
        self.report["head"] = HeadParams("prior")
        del self.report["examples"]
        text = render_explanation(self.report)
        self.assertIn("Prior-only mode", text)
        self.assertIn("No rule veto is used", text)
        self.assertNotIn("Pipeline output: bad — active veto", text)
        self.assertIn("Pipeline output: good — prior-only head", text)

    def test_partial_without_trigger_uses_majority_prior(self):
        self.report["assessments"]["fin-b1"].status = "partial"
        self.report["examples"] = select_examples(self.report["items_by_id"].values(), self.report["assessments"],
                                                  self.report["head"], self.report["rulebook"])
        text = render_explanation(self.report)
        self.assertIn("Pipeline output: good — topic-majority fallback", text)

    def test_valid_pointers_only_and_no_fabricated_rationale(self):
        a = self.report["assessments"]["cre-b0"]
        a.evidence[0].where += ["msg 99", "context.missing"]
        self.report["examples"] = select_examples(self.report["items_by_id"].values(), self.report["assessments"],
                                                  self.report["head"], self.report["rulebook"])
        text = render_explanation(self.report)
        self.assertNotIn("msg 99", text)
        self.assertNotIn("context.missing", text)
        self.assertEqual(_consistency_issues(self.report), [])

    def test_unknown_metrics_and_hashes_not_invented(self):
        del self.report["metrics"]
        del self.report["hashes"]
        text = render_explanation(self.report)
        self.assertIn("Pre-refit S macro: not recorded", text)
        self.assertIn("rulebook.json: not recorded", text)

    def test_empty_report_has_no_fabricated_examples(self):
        report = {"head": HeadParams("fixed"), "rulebook": Rulebook([]), "items_by_id": {}, "assessments": {}}
        text = render_explanation(report)
        self.assertIn("No learned rules in this bundle", text)
        self.assertIn("no missed training example found", text)
        self.assertNotIn("Expert label:", text)


class PriorExplanationTests(unittest.TestCase):
    def test_prior_only_is_explicit_and_shows_per_topic_labels(self):
        priors = {"by_topic": {"credit": "good", "savings": "bad"}, "global": "good",
                  "counts": {"credit": {"good": 3, "bad": 2}, "savings": {"good": 1, "bad": 4}}}
        text = render_prior_only_explanation({"run_id": "initial", "train_fingerprint": "abc"}, priors, "API unavailable")
        for value in ("No rules were learned", "API unavailable", "| credit | good | 3 / 2 |",
                      "| savings | bad | 1 / 4 |", "Unseen-topic fallback: good", "Ties go to good"):
            self.assertIn(value, text)
        self.assertNotIn("Expert label:", text)

    def test_prior_reason_escaped(self):
        text = render_prior_only_explanation({}, {}, "<error>\n# fake section")
        self.assertIn("&lt;error&gt; \\# fake section", text)
        self.assertNotIn("\n# fake", text)


if __name__ == "__main__":
    unittest.main()
