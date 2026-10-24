"""WP5 decision tests: synthetic, table-driven and free of model calls."""
import sys
import unittest
from pathlib import Path

from rulejudge.contracts import Assessment, HeadParams, TOPICS
from rulejudge.decision import (apply_floor, decide, fit_fitted_head, refit_final,
                                rule_stats, select_head, topic_macro_accuracy)
from tests.fixtures import make_item

PRIORS = {"by_topic": {"credit": "bad"}, "global": "good"}


def assessed(item, support, status="ok"):
    return Assessment(item.item_id, status, support)


class DecisionTest(unittest.TestCase):
    def test_decide_truth_table(self):
        item = make_item("i", label="good")
        # status, support, head, label, source, triggered
        cases = [
            ("ok", {"R01": "na"}, HeadParams("fixed"), "good", "model", []),
            ("ok", {}, HeadParams("fixed"), "good", "model", []),
            ("ok", {"R01": 2}, HeadParams("fixed"), "bad", "model", ["R01"]),
            ("partial", {"R01": 2}, HeadParams("fixed"), "bad", "partial_model", ["R01"]),
            ("partial", {"R01": 1}, HeadParams("fixed"), "bad", "fallback_prior", []),
            ("failed", {"R01": 3}, HeadParams("fixed"), "bad", "fallback_prior", []),
            (None, {}, HeadParams("fixed"), "bad", "fallback_prior", []),
            ("ok", {"R01": 3}, HeadParams("fixed", disabled=["R01"]), "good", "model", []),
            ("ok", {"R01": 1}, HeadParams("fixed", 2), "good", "model", []),
            ("ok", {"R01": 2}, HeadParams("fixed", 3), "good", "model", []),
            ("ok", {"R01": 3}, HeadParams("fixed", 3), "bad", "model", ["R01"]),
            ("ok", {"R01": 1}, HeadParams("fixed", 1), "bad", "model", ["R01"]),
            ("ok", {"R01": "3", "R02": True}, HeadParams("fixed", 1), "good", "model", []),
            ("ok", {"R01": 0}, HeadParams("prior"), "bad", "prior_mode", []),
        ]
        for status, support, head, label, source, triggered in cases:
            with self.subTest(status=status, support=support, head=head):
                assessment = None if status is None else assessed(item, support, status)
                result = decide(item, assessment, head, PRIORS)
                self.assertEqual((result.item_id, result.label, result.source, result.triggered),
                                 ("i", label, source, triggered))

    def test_prior_global_default_and_partial_nontrigger(self):
        item = make_item("s", topic="savings")
        for priors, expected in [(PRIORS, "good"), ({"global": "bad"}, "bad"), ({}, "good")]:
            with self.subTest(priors=priors):
                self.assertEqual(decide(item, None, HeadParams("fixed"), priors).label, expected)
                self.assertEqual(decide(item, assessed(item, {}, "partial"),
                                        HeadParams("fixed"), priors).label, expected)

    def test_triggers_are_sorted_and_noncompensating(self):
        item = make_item("i")
        a = assessed(item, {"R03": 3, "R01": 2, "R02": 0, "R04": "na"})
        d = decide(item, a, HeadParams("fixed"), PRIORS)
        self.assertEqual((d.label, d.triggered), ("bad", ["R01", "R03"]))

    def test_metric_matches_official_score(self):
        # The organizer's module intentionally imports common from its src path.
        src = str(Path(__file__).resolve().parents[1] / "src")
        sys.path.insert(0, src)
        try:
            from evaluate import score
            items = []
            for index, topic in enumerate(TOPICS):
                for n in range(index + 2):
                    items.append(make_item(f"{topic}-{n}", topic, "bad" if n % 2 else "good"))
            predictions = {i.item_id: ("bad" if n % 3 else "good") for n, i in enumerate(items)}
            official = score(
                [{"item_id": i.item_id, "topic": i.topic, "label": i.label} for i in items],
                [{"item_id": iid, "label": label} for iid, label in predictions.items()])
        finally:
            sys.path.remove(src)
        ours = topic_macro_accuracy(items, predictions)
        self.assertEqual(ours["macro"], official["score"])
        self.assertEqual(ours["n"], official["n"])
        self.assertTrue(ours["all_topics_present"])
        for topic in TOPICS:
            for key in ("n", "correct", "accuracy"):
                self.assertEqual(ours["by_topic"][topic][key], official["by_topic"][topic][key])

    def test_metric_is_macro_not_micro(self):
        items = [make_item(f"c{i}", "credit", "good") for i in range(9)]
        items.append(make_item("s", "savings", "bad"))
        result = topic_macro_accuracy(items, {i.item_id: "good" for i in items})
        self.assertEqual(result["macro"], 0.5)
        self.assertFalse(result["all_topics_present"])
        self.assertEqual(result["n"], 10)

    def test_empty_metric(self):
        self.assertEqual(topic_macro_accuracy([], {}),
                         {"macro": 0.0, "by_topic": {}, "n": 0, "all_topics_present": False})

    def test_metric_rejects_invalid_labels_and_misalignment(self):
        item = make_item("i", label="good")
        cases = [([item], {}), ([item], {"i": "BAD"}),
                 ([make_item("i")], {"i": "good"}),
                 ([make_item("i", topic="other", label="good")], {"i": "good"}),
                 ([item, item], {"i": "good"}), ([item], {"i": "good", "extra": "good"})]
        for items, predictions in cases:
            with self.subTest(items=items, predictions=predictions):
                with self.assertRaises(ValueError):
                    topic_macro_accuracy(items, predictions)

    def test_fit_disables_noise_and_keeps_genuine_rule(self):
        items = [make_item(f"g{i}", label="good") for i in range(4)]
        items += [make_item(f"b{i}", label="bad") for i in range(3)]
        assessments = {i.item_id: assessed(i, {"noise": 3 if i.label == "good" else 0,
                                               "real": 3 if i.label == "bad" else 0}) for i in items}
        head = fit_fitted_head(items, assessments, PRIORS)
        self.assertEqual((head.kind, head.tau, head.disabled, head.fit_on),
                         ("fitted", 2, ["noise"], "F"))

    def test_fit_three_trigger_minimum_and_precision_gate(self):
        cases = [(2, 0, []), (2, 2, []), (3, 1, ["R01"])]
        for good, bad, disabled in cases:
            with self.subTest(good=good, bad=bad):
                items = [make_item(f"g{i}", label="good") for i in range(good)]
                items += [make_item(f"b{i}", label="bad") for i in range(bad)]
                assessments = {i.item_id: assessed(i, {"R01": 3}) for i in items}
                self.assertEqual(fit_fitted_head(items, assessments, PRIORS).disabled, disabled)

    def test_fit_disabling_cap_and_rule_id_ties(self):
        items = [make_item(f"g{i}", label="good") for i in range(3)]
        assessments = {i.item_id: assessed(i, {f"R{r:02}": 3 for r in (4, 3, 2, 1)}) for i in items}
        head = fit_fitted_head(items, assessments, PRIORS)
        self.assertEqual((head.tau, head.disabled), (2, ["R01", "R02", "R03"]))

    def test_fit_tie_prefers_lower_precision(self):
        # Neither single disable changes accuracy because the rules overlap on good rows.
        items = [make_item(f"g{i}", label="good") for i in range(3)] + [make_item("b", label="bad")]
        assessments = {i.item_id: assessed(i, {"R01": 3, "R02": 3 if i.label == "good" else 0})
                       for i in items}
        self.assertEqual(fit_fitted_head(items, assessments, PRIORS).disabled, ["R02", "R01"])

    def test_fit_does_not_disable_if_macro_accuracy_decreases(self):
        # Precision is low overall, but the rule is essential to a small topic's accuracy.
        items = [make_item(f"g{i}", "credit", "good") for i in range(10)]
        items += [make_item("b", "savings", "bad")]
        assessments = {i.item_id: assessed(i, {"R01": 3 if i.item_id in {"g0", "g1", "g2", "b"} else 0})
                       for i in items}
        self.assertEqual(fit_fitted_head(items, assessments, {"global": "good"}).disabled, [])

    def test_fit_uses_partial_not_failed_for_precision(self):
        items = [make_item(f"g{i}", label="good") for i in range(3)]
        cases = [("partial", ["R01"]), ("failed", [])]
        for status, expected in cases:
            with self.subTest(status=status):
                aa = {i.item_id: assessed(i, {"R01": 3}, status) for i in items}
                self.assertEqual(fit_fitted_head(items, aa, {"global": "good"}).disabled, expected)

    def test_fit_threshold_selection_and_empty_ties(self):
        cases = [([("good", 1), ("bad", 2)], 2),
                 ([("good", 2), ("bad", 3)], 3),
                 ([("good", 0), ("bad", 1)], 1), ([], 2)]
        for labels_supports, expected_tau in cases:
            with self.subTest(rows=labels_supports):
                items = [make_item(str(n), label=label) for n, (label, _) in enumerate(labels_supports)]
                aa = {i.item_id: assessed(i, {"R01": labels_supports[n][1]}) for n, i in enumerate(items)}
                head = fit_fitted_head(items, aa, PRIORS, fit_on="F+S")
                self.assertEqual((head.tau, head.fit_on), (expected_tau, "F+S"))

    def test_selection_incumbent_ties_and_best_score(self):
        item = make_item("s", label="good")
        fixed, fitted = HeadParams("fixed", 2), HeadParams("fitted", 3)
        for support, expected in [(0, fixed), (2, fitted)]:
            with self.subTest(support=support):
                chosen, report = select_head([fixed, fitted], [item],
                                             {"s": assessed(item, {"R01": support})}, PRIORS)
                self.assertIs(chosen, expected)
                self.assertEqual(len(report["candidates"]), 2)
                self.assertIn("S_macro", report["candidates"][0])
        chosen, report = select_head([fixed, fitted], [], {}, PRIORS)
        self.assertIs(chosen, fitted)
        self.assertEqual(report["reason"], "no_S")
        self.assertIsNone(report["candidates"][0]["S_macro"])
        with self.assertRaises(ValueError):
            select_head([], [item], {}, PRIORS)

    def test_floor_requires_strict_improvement_on_both(self):
        # With a good prior and gold good, support 3 loses; support 0 ties.
        f, s = make_item("f", label="good"), make_item("s", label="good")
        head = HeadParams("fixed")
        cases = [(3, 3, True), (3, 0, False), (0, 3, False), (0, 0, False)]
        for f_support, s_support, switches in cases:
            with self.subTest(f=f_support, s=s_support):
                aa = {"f": assessed(f, {"R01": f_support}), "s": assessed(s, {"R01": s_support})}
                chosen, report = apply_floor(head, [f], [s], aa, {"global": "good"})
                self.assertEqual(chosen.kind == "prior", switches)
                self.assertEqual(report["applied"], switches)
                if not switches:
                    self.assertIs(chosen, head)

    def test_floor_empty_s_uses_f_only(self):
        f = make_item("f", label="good")
        for support, expected in [(3, "prior"), (0, "fixed")]:
            with self.subTest(support=support):
                chosen, report = apply_floor(HeadParams("fixed"), [f], [],
                                             {"f": assessed(f, {"R01": support})}, {"global": "good"})
                self.assertEqual(chosen.kind, expected)
                self.assertIsNone(report["S_prior"])

    def test_floor_rejects_a_prior_that_wins_only_one_partition(self):
        head = HeadParams("fixed")
        for f_label, s_label in [("bad", "good"), ("good", "bad")]:
            with self.subTest(f_label=f_label, s_label=s_label):
                f, s = make_item("f", label=f_label), make_item("s", label=s_label)
                aa = {i.item_id: assessed(i, {"R01": 3}) for i in (f, s)}
                chosen, report = apply_floor(head, [f], [s], aa, {"global": "good"})
                self.assertIs(chosen, head)
                self.assertFalse(report["applied"])
        chosen, report = apply_floor(head, [], [], {}, {"global": "good"})
        self.assertIs(chosen, head)
        self.assertFalse(report["applied"])

    def test_refit_only_fitted_and_does_not_mutate(self):
        items = [make_item(f"g{i}", label="good") for i in range(3)]
        aa = {i.item_id: assessed(i, {"R01": 3}) for i in items}
        for kind in ("fixed", "prior", "fitted"):
            with self.subTest(kind=kind):
                head = HeadParams(kind)
                result = refit_final(head, items, aa, PRIORS)
                if kind == "fitted":
                    self.assertEqual((result.fit_on, result.disabled), ("F+S", ["R01"]))
                    self.assertEqual(head.disabled, [])
                else:
                    self.assertIs(result, head)

    def test_rule_stats_counts_and_disabled_diagnostics(self):
        items = [make_item("b", label="bad"), make_item("g", label="good"),
                 make_item("p", label="good"), make_item("f", label="bad"), make_item("u")]
        aa = {"b": assessed(items[0], {"R01": 3, "R02": 2}),
              "g": assessed(items[1], {"R01": 1, "R02": "na"}),
              "p": assessed(items[2], {"R01": 2, "R02": 3}, "partial"),
              "f": assessed(items[3], {"R01": 3}, "failed"),
              "u": assessed(items[4], {"R01": 3})}
        stats = rule_stats(items, aa, HeadParams("fixed", disabled=["R02"]))
        self.assertEqual(stats["R01"], {"applicable": 3, "triggered": 2, "triggered_bad": 1,
                                        "triggered_good": 1, "sole_triggered": 2,
                                        "precision": 0.5, "disabled": False})
        self.assertEqual(stats["R02"], {"applicable": 2, "triggered": 2, "triggered_bad": 1,
                                        "triggered_good": 1, "sole_triggered": 0,
                                        "precision": 0.5, "disabled": True})

    def test_fit_and_stats_are_permutation_invariant(self):
        items = [make_item(f"g{i}", label="good") for i in range(5)]
        items += [make_item(f"b{i}", label="bad") for i in range(3)]
        aa = {i.item_id: assessed(i, {"R02": 3 if i.label == "bad" else 0,
                                      "R01": 3 if i.label == "good" else 0}) for i in items}
        forward = fit_fitted_head(items, aa, PRIORS)
        reversed_aa = {iid: Assessment(a.item_id, a.status, dict(reversed(list(a.support.items()))))
                       for iid, a in reversed(list(aa.items()))}
        self.assertEqual(forward, fit_fitted_head(list(reversed(items)), reversed_aa, PRIORS))
        self.assertEqual(rule_stats(items, aa, forward),
                         rule_stats(list(reversed(items)), reversed_aa, forward))

    def test_rule_stats_preserves_requested_but_missing_rules(self):
        item = make_item("i", label="good")
        assessment = assessed(item, {"R01": 0}, "partial")
        assessment.requested_rule_ids = ["R01", "R02"]
        stats = rule_stats([item], {"i": assessment}, HeadParams("fixed"))
        self.assertEqual(stats["R02"]["applicable"], 0)
        self.assertEqual(stats["R02"]["triggered"], 0)
        self.assertIsNone(stats["R02"]["precision"])

    def test_rule_stats_na_and_zero_trigger_precision(self):
        item = make_item("i", label="good")
        stats = rule_stats([item], {"i": assessed(item, {"R01": "na", "R02": 0})}, HeadParams("fixed"))
        self.assertEqual(stats["R01"]["applicable"], 0)
        self.assertIsNone(stats["R01"]["precision"])
        self.assertEqual(stats["R02"]["applicable"], 1)
        self.assertEqual(stats["R02"]["triggered"], 0)


if __name__ == "__main__":
    unittest.main()
