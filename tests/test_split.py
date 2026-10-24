"""WP5 split tests use unique synthetic content except explicit duplicate cases."""
import hashlib
import unittest
from dataclasses import replace

from rulejudge.contracts import TOPICS
from rulejudge.split import split_fs
from tests.fixtures import make_item, toy_dataset


def rank(item_id, seed="rulejudge-v1"):
    return hashlib.sha256((seed + item_id).encode("utf-8")).hexdigest(), item_id


class SplitTest(unittest.TestCase):
    def test_determinism_stratification_and_hash_order(self):
        items = toy_dataset(5, 5)
        f_ids, s_ids = split_fs(items)
        self.assertEqual(split_fs(list(reversed(items))), (f_ids, s_ids))
        self.assertFalse(set(f_ids) & set(s_ids))
        self.assertEqual(set(f_ids) | set(s_ids), {i.item_id for i in items})
        self.assertEqual((len(f_ids), len(s_ids)), (32, 8))
        for topic in TOPICS:
            for label in ("good", "bad"):
                stratum = [i.item_id for i in items if (i.topic, i.label) == (topic, label)]
                expected = min(stratum, key=rank)
                self.assertEqual(set(s_ids) & set(stratum), {expected})

    def test_rounding_and_minimum_one_for_three_items(self):
        for n, fraction, expected_s in [(3, 0.0, 1), (3, 0.2, 1), (5, 0.5, 2), (6, 0.5, 3)]:
            with self.subTest(n=n, fraction=fraction):
                items = toy_dataset(n, n)
                f_ids, s_ids = split_fs(items, fraction)
                self.assertEqual(len(s_ids), 8 * expected_s)
                self.assertEqual(len(f_ids), 8 * (n - expected_s))

    def test_small_strata_fall_back(self):
        items = toy_dataset(2, 2)
        f_ids, s_ids = split_fs(items)
        self.assertEqual(set(f_ids), {i.item_id for i in items})
        self.assertEqual(s_ids, [])

    def test_missing_topic_in_either_partition_falls_back(self):
        cases = [toy_dataset(5, 5)[:-9], toy_dataset(3, 3)]
        for items, fraction in [(cases[0], 0.2), (cases[1], 1.0)]:
            with self.subTest(fraction=fraction):
                f_ids, s_ids = split_fs(items, fraction)
                self.assertEqual(set(f_ids), {i.item_id for i in items})
                self.assertEqual(s_ids, [])

    def test_only_topics_present_need_coverage(self):
        items = [make_item(str(i), label="good", assistant=f"Unique answer {i}") for i in range(5)]
        f_ids, s_ids = split_fs(items)
        self.assertEqual((len(f_ids), len(s_ids)), (4, 1))

    def test_duplicates_stay_together_even_with_conflicting_labels(self):
        items = toy_dataset(8, 8)
        original = items[0]
        duplicates = [replace(original, item_id="copy-good"),
                      replace(original, item_id="copy-bad", label="bad")]
        items += duplicates
        f_ids, s_ids = split_fs(items)
        self.assertTrue(s_ids)  # Exercise an actual split rather than just the safe fallback.
        for duplicate in duplicates:
            self.assertEqual(original.item_id in f_ids, duplicate.item_id in f_ids)
        self.assertEqual(split_fs(items), split_fs(list(reversed(items))))

    def test_context_key_order_does_not_split_duplicates(self):
        items = toy_dataset(8, 8)
        original = make_item("canonical-a", label="good", context={"a": 1, "b": 2})
        duplicate = replace(original, item_id="canonical-b", context={"b": 2, "a": 1})
        f_ids, s_ids = split_fs(items + [original, duplicate])
        self.assertTrue(s_ids)
        self.assertEqual("canonical-a" in f_ids, "canonical-b" in f_ids)

    def test_identical_topic_content_cannot_leak(self):
        items = [make_item(f"{topic}-{n}", topic, "good") for topic in TOPICS for n in range(5)]
        f_ids, s_ids = split_fs(items)
        self.assertEqual(set(f_ids), {i.item_id for i in items})
        self.assertEqual(s_ids, [])

    def test_seed_changes_membership_but_not_stratum_counts(self):
        items = toy_dataset(10, 10)
        a, b = split_fs(items, seed="first"), split_fs(items, seed="second")
        self.assertNotEqual(set(a[1]), set(b[1]))
        self.assertEqual(len(a[1]), len(b[1]))

    def test_empty_and_invalid_parameters(self):
        self.assertEqual(split_fs([]), ([], []))
        for fraction in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    split_fs(toy_dataset(), fraction)
        duplicate = make_item("same", label="good")
        with self.assertRaises(ValueError):
            split_fs([duplicate, duplicate])


if __name__ == "__main__":
    unittest.main()
