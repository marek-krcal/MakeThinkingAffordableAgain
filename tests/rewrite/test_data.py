import unittest

from tc.data import select, strict_pairs, union


class DataTests(unittest.TestCase):
    def test_selection_preserves_identity_and_excludes_test(self):
        rows = [{"id": str(i), "split": "train", "seed_index": 100 + i} for i in range(8)]
        rows.append({"id": "test", "split": "test"})
        result = select(rows, exclude=["0"], count=4, seed=7)
        self.assertEqual(result, select(rows, exclude=["0"], count=4, seed=7))
        self.assertEqual(len(result), 4)
        self.assertTrue(all(r["id"] not in ("0", "test") for r in result))
        self.assertTrue(all(r["seed_index"] == 100 + int(r["id"]) for r in result))

    def test_union_conflicts_and_order(self):
        a, b = {"id": "a"}, {"id": "b"}
        self.assertEqual(union([[b, a], [b]]), [b, a])
        with self.assertRaises(ValueError):
            union([[a], [{"id": "a", "changed": True}]])

    def test_partial_and_all_correct_groups(self):
        responses = [{"correct": True, "completion_tokens": 2},
                     {"correct": True, "completion_tokens": 4},
                     {"correct": True, "completion_tokens": 4}]
        self.assertEqual(strict_pairs(responses), [(0, 1), (0, 2)])
        self.assertEqual(strict_pairs([responses[0], {"correct": False}, {}]), [(0, 1)])
        self.assertEqual(strict_pairs([{"correct": False}, {}]), [])
