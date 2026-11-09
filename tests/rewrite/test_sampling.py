import unittest

from tc.data import strict_pairs
from tc.sampling import views


class SamplingTests(unittest.TestCase):
    def test_extended_balance_rotation_and_partial_groups(self) -> None:
        for count in range(1, 17):
            for correct in range(count + 1):
                group = {'id': 'g', 'responses': [dict(sample_index=i, correct=i < correct,
                         completion_tokens=[1] * (i + 1)) for i in range(count)]}
                for epoch in (0, 1, 3):
                    selected = views(group, epoch)
                    self.assertEqual(selected, views(group, epoch))
                    expected = 0 if not strict_pairs(group['responses']) else (2 if count <= 8 else (count + 3) // 4)
                    self.assertEqual(len(selected), expected)
                    for view in selected:
                        self.assertEqual(len(view['responses']), min(4, count))
                        self.assertEqual(len({r['sample_index'] for r in view['responses']}), min(4, count))
                        self.assertTrue(strict_pairs(view['responses']))
        group = {'id': 'g', 'responses': [dict(sample_index=i, correct=i < 8,
                 completion_tokens=[1] * (i + 1)) for i in range(16)]}
        self.assertEqual([sum(r['correct'] for r in v['responses']) for v in views(group, 0)], [3, 2, 2, 2])
        self.assertEqual([sum(r['correct'] for r in v['responses']) for v in views(group, 1)], [2, 2, 2, 3])
