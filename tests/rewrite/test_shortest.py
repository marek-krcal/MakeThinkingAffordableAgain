import json
from pathlib import Path
import tempfile
import unittest

from tc.data import shortest_pairs
from tc.preferences import PreferenceGroups
from tc.sampling import views


class ShortestTests(unittest.TestCase):
    def test_boundaries_and_pair_exposure(self) -> None:
        responses = [dict(sample_index=i, correct=correct, completion_tokens=[1] * length)
                     for i, (correct, length) in enumerate([
                         (True, 149), (False, 99), (True, 100), (False, 100),
                         (False, 101), (True, 150), (True, 100), (True, 200)])]
        self.assertEqual(shortest_pairs(responses), [(2, 4), (2, 5), (2, 7)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'groups.jsonl'
            path.write_text(json.dumps(dict(id='g', split='train', prompt_tokens=[2], responses=responses)) + '\n')
            groups = PreferenceGroups(input=str(path), source='groups', pairing='shortest').form_groups()
            for epoch in (0, 1, 3):
                selected = views(groups[0], epoch)
                self.assertEqual([[r['sample_index'] for r in v['responses']] for v in selected],
                                 [[2, 4], [2, 5], [2, 7]])
                self.assertTrue(all(v['pairs'] == [(0, 1)] for v in selected))
            # Saved pair metadata must never alter a later all-preferences run.
            path.write_text(json.dumps(groups[0]) + '\n')
            self.assertNotIn('pairs', PreferenceGroups(input=str(path), source='groups').form_groups()[0])
        self.assertEqual(shortest_pairs([r for r in responses if not r['correct']]), [])
        self.assertEqual(shortest_pairs([responses[2], responses[6]]), [])

    def test_wrong_length_threshold_survives_group_formation(self) -> None:
        responses = [dict(sample_index=i, correct=correct, completion_tokens=[1] * length)
                     for i, (correct, length) in enumerate([
                         (True, 10), (False, 14), (False, 15), (False, 16), (True, 14), (True, 15)])]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'groups.jsonl'
            path.write_text(json.dumps(dict(id='g', split='train', prompt_tokens=[2], responses=responses)) + '\n')
            groups = PreferenceGroups(input=str(path), source='groups', pairing='shortest', wrong_ratio=1.5).form_groups()
            self.assertEqual(groups[0]['pairs'], [(0, 2), (0, 3), (0, 5)])
            self.assertEqual([[r['sample_index'] for r in view['responses']] for view in views(groups[0], 0)],
                             [[0, 2], [0, 3], [0, 5]])
        self.assertEqual(shortest_pairs(responses), [(0, 1), (0, 2), (0, 3), (0, 5)])
        for ratio in (0.9, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                PreferenceGroups(wrong_ratio=ratio)

    def test_reject_invalid_pairs_and_ratio(self) -> None:
        for ratio in (1, float('nan'), float('inf'), '1.5'):
            with self.assertRaises(ValueError):
                PreferenceGroups(correct_ratio=ratio)
        group = dict(id='g', responses=[dict(sample_index=0, correct=True, completion_tokens=[1]),
                                       dict(sample_index=1, correct=False, completion_tokens=[2, 3])])
        for pairs in ([], [(1, 0)], [(0, 1), (0, 1)], [(0, 2)]):
            with self.assertRaises(ValueError):
                views(dict(group, pairs=pairs), 0)
