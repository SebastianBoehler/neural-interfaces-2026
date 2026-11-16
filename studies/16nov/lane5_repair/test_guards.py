"""Small guard tests; inference and numerical policy parity run separately."""
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
os.environ['NI26_WS'] = str(ROOT)
spec = importlib.util.spec_from_file_location('repair_score_test', ROOT / 'studies/16nov/lane5_repair/score.py')
score = importlib.util.module_from_spec(spec); spec.loader.exec_module(score)


class GuardTests(unittest.TestCase):
    def test_no_confident_rows_are_recorded_as_skips(self):
        status = {'windows': 61, 'attempts': 2, 'updates': 0, 'skips': 2,
                  'selected_total': 0, 'max_grad': 0., 'delta': 0., 'selections': [0, 0]}
        score.new.check_status(status, 61)
        for bad in ({**status, 'updates': 1, 'skips': 1}, {**status, 'delta': .04},
                    {**status, 'selections': [0, 65], 'selected_total': 65}):
            with self.assertRaises(ValueError):
                score.new.check_status(bad, 61)

    def test_legacy_ties_are_counted_and_large_changes_fail(self):
        reference = np.array([[.5, .5, 0.], [.8, .1, .1]])
        self.assertEqual(score.new.verify_legacy(reference, reference, 'test')['near_ties'], 1)
        changed = reference.copy(); changed[1] = [.1, .8, .1]
        with self.assertRaises(ValueError):
            score.new.verify_legacy(changed, reference, 'test')

    def test_missing_matrix_fails_before_inputs_or_labels_are_read(self):
        previous = score.ROOT
        try:
            with tempfile.TemporaryDirectory() as tmp:
                score.ROOT = Path(tmp)
                with self.assertRaisesRegex(ValueError, 'incomplete matrix'):
                    score.load_complete({'subject': []}, None, None, None, {})
        finally:
            score.ROOT = previous


if __name__ == '__main__':
    unittest.main()
