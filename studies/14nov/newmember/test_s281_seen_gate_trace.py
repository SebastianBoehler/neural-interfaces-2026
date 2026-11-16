"""Gate trace row, person and tie checks."""
import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('s281_seen_gate_trace', HERE / 's281_seen_gate_trace.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


class TraceTest(unittest.TestCase):
    def trace(self):
        return dict(rows=np.array([3, 1, 2]), names=['P04', 'P10'],
                    contrast_r=np.array([0., .5, .4]), contrast_j=np.array([-1, 0, 0]),
                    contrast_score=np.array([np.nan, .3, .3]), bio_r=np.array([0., .5, .8]),
                    bio_j=np.array([-1, 1, 1]), bio_score=np.array([np.nan, .3, .4]))

    def test_winner_identity_and_tie_are_delivered_in_output_row_order(self):
        result = mod.pack([self.trace()], np.array([1, 2, 3]))
        self.assertEqual(result['winner_person'].tolist(), ['P04', 'P10', ''])
        np.testing.assert_array_equal(result['winner_r'], [.5, .8, 0.])

    def test_bad_person_and_missing_rows_fail(self):
        t = self.trace(); t['bio_j'][1] = 3
        with self.assertRaises(AssertionError): mod.pack([t], np.array([1, 2, 3]))
        with self.assertRaisesRegex(AssertionError, 'cover'):
            mod.pack([self.trace()], np.array([1, 2, 3, 4]))

    def test_nan_on_positive_gate_fails(self):
        t = self.trace(); t['bio_score'][2] = np.nan
        with self.assertRaises(AssertionError): mod.pack([t], np.array([1, 2, 3]))


if __name__ == '__main__': unittest.main()
