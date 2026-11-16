import importlib.util
import unittest
from pathlib import Path
import numpy as np

PATH = Path(__file__).with_name('stage_candidates.py')


class CandidateStageTests(unittest.TestCase):
    def test_canonical_probabilities_are_required(self):
        self.assertTrue(PATH.exists(), 'candidate staging guard not implemented')
        spec = importlib.util.spec_from_file_location('candidate_stage', PATH)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        p = np.full((3, 3), 1/3)
        m.check_output(np.arange(3), p, 3)
        for idx, value in ((np.array([1, 0, 2]), p), (np.arange(3), p * 2)):
            with self.assertRaises(ValueError):
                m.check_output(idx, value, 3)


if __name__ == '__main__':
    unittest.main()
