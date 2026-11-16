import importlib.util
import unittest
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('stage_metrics', Path(__file__).with_name('metrics.py'))
m = importlib.util.module_from_spec(spec)


class StageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): spec.loader.exec_module(m)

    def test_weighted_transitions_sum_to_accuracy_delta(self):
        y = np.array([0,0,1,0,1,1]); cell = np.array([1,1,1,2,2,2]); person = np.array(['P01']*3+['P02']*3)
        before = np.array([0,1,1,0,0,1]); after = np.array([0,0,0,1,1,1])
        up, down, confusion = m.transitions(before, after, y, cell, person, ('P01','P02'))
        self.assertTrue(np.allclose(up.sum(1) - down.sum(1), [-25., -25.]))
        self.assertAlmostEqual(confusion[0].sum(), 100.)
    def test_probability_validation_never_accepts_signed_scores(self):
        valid = np.array([[.5,.3,.2],[.1,.2,.7]])
        m.probability(valid)
        with self.assertRaises(ValueError): m.probability(np.array([[1.1,0.,-.1]]))
        with self.assertRaises(ValueError): m.probability(np.array([[.5,.5,.5]]))

if __name__ == '__main__': unittest.main()
