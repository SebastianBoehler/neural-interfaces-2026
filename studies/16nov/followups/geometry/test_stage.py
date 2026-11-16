import unittest

from stage import expected_names


class StageTests(unittest.TestCase):
    def test_matrix_has_four_treatments_and_one_zero_for_each_order(self):
        treatment = expected_names(False)
        complete = expected_names(True)
        self.assertEqual(len(treatment), 32)
        self.assertEqual(len(complete), 40)
        self.assertEqual(len(complete - treatment), 8)
        self.assertIn('geo_d100_runs1.npz', treatment)
        self.assertNotIn('geo_zero_within0.npz', treatment)


if __name__ == '__main__':
    unittest.main()
