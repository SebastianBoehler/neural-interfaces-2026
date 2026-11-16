import importlib.util
import unittest
from pathlib import Path
import numpy as np

PATH = Path(__file__).with_name('multistream.py')


class MultiStreamTests(unittest.TestCase):
    def test_order_preserves_all_rows_and_uses_given_rank(self):
        self.assertTrue(PATH.exists(), 'multi-order producer not implemented')
        spec = importlib.util.spec_from_file_location('multistream', PATH)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        self.assertEqual(m.ordered_rows(np.array([1, 3, 5]), np.array([4, 2, 0])), [5, 3, 1])
        with self.assertRaises(ValueError):
            m.ordered_rows(np.array([1, 3, 5]), np.array([0, 0, 1]))

    def test_families_and_seeds_have_a_single_unambiguous_task(self):
        self.assertTrue(PATH.exists(), 'multi-order producer not implemented')
        spec = importlib.util.spec_from_file_location('multistream', PATH)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        self.assertEqual(len({m.mapping(i) for i in range(24)}), 24)
        self.assertEqual(m.mapping(0), ('eegn', 0))
        self.assertEqual(m.mapping(23), ('fb_coral10', 5))
        with self.assertRaises(ValueError):
            m.mapping(24)


if __name__ == '__main__':
    unittest.main()
