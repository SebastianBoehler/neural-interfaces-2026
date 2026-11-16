import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('joint_run', HERE / 'run.py')
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)


class ContractTests(unittest.TestCase):
    def test_combined_recipes_bind_exact_knobs(self):
        for arm, add, swap in [('E05CS', 'eegn', 'fb_coral10'), ('D1CA', 'eegn_dist2,fb_coral10', 'none'),
                               ('FBL15', 'none', 'none'), ('zero', 'eegn', 'none')]:
            recipe = run.validate(arm, add, swap, '1' if arm == 'zero' else '0', '0,1,2,3,4,5')
            self.assertEqual(recipe['name'], arm)
            with self.assertRaises(ValueError):
                run.validate(arm, 'none', 'none', '0', '0,1,2,3,4')
        with self.assertRaises(ValueError):
            run.validate('E05CS', 'eegn', 'none', '0', '0,1,2,3,4,5')


if __name__ == '__main__':
    unittest.main()
