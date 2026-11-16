import importlib.util
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class PostParallelTests(unittest.TestCase):
    def test_only_registered_treatment_batch_changes(self):
        spec = importlib.util.spec_from_file_location('post_parallel', HERE / 'post_parallel.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertTrue(hasattr(module, 'treatment_command'), 'treatment command rewrite not implemented')
        old = ['sbatch', '--array=1-31%3', 'studies/16nov/lane1/chain_treatment.sbatch', 'coral']
        new = module.treatment_command(old)
        self.assertEqual(new[-2], 'studies/16nov/lane1/chain_treatment_parallel.sbatch')
        self.assertEqual(new[:-2], old[:-2])
        score = ['sbatch', 'studies/16nov/lane1/chain_score.sbatch']
        self.assertEqual(module.treatment_command(score), score)


if __name__ == '__main__':
    unittest.main()
