import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).with_name('parallel_cp.py')


class ParallelCPTests(unittest.TestCase):
    def test_six_distinct_base_commands(self):
        self.assertTrue(PATH.exists(), 'parallel CP helper not implemented')
        spec = importlib.util.spec_from_file_location('parallel_cp', PATH)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        cmds = module.commands(Path('/tmp/example'), '/python', 'E', 'within0')
        self.assertEqual(len(cmds), 6)
        self.assertEqual(len({name for name, _ in cmds}), 6)
        self.assertTrue(all(cmd[-3:-1] == ['E', 'within0'] for _, cmd in cmds))


if __name__ == '__main__':
    unittest.main()
