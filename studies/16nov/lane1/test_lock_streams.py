import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).with_name('lock_streams.py')


class LockTests(unittest.TestCase):
    def test_manifest_rejects_duplicate_names(self):
        self.assertTrue(PATH.exists(), 'stream lock not implemented')
        spec = importlib.util.spec_from_file_location('stream_lock', PATH)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        with self.assertRaises(ValueError):
            m.merge_entries([{'x.npz': 'a'}, {'x.npz': 'b'}])
        self.assertEqual(m.merge_entries([{'x.npz': 'a'}, {'y.json': 'b'}]), {'x.npz': 'a'})


if __name__ == '__main__':
    unittest.main()
