import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).with_name('post_smoke.py')


class FollowupTests(unittest.TestCase):
    def test_remaining_indices_never_repeat_smoke_and_cover_registered_arms(self):
        self.assertTrue(PATH.exists(), 'post-smoke scheduler not implemented')
        spec = importlib.util.spec_from_file_location('post_smoke', PATH)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        self.assertEqual(set(m.remaining('coral')), set(range(1, 32)))
        self.assertEqual(set(m.remaining('plain')), set(range(1, 8)) | set(range(9, 16)))
        self.assertEqual(set(m.remaining('dist')), set(range(17, 24)))
        with self.assertRaises(ValueError):
            m.remaining('other')


if __name__ == '__main__':
    unittest.main()
