"""Real LibTorch C ABI checkpoint collision and new guard tests."""
import concurrent.futures
import ctypes as c
import os
import tempfile
import threading
import unittest
from pathlib import Path


class PublicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = c.CDLL(os.environ['LANE5_LIBRARY'])
        cls.lib.n5_create.argtypes = [c.c_int, c.c_int, c.c_int64]
        cls.lib.n5_create.restype = c.c_void_p
        cls.lib.n5_destroy.argtypes = [c.c_void_p]
        cls.lib.n5_save.argtypes = [c.c_void_p, c.c_char_p]
        cls.lib.n5_save.restype = c.c_int
        cls.lib.n5_load.argtypes = [c.c_void_p, c.c_char_p]
        cls.lib.n5_load.restype = c.c_int

    def test_consistency_gradient_guard_is_exported(self):
        self.assertTrue(hasattr(self.lib, 'n5_consistency_grad'), 'missing native KL gradient guard')

    def test_concurrent_creators_have_one_winner_and_never_replace(self):
        with tempfile.TemporaryDirectory(prefix='lane5-publish-') as tmp:
            path = os.fsencode(Path(tmp) / 'checkpoint.pt')
            barrier = threading.Barrier(8)

            def save(seed):
                handle = self.lib.n5_create(3, 0, seed)
                if not handle:
                    raise RuntimeError('native create failed')
                try:
                    barrier.wait(timeout=30)
                    return self.lib.n5_save(handle, path)
                finally:
                    self.lib.n5_destroy(handle)

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(save, range(8)))
            self.assertEqual(results.count(0), 1, results)
            self.assertEqual(results.count(-1), 7, results)
            saved = Path(os.fsdecode(path)).read_bytes()
            handle = self.lib.n5_create(3, 0, 99)
            try:
                self.assertEqual(self.lib.n5_load(handle, path), 0)
                self.assertEqual(self.lib.n5_save(handle, path), -1)
            finally:
                self.lib.n5_destroy(handle)
            self.assertEqual(Path(os.fsdecode(path)).read_bytes(), saved)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ['checkpoint.pt'])


if __name__ == '__main__':
    unittest.main()
