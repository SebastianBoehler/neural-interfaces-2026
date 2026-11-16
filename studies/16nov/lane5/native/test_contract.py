"""Fast contract checks; numerical parity lives in the lane5 harness."""
import ctypes
import os
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class NativeContractTests(unittest.TestCase):
    def test_shared_library_exports_required_abi(self):
        library = Path(os.environ.get('LANE5_LIBRARY', HERE / 'build/liblane5.so'))
        self.assertTrue(library.is_file(), 'native shared library has not been built')
        native = ctypes.CDLL(str(library))
        for name in ('n5_error', 'n5_create', 'n5_clone', 'n5_destroy', 'n5_features',
                     'n5_set_norm', 'n5_param_count', 'n5_get_params', 'n5_set_params',
                     'n5_head_forward', 'n5_head_grads', 'n5_adam_step', 'n5_predict',
                     'n5_fit', 'n5_reset', 'n5_step', 'n5_status', 'n5_fit_status',
                     'n5_consistency_grad', 'n5_save', 'n5_load'):
            with self.subTest(symbol=name):
                self.assertTrue(hasattr(native, name), name)


if __name__ == '__main__':
    unittest.main()
