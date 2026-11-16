"""Fail-closed checks for the held-out FBLight adapter."""
import importlib.util
import unittest
from pathlib import Path
import numpy as np

PATH = Path(__file__).with_name('s281_seen_fbl_dev.py')
spec = importlib.util.spec_from_file_location('s281_seen_fbl_dev', PATH)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


class DevAdapterTest(unittest.TestCase):
    def test_unverified_order_base_and_session_fail_before_data_access(self):
        for order, base, session in [('full0', 'B2_s0', 'x'), ('within0', 'B2_s2', 'x'), ('within0', 'B2_s0', 'x')]:
            with self.assertRaises(AssertionError): mod.prepare(order, base, session, {})

    def test_verified_session_cannot_be_created_directly(self):
        with self.assertRaisesRegex(AssertionError, 'use prepare'):
            mod.VerifiedSession(None, {}, np.empty(0), b'', {})

    def test_changed_inputs_fail_before_finetuning(self):
        r = {k: np.array([1.]) for k in ('index', 'prefix', 'grid', 'emitted', 'recognised')}
        v = mod.VerifiedSession(mod._TOKEN, r, np.array([0.]), b'trace', {})
        v.inputs[0] = 1
        with self.assertRaisesRegex(AssertionError, 'inputs changed'):
            mod.consume(v)


if __name__ == '__main__': unittest.main()
