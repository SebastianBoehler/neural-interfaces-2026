"""Fail-closed validation tests; no model training or treatment score."""
import importlib.util
import unittest
from pathlib import Path
import numpy as np

PATH = Path(__file__).with_name('sinc_qf_rebuild.py')


class RebuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PATH.exists(): raise AssertionError('reconstruction driver missing')
        spec=importlib.util.spec_from_file_location('sinc_qf_rebuild',PATH)
        cls.m=importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.m)

    def test_parity_checks_numeric_bound_and_labels(self):
        ref=np.array([[.9,.05,.05],[.499999,.5,.000001]])
        self.m.check_parity(ref,ref.copy())
        altered=ref.copy(); altered[0,0]-=.001
        with self.assertRaisesRegex(ValueError,'parity'): self.m.check_parity(altered,ref)
        altered=ref.copy(); altered[1,:2]=altered[1,1::-1]
        with self.assertRaisesRegex(ValueError,'parity'): self.m.check_parity(altered,ref)

    def test_parity_rejects_wrong_shape_and_nonfinite(self):
        with self.assertRaisesRegex(ValueError,'shape'): self.m.check_parity(np.ones((1,3)),np.ones((2,3)))
        with self.assertRaisesRegex(ValueError,'finite'): self.m.check_parity(np.array([[np.nan,0.,1.]]),np.array([[0.,0.,1.]]))

    def test_fold_cannot_include_held_session_or_exclude_own_session_one(self):
        d={'subject':np.array(['Dreyer2026Proteus/P10']*3+['Dreyer2026Proteus/P01']),
           'session':np.array(['1','2','3','1'])}
        self.m.check_fold(d,'P10_s23',np.array([True,False,False,True]),np.array([False,True,True,False]))
        with self.assertRaisesRegex(ValueError,'fold'):
            self.m.check_fold(d,'P10_s23',np.array([True,True,False,True]),np.array([False,True,True,False]))
        with self.assertRaisesRegex(ValueError,'fold'):
            self.m.check_fold(d,'P10_s23',np.array([False,False,False,True]),np.array([False,True,True,False]))


if __name__=='__main__': unittest.main()
