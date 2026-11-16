import importlib.util
import unittest
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('repair_transform', Path(__file__).with_name('transform.py'))
m = importlib.util.module_from_spec(spec)


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): spec.loader.exec_module(m)

    def setUp(self):
        rng = np.random.default_rng(12); p = rng.uniform(.1,1.,(90,3)); self.p = p / p.sum(1,keepdims=True)
        self.sessions = np.array(['a']*45 + ['b']*45)

    def test_identity_and_temperature_argmax(self):
        np.testing.assert_array_equal(m.apply(self.p,self.sessions,'temp',1.), self.p)
        np.testing.assert_array_equal(m.apply(self.p,self.sessions,'center',0.), self.p)
        for t in (.5,2.):
            out=m.apply(self.p,self.sessions,'temp',t)
            np.testing.assert_array_equal(out.argmax(1),self.p.argmax(1)); self.assertFalse(np.array_equal(out,self.p))

    def test_center_strict_prefix_and_session_reset(self):
        out=m.apply(self.p,self.sessions,'center',.5)
        np.testing.assert_array_equal(out[:20],self.p[:20]); np.testing.assert_array_equal(out[45:65],self.p[45:65])
        altered=self.p.copy(); altered[35:45]=altered[35:45,::-1]
        np.testing.assert_array_equal(m.apply(altered,self.sessions,'center',.5)[:35],out[:35])
        separate=m.apply(self.p[45:],self.sessions[45:],'center',.5)
        np.testing.assert_array_equal(out[45:],separate)
        self.assertFalse(np.array_equal(out,self.p))

    def test_real_guard_checks_all_sessions_and_rejects_noncontiguous(self):
        guarded = m.check_causal(self.p, self.sessions, .25)
        direct = m.apply(self.p, self.sessions, 'center', .25)
        np.testing.assert_array_equal(guarded, direct)
        bad_sessions = self.sessions.copy(); bad_sessions[30] = 'b'
        with self.assertRaises(ValueError): m.apply(self.p, bad_sessions, 'center', .25)

    def test_invalid_probability_fails(self):
        bad=self.p.copy(); bad[0,0]=0.
        with self.assertRaises(ValueError): m.apply(bad,self.sessions,'temp',2.)

if __name__ == '__main__': unittest.main()
