"""Schedule and isolation checks for the seen-session FBLight consumer."""
import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('s281_seen_fbl', HERE / 's281_seen_fbl.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)


class FakeFT:
    def __init__(self): self.lab, self.nets, self.refits = [], None, []
    def step(self, members, inputs, out, relabel):
        t = len(self.lab); self.lab.append(out)
        if t >= 20 and (t - 20) % 10 == 0 and t <= 250:
            p = relabel(t); self.refits.append(t); assert p.shape == (t, 3)
        if t >= 20: self.nets = True; return {'fbl': np.array([.2, .3, .5])}
        return {}


def record(n, stop):
    grid = np.arange(20, 261, 10); prefix = np.full((n, len(grid), 3), np.nan)
    for k, t in enumerate(grid):
        if t <= 250 and t < min(stop, n): prefix[:t, k] = [.2, .3, .5]
    return dict(index=np.arange(n), prefix=prefix, grid=grid,
                recognised=np.arange(n) >= stop, emitted=np.tile([.2, .3, .5], (n, 1)))


class ConsumerTest(unittest.TestCase):
    def test_invalid_provenance_rejected_before_file_access(self):
        with self.assertRaisesRegex(AssertionError, 'invalid order or base'):
            mod.session_inputs('../bad', 'B2_s0', 'irrelevant', {})
        with self.assertRaisesRegex(AssertionError, 'invalid order or base'):
            mod.session_inputs('within0', '../bad', 'irrelevant', {})

    def test_recognition_stops_and_new_session_requires_fresh_state(self):
        r = record(65, 60); X = np.zeros((65, 41, 480), dtype=np.float32); ft = FakeFT()
        out, times = mod.run_session(ft, X, r)
        self.assertEqual(times, [20, 30, 40, 50]); self.assertEqual(len(ft.lab), 60)
        self.assertTrue(np.isnan(out[:20]).all() and np.isfinite(out[20:60]).all() and np.isnan(out[60:]).all())
        with self.assertRaisesRegex(AssertionError, 'fresh'):
            mod.run_session(ft, X, r)
        self.assertEqual(mod.run_session(FakeFT(), X, r)[1], times)

    def test_cap_includes_250_and_skips_260(self):
        r = record(270, 265); X = np.zeros((270, 41, 480), dtype=np.float32); ft = FakeFT()
        out, times = mod.run_session(ft, X, r)
        self.assertEqual(times[-1], 250); self.assertNotIn(260, times)
        self.assertTrue(np.isfinite(out[260:265]).all() and np.isnan(out[265:]).all())

    def test_missing_prefix_and_unlatched_recognition_fail(self):
        r = record(31, 31); X = np.zeros((31, 41, 480), dtype=np.float32)
        r['prefix'][0, 0] = np.nan
        with self.assertRaisesRegex(AssertionError, 'invalid refit prefix'):
            mod.run_session(FakeFT(), X, r)
        r = record(31, 25); r['recognised'][28] = False
        with self.assertRaisesRegex(AssertionError, 'unlatched'):
            mod.run_session(FakeFT(), X, r)


if __name__ == '__main__': unittest.main()
