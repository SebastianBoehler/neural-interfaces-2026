"""Real-window checks of the §281 package-stage adapter."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
MODULE = Path(__file__).with_name("s281_cp.py")


class PersonalisationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(ROOT / "data/proteus_windows.npz") as data:
            cls.windows = data["X"][:24].astype(np.float64)
        cls.raw = np.full((24, 3), 0.1)
        cls.raw[np.arange(24), np.arange(24) % 3] = 0.8
        cls.filtered = np.roll(cls.raw, 1, axis=1)

    def setUp(self):
        self.assertTrue(MODULE.is_file(), "§281 personalisation adapter must exist")
        spec = importlib.util.spec_from_file_location("s281_cp", MODULE)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.config = dict(self.mod.CONFIG, v6_cp_t0=6, v6_cp_k=3, v6_cp_w0=6, v6_cp_R=5)

    def test_matches_direct_package_and_resets_between_sessions(self):
        actual = self.mod.apply_sessions(self.windows, self.raw, self.filtered,
                                         [np.arange(12), np.arange(12, 24)], self.config)
        expected = np.empty_like(actual)
        for rows in (np.arange(12), np.arange(12, 24)):
            cp = self.mod.CP.CausalPers(self.config)
            for row in rows:
                expected[row] = cp.step(self.windows[row], self.raw[row], self.filtered[row], False)
            self.assertEqual(cp.log["err"], 0)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
        self.assertFalse(np.allclose(actual[6:12], self.filtered[6:12]))

    def test_zero_cap_preserves_filtered_labels(self):
        config = dict(self.config, v6_cp_cap=0.0)
        out = self.mod.apply_sessions(self.windows, self.raw, self.filtered,
                                      [np.arange(12), np.arange(12, 24)], config)
        np.testing.assert_array_equal(out.argmax(1), self.filtered.argmax(1))

    def test_future_posteriors_do_not_affect_earlier_outputs(self):
        rows = [np.arange(12), np.arange(12, 24)]
        original = self.mod.apply_sessions(self.windows, self.raw, self.filtered, rows, self.config)
        raw = self.raw.copy()
        raw[9:12] = raw[9:12, ::-1]
        altered = self.mod.apply_sessions(self.windows, raw, self.filtered, rows, self.config)
        np.testing.assert_allclose(original[:9], altered[:9], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
