"""Contract checks for seen-session prefix labels."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).with_name("s281_seen_relabel.py")


class SeenRelabelTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.is_file(), "seen-session prefix producer must exist")
        spec = importlib.util.spec_from_file_location("s281_seen_relabel", PATH)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.positions = np.r_[np.arange(25), np.arange(25)]
        self.sessions = np.r_[np.zeros(25, int), np.ones(25, int)]
        self.grid = np.array([20])
        self.recognised = np.zeros(50, bool)
        self.recognised[23:25] = True
        self.recognised[25 + 22:] = True
        self.prefix = np.full((50, 1, 3), np.nan)
        self.prefix[self.positions < 20, 0] = [0.2, 0.3, 0.5]

    def test_valid_session_prefixes_pass(self):
        self.mod.validate_prefix(self.prefix, self.positions, self.sessions, self.grid, self.recognised)

    def test_future_window_in_prefix_fails(self):
        self.prefix[22, 0] = [0.2, 0.3, 0.5]
        with self.assertRaisesRegex(AssertionError, "future"):
            self.mod.validate_prefix(self.prefix, self.positions, self.sessions, self.grid, self.recognised)

    def test_missing_pre_recognition_label_fails(self):
        self.prefix[3, 0] = np.nan
        with self.assertRaisesRegex(AssertionError, "missing"):
            self.mod.validate_prefix(self.prefix, self.positions, self.sessions, self.grid, self.recognised)

    def test_recognition_cannot_unlatch(self):
        self.recognised[24] = False
        with self.assertRaisesRegex(AssertionError, "unlatched"):
            self.mod.validate_prefix(self.prefix, self.positions, self.sessions, self.grid, self.recognised)

    def test_cap_250_includes_refit_at_250_but_not_260(self):
        n = 261; pos = np.arange(n); sessions = np.zeros(n, int); grid = np.arange(20, 261, 10)
        prefix = np.full((n, len(grid), 3), np.nan); recognised = np.zeros(n, bool)
        for k, t in enumerate(grid):
            if t <= 250:
                prefix[:t, k] = [0.2, 0.3, 0.5]
        self.mod.validate_prefix(prefix, pos, sessions, grid, recognised)
        missing = prefix.copy(); missing[:250, -2] = np.nan
        with self.assertRaisesRegex(AssertionError, "missing"):
            self.mod.validate_prefix(missing, pos, sessions, grid, recognised)
        prefix[:260, -1] = [0.2, 0.3, 0.5]
        with self.assertRaisesRegex(AssertionError, "post-cap"):
            self.mod.validate_prefix(prefix, pos, sessions, grid, recognised)

    def test_base_must_match_trace_and_cannot_default_to_pool(self):
        self.assertTrue(hasattr(self.mod, "require_matching_base"), "base provenance check must exist")
        trace = {"B2_s0|shadow": np.ones((2, 3)), "B2_s0|recognised": np.zeros(2, bool)}
        self.mod.require_matching_base(trace, "B2_s0")
        with self.assertRaisesRegex(AssertionError, "matching base"):
            self.mod.require_matching_base(trace, "B2_pool")

    def test_artifact_rejects_wrong_or_missing_base_and_order(self):
        artifact = {"base": np.array("B2_s0"), "order": np.array("within0")}
        self.mod.require_matching_artifact(artifact, "within0", "B2_s0")
        with self.assertRaisesRegex(AssertionError, "base mismatch"):
            self.mod.require_matching_artifact(artifact, "within0", "B2_pool")
        with self.assertRaisesRegex(AssertionError, "base mismatch"):
            self.mod.require_matching_artifact({"order": np.array("within0")}, "within0", "B2_s0")
        with self.assertRaisesRegex(AssertionError, "order mismatch"):
            self.mod.require_matching_artifact(artifact, "within1", "B2_s0")


if __name__ == "__main__":
    unittest.main()
