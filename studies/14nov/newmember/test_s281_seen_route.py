"""Checks for returning-session recognition traces."""
import importlib.util
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).with_name("s281_seen_route.py")


class SeenRouteTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.is_file(), "seen-session routing diagnostic must exist")
        spec = importlib.util.spec_from_file_location("s281_seen_route", PATH)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_recognition_latches_and_resets_for_each_session(self):
        traces = [dict(rows=np.array([12, 11, 10]), contrast=np.array([0., .3, 0.]), bio=np.zeros(3)),
                  dict(rows=np.array([20, 22, 21]), contrast=np.zeros(3), bio=np.array([0., 0., .2]))]
        rows = np.array([10, 11, 12, 20, 21, 22])
        result = self.mod.latch_traces(traces, rows)
        np.testing.assert_array_equal(result, [True, True, False, False, True, False])

    def test_duplicate_or_missing_row_fails(self):
        traces = [dict(rows=np.array([10, 11]), contrast=np.zeros(2), bio=np.zeros(2)),
                  dict(rows=np.array([11, 13]), contrast=np.zeros(2), bio=np.zeros(2))]
        with self.assertRaisesRegex(AssertionError, "partition"):
            self.mod.latch_traces(traces, np.arange(10, 14))

    def test_nonfinite_routing_ramp_fails(self):
        traces = [dict(rows=np.array([0]), contrast=np.array([np.nan]), bio=np.zeros(1))]
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            self.mod.latch_traces(traces, np.arange(1))


if __name__ == "__main__":
    unittest.main()
