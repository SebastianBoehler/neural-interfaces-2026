"""Tests for the routing-trace completeness and no-route checks."""
import importlib.util
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).with_name("s281_routing.py")


class RoutingTraceTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.is_file(), "routing diagnostic must exist")
        spec = importlib.util.spec_from_file_location("s281_routing", PATH)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.sessions = [dict(rows=np.array([0, 1]), contrast=np.zeros(2), bio=np.zeros(2)),
                         dict(rows=np.array([2, 3]), contrast=np.zeros(2), bio=np.zeros(2))]

    def test_complete_zero_routes_pass(self):
        result = self.mod.check_trace(self.sessions, np.arange(4), np.array([0, 0, 1, 1]))
        self.assertEqual(result["sessions"], 2)
        self.assertEqual(result["routed_sessions"], 0)

    def test_positive_gate_is_reported(self):
        self.sessions[1]["bio"][1] = 0.2
        result = self.mod.check_trace(self.sessions, np.arange(4), np.array([0, 0, 1, 1]))
        self.assertEqual(result["routed_sessions"], 1)
        self.assertEqual(result["bio_windows"], 1)

    def test_missing_or_duplicated_windows_fail(self):
        self.sessions[1]["rows"] = np.array([1, 3])
        with self.assertRaisesRegex(AssertionError, "partition"):
            self.mod.check_trace(self.sessions, np.arange(4), np.array([0, 0, 1, 1]))

    def test_mixed_session_is_rejected(self):
        self.sessions[0]["rows"] = np.array([0, 2])
        self.sessions[1]["rows"] = np.array([1, 3])
        with self.assertRaisesRegex(AssertionError, "session"):
            self.mod.check_trace(self.sessions, np.arange(4), np.array([0, 0, 1, 1]))

    def test_archive_must_match_its_remote_manifest(self):
        self.assertTrue(hasattr(self.mod, "verify_archive_file"), "archive hash check must exist")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "reference.npz"
            path.write_bytes(b"original archive")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = Path(d) / "sha256.remote.txt"
            manifest.write_text(f"{digest}  {path.name}\n")
            self.assertEqual(self.mod.verify_archive_file(path, manifest), digest)
            path.write_bytes(b"changed archive")
            with self.assertRaisesRegex(AssertionError, "archive SHA"):
                self.mod.verify_archive_file(path, manifest)


if __name__ == "__main__":
    unittest.main()
