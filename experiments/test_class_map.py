"""Real-data check for the pooled external-label map (20 Nov audit, bugs_datapath 3): reproduces the old mislabel of
shin2018 (identity copy of y) and passes only with the explicit name map. python experiments/test_class_map.py"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import class_map
import protocol

ROOT = Path(__file__).resolve().parents[1]


class ClassMapTest(unittest.TestCase):
    def test_audit_bug_reproduced_and_fixed(self):
        with np.load(ROOT / "data/shin2018/windows.npz") as z:
            names, y = class_map.class_names("shin2018", z), z["y"]
        self.assertEqual(names, ["baseline", "WG"])
        # legacy path: y copied unchanged -> baseline index 0 is MI, WG index 1 is subtraction, none is word generation
        self.assertEqual(np.bincount(y, minlength=3).tolist(), [1560, 1560, 0])
        fixed = class_map.task_labels("shin2018", names, y)
        self.assertEqual(sorted(set(fixed.tolist())), [-1, 2])  # baseline is no task class, WG is class 2
        self.assertTrue((fixed[y == names.index("WG")] == 2).all() and (fixed[y == names.index("baseline")] == -1).all())
        d = protocol.load(("shin2018",))
        ext = d["y"][d["split"] == "external"]
        self.assertEqual(np.bincount(ext, minlength=3).tolist(), [0, 0, 1560])

    def test_scherer_and_shin2017_chunks(self):
        d = protocol.load(("scherer2015",))
        self.assertEqual(np.bincount(d["y"][d["split"] == "external"], minlength=3).tolist(), [710, 710, 710])
        for name, want in (("shin2017a", {"left_hand": 0}), ("shin2017b", {"subtraction": 1, "rest": -1})):
            with np.load(next((ROOT / "data" / name).glob("windows_*.npz"))) as z:
                names = class_map.class_names(name, z)
                y = z["y"]
            got = class_map.task_labels(name, names, y)
            self.assertEqual({c: int(got[y == names.index(c)][0]) for c in want}, want)

    def test_unmapped_and_classless_fail_loudly(self):
        with self.assertRaises(AssertionError):
            class_map.task_labels("x", ["left_hand", "mystery"], np.array([0, 1]))
        with self.assertRaises(AssertionError):
            class_map.class_names("unregistered", {"y": np.zeros(1)})
        with np.load(ROOT / "data/shin2018/windows.npz") as z:  # names swapped away from the pinned export
            with self.assertRaises(AssertionError):
                class_map.class_names("shin2018", {"classes": np.array(["S16", "S32"])})
            self.assertIn("classes", z.files)

    def test_own_head_map(self):
        self.assertEqual(class_map.own_head_labels("ds007221lr", ["right_hand", "left_hand"]).tolist(), [1, 0])
        y = class_map.own_head_labels("ds007221g", ["rest", "feet", "left_hand", "right_hand"])
        self.assertEqual(y.tolist(), [2, 0, 1, 3])
        with self.assertRaises(AssertionError):
            class_map.own_head_labels("netbci", ["right_hand", "mystery"])


if __name__ == "__main__":
    unittest.main()
