"""Guard checks for the isolated §281 runner."""
import importlib.util
import unittest
import subprocess
import sys
from pathlib import Path

import numpy as np

PATH = Path(__file__).with_name("s281_guard.py")


class GuardTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.is_file(), "§281 guard module must exist")
        spec = importlib.util.spec_from_file_location("s281_guard", PATH)
        self.guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.guard)
        self.control = {"B|input": np.array([[0.2, 0.8]]), "B|gate": np.array([1]), "B|V6Fr": np.array([[0.1, 0.9]])}

    def test_equal_zero_arm_and_gate_pass(self):
        self.guard.verify(self.control, self.control, zero=True)

    def test_changed_gate_input_rejected(self):
        arm = dict(self.control, **{"B|input": np.array([[0.3, 0.7]]), "B|V6Fr": np.array([[0.2, 0.8]])})
        with self.assertRaisesRegex(AssertionError, "gate input"):
            self.guard.verify(self.control, arm, zero=False)

    def test_archived_control_accepts_equal_outputs(self):
        bases = ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2")
        outputs = {f"{b}|{arm}": np.array([[0.25, 0.75]], dtype=np.float32) for b in bases for arm in ("P0r", "V6Fr")}
        self.assertTrue(hasattr(self.guard, "verify_archive"), "archive verifier missing")
        self.guard.verify_archive(outputs, outputs)

    def test_archived_control_rejects_changed_value(self):
        bases = ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2")
        outputs = {f"{b}|{arm}": np.array([[0.25, 0.75]], dtype=np.float32) for b in bases for arm in ("P0r", "V6Fr")}
        reference = dict(outputs, **{"B2_s0|V6Fr": np.array([[0.26, 0.74]], dtype=np.float32)})
        self.assertTrue(hasattr(self.guard, "verify_archive"), "archive verifier missing")
        with self.assertRaisesRegex(AssertionError, "B2_s0.*V6Fr"):
            self.guard.verify_archive(outputs, reference)

    def test_archived_control_rejects_missing_key(self):
        bases = ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2")
        outputs = {f"{b}|{arm}": np.array([[0.25, 0.75]], dtype=np.float32) for b in bases for arm in ("P0r", "V6Fr")}
        reference = dict(outputs); del reference["B2_s2|P0r"]
        self.assertTrue(hasattr(self.guard, "verify_archive"), "archive verifier missing")
        with self.assertRaisesRegex(AssertionError, "B2_s2.*P0r"):
            self.guard.verify_archive(outputs, reference)

    def test_unchanged_nonzero_arm_rejected(self):
        with self.assertRaisesRegex(AssertionError, "did not change"):
            self.guard.verify(self.control, self.control, zero=False)

    def test_changed_zero_arm_rejected(self):
        arm = dict(self.control, **{"B|V6Fr": np.array([[0.2, 0.8]])})
        with self.assertRaisesRegex(AssertionError, "zero"):
            self.guard.verify(self.control, arm, zero=True)

class RunnerSourceTest(unittest.TestCase):
    def test_isolated_python_can_load_guard(self):
        path = PATH.with_name("run_s_v66.py")
        p = subprocess.run([sys.executable, "-I", str(path)], capture_output=True, text=True)
        self.assertIn("supply one delivery order", p.stderr)
        self.assertNotIn("ModuleNotFoundError", p.stderr)

    def test_runner_restricts_member_changes_to_v6fr_and_uses_capped_r15(self):
        path = PATH.with_name("run_s_v66.py")
        self.assertTrue(path.is_file(), "§281 corrected runner must exist")
        spec = importlib.util.spec_from_file_location("run_s_v66", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        source = path.with_name("run_s.py").read_text()
        fixed = runner.patch_runner(source)
        self.assertIn('FT_V="r15c250"', fixed)
        self.assertIn('f"ttt18{O}LPC"', fixed)
        self.assertIn('CURRENT["arm"] != "V6Fr"', fixed)
        self.assertIn('execute_chain()', fixed)
        compile(fixed, str(path), 'exec')

    def test_control_requires_archive_before_saving(self):
        source = PATH.with_name("run_s_v66.py").read_text()
        self.assertIn("s281_v66_reference/s242c250c_", source)
        self.assertLess(source.index("verify_archive(outputs, z)"), source.index("np.savez_compressed(audit"))

    def test_chain_gate_and_save_are_instrumented(self):
        path = PATH.with_name("run_s_v66.py")
        spec = importlib.util.spec_from_file_location("run_s_v66", path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        source = (path.parents[2] / "13nov/shufcombo.py").read_text()
        fixed = runner.patch_chain(source)
        self.assertIn('CURRENT["arm"] = arm', fixed)
        self.assertIn('g = capture_gate(bn, arm, gm.gate,', fixed)
        self.assertIn('checked_save(OUTS)', fixed)
        compile(fixed, str(path), "exec")


if __name__ == "__main__":
    unittest.main()
