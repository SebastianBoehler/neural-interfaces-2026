"""Exercise the returning-stream array wrapper without starting a training job."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CELL = ROOT / "studies/15nov/s281/cell_seen.sh"


class SeenCellTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(CELL.is_file(), "returning-stream wrapper must exist")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        stub = Path(self.tmp.name) / "bash"
        stub.write_text('#!/bin/sh\nprintf "EXEC=%s,%s SCHEME=%s PSEED=%s FAMS=%s BASE_ONLY=%s PROTOS=%s DRAW=%s\\n" "$1" "$2" "$SCHEME" "$PSEED" "$FAMS" "$BASE_ONLY" "$PROTOS" "${DRAW-unset}"\n')
        stub.chmod(0o755)
        self.env = dict(os.environ, PATH=self.tmp.name + os.pathsep + os.environ["PATH"])
        for name in ("DRAW", "FAMS", "SCHEME", "PSEED", "PROTOS", "BASE_ONLY"):
            self.env.pop(name, None)

    def run_cell(self, task, **env):
        return subprocess.run(["/bin/bash", str(CELL), str(task)], cwd=ROOT,
                              env=dict(self.env, **env), text=True, capture_output=True)

    def test_order_mapping_and_seen_protocol(self):
        for task, scheme, seed in ((0, "full", 0), (12, "full", 1), (24, "runs", 0), (36, "runs", 1),
                                   (48, "within", 0), (60, "within", 1), (72, "within", 2), (84, "within", 3)):
            with self.subTest(task=task):
                result = self.run_cell(task)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"EXEC=studies/15nov/s281/task.sh,0 SCHEME={scheme} PSEED={seed}", result.stdout)
                self.assertIn("FAMS=fb_coral,fb_am BASE_ONLY=1 PROTOS=seen_session DRAW=unset", result.stdout)

    def test_within_order_task_mapping_preserves_family_and_seed(self):
        for task, inner in ((48, 0), (49, 1), (50, 2), (59, 11), (95, 11)):
            with self.subTest(task=task):
                result = self.run_cell(task)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"EXEC=studies/15nov/s281/task.sh,{inner} ", result.stdout)

    def test_invalid_array_tasks_fail(self):
        for task in (-1, 96, "bogus"):
            with self.subTest(task=task):
                self.assertNotEqual(self.run_cell(task).returncode, 0)

    def test_nonempty_draw_is_rejected(self):
        self.assertNotEqual(self.run_cell(48, DRAW="1").returncode, 0)

    def test_plain_fb_control_has_only_six_seeds(self):
        self.assertEqual(self.run_cell(48, FAMS="fb").returncode, 0)
        self.assertEqual(self.run_cell(53, FAMS="fb").returncode, 0)
        self.assertNotEqual(self.run_cell(54, FAMS="fb").returncode, 0)

    def test_unknown_family_override_is_rejected(self):
        self.assertNotEqual(self.run_cell(48, FAMS="other").returncode, 0)


if __name__ == "__main__":
    unittest.main()
