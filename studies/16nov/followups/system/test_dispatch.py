"""Exercise the frozen dispatcher shell with controlled Slurm responses."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


class DispatcherTests(unittest.TestCase):
    def run_dispatch(self, parent='COMPLETED|0:0', smoke=0):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        root = Path(temp.name); binpath = root / 'bin'; binpath.mkdir()
        (root / 'runs/lane1_inputs').mkdir(parents=True)
        calls = root / 'calls'
        scripts = {
            'sha256sum': 'exit 0',
            'sacct': 'echo "$FAKE_PARENT"',
            'grep': 'exit 0',
            'fakepython': 'exit "$FAKE_SMOKE"',
            'squeue': 'exit 0',
            'sbatch': 'echo "$*" >> "$FAKE_CALLS"; n=$(wc -l < "$FAKE_CALLS"); echo "$((500+n))"',
        }
        for name, body in scripts.items():
            path = binpath / name; path.write_text('#!/bin/bash\n' + body + '\n'); path.chmod(0o755)
        source = (HERE / 'dispatch2.sbatch').read_text()
        source = source.replace('cd /mnt/lustre/work/gehler/gfh527/ni26_lanes294_20261116/system', f'cd {root}')
        source = source.replace('/mnt/lustre/work/gehler/gfh527/neural-interfaces-2026/.venv/bin/python', 'fakepython')
        env = dict(os.environ, PATH=f'{binpath}:{os.environ["PATH"]}', FAKE_PARENT=parent,
                   FAKE_SMOKE=str(smoke), FAKE_CALLS=str(calls), SLURM_JOB_ID='123')
        result = subprocess.run(['bash'], input=source, text=True, env=env, capture_output=True)
        return result, calls.read_text().splitlines() if calls.exists() else []

    def test_completed_checked_parents_are_not_slurm_dependencies(self):
        result, calls = self.run_dispatch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 3)
        self.assertIn('--array=1-7,64-71,97-111%4', calls[0])
        self.assertIn('--array=8-63,72-95,112-119,121-127%4', calls[1])
        self.assertIn('--dependency=afterok:501:502', calls[2])
        self.assertNotIn('3113408', calls[2])

    def test_failed_parent_or_artifact_guard_never_submits(self):
        for kwargs in ({'parent': 'FAILED|1:0'}, {'smoke': 1}):
            result, calls = self.run_dispatch(**kwargs)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
