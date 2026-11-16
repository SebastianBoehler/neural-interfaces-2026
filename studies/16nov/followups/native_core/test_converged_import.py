"""Isolated Python must resolve the assembler's own sibling module."""

import subprocess
import sys
from pathlib import Path


def test_assembler_loads_in_isolated_python():
    path = Path(__file__).with_name('converged_assemble.py')
    code = f'import runpy; runpy.run_path({str(path)!r}, run_name="isolated_probe")'
    result = subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
