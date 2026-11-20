"""Dispatch one Slurm array element without running its recipe on the login node."""

import json
import os
from pathlib import Path
import sys

entry = json.loads(Path(sys.argv[1]).read_text())[int(os.environ["SLURM_ARRAY_TASK_ID"])]
os.execve(sys.executable, [sys.executable, "-B", entry["program"], *entry["args"]], {**os.environ, **entry["env"]})
