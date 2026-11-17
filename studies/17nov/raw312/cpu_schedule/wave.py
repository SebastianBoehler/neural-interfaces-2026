"""Only change future CPU scheduler requests; original scientific source stays pinned."""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_20261117_v2')
WRAPPER = Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_schedule_20261117/wave.sbatch')
spec = importlib.util.spec_from_file_location('frozen312_cpu_waves', ROOT / 'dispatch_cpu.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
original = subprocess.check_output


def scheduling(argv, *args, **kwargs):
    command = list(argv)
    if command[0] == 'sbatch':
        if 'work_cpu.sbatch' in command:
            command.insert(1, '--time=00:10:00')
        if 'dispatch_cpu.sbatch' in command:
            command[command.index('dispatch_cpu.sbatch')] = str(WRAPPER)
    return original(command, *args, **kwargs)


module.subprocess.check_output = scheduling
module.main(int(sys.argv[1]))
