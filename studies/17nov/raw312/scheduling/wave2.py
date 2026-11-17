"""Scheduling-only ten-minute request override; frozen scientific producer unchanged."""
import importlib.util
import os
from pathlib import Path

ROOT = Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_20261117_matrix2')
os.environ['NI26_WS'] = str(ROOT)
spec = importlib.util.spec_from_file_location('frozen312_dispatch', ROOT / 'dispatch.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
submit = module.submit


def shorter(*args):
    if '--array=80-159%2' in args:
        args = ('--time=00:10:00', *args)
    return submit(*args)


module.submit = shorter
module.main()
