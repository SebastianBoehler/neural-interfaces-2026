"""Prospective elapsed-cost gate before independent all-CPU representation matrix."""
import importlib.util
import json
from pathlib import Path

ROOT = Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_20261117_v2')
timing = json.loads((ROOT / 'probe_timing.json').read_text())
if timing['elapsed'] <= 0 or timing['elapsed'] > 540:
    raise RuntimeError(f'CPU complete-cell cost exceeds prospective540s release guard: {timing};no wave released')
if not (ROOT / 'output/fb_T_s0_P01.npz').is_file():
    raise ValueError('guarded canonical CPU producer output missing')
spec = importlib.util.spec_from_file_location('frozen_cpu_dispatch', ROOT / 'dispatch_cpu.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module.main(0)
print(f'complete source-locked CPU cell passed;actualtiming={timing}', flush=True)
