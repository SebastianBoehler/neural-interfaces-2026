"""Run the six unchanged §281 CP variants in separate checked children."""
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
ARMS = ('zero304','PLA','PLS','UCA','UCS','CCA10','CCS10','CCA30','CCS30')
ORDERS = ('within0','within1','within2','within3','full0','full1','runs0','runs1')
BASES = ('B2_pool','B2_s345','B2_s012','B2_s0','B2_s1','B2_s2')


def main(arm, order):
    if arm not in ARMS or order not in ORDERS or Path.cwd().resolve() != ROOT:
        raise ValueError('invalid §304 CP cell')
    path = ROOT / 'studies/15nov/s281/galv_v66_job.py'
    spec = importlib.util.spec_from_file_location('children', path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    logdir = ROOT / 'logs/cp304'; logdir.mkdir(parents=True, exist_ok=True)
    cp = ROOT / 'studies/14nov/newmember/s281_cp.py'
    commands = [(f'{arm}_{order}_{b}', [sys.executable, '-B', '-I', str(cp), arm, order, b]) for b in BASES]
    mod.run_children(commands, logdir, dict(os.environ))


if __name__ == '__main__': main(*sys.argv[1:])
