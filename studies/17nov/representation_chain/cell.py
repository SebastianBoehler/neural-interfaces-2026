"""One existing chain/order cell plus its six proven parallel CP children."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

from config import BASES, ORDERS

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)


def main(arm, order):
    config = json.loads((ROOT / 'representation_config.json').read_text())
    add, swap, zero = config['recipes'][arm]
    if order not in ORDERS:
        raise ValueError('invalid pinned order')
    env = dict(os.environ, ADD=add, SWAP=swap, ZERO=str(int(zero)), NM_NAME=arm, NM_SEEDS='0,1')
    runner = ROOT / 'studies/14nov/newmember/run_s_v66.py'
    subprocess.run([sys.executable, '-B', '-I', str(runner), order], env=env, check=True)
    path = ROOT / 'studies/15nov/s281/galv_v66_job.py'
    spec = importlib.util.spec_from_file_location('proven_cp_children', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cp = ROOT / 'studies/14nov/newmember/s281_cp.py'
    commands = [(f'{arm}_{order}_{b}', [sys.executable, '-B', '-I', str(cp), arm, order, b]) for b in BASES]
    log = ROOT / f'logs/cp_{arm}_{order}'; log.mkdir(exist_ok=False)
    module.run_children(commands, log, env)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
