"""Existing-guard smoke verification, then capacity-bounded treatment release."""
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from config import BASES, ORDERS

ROOT = Path(os.environ['NI26_WS']).resolve(strict=True)
sys.path.insert(0, str(ROOT / 'studies/14nov/newmember'))


def verify_smoke():
    scorer = importlib.import_module('score_s_v66p')
    guard = importlib.import_module('s281_guard')
    from s281_routing import verify_archive_file
    config = json.loads((ROOT / 'representation_config.json').read_text())
    cache = {}
    with np.load(ROOT / 'runs/replay_data/bases.npz', allow_pickle=True) as z:
        n = len(z['y|all_loso']); count = len(np.unique(z['sess|all_loso']))
    for order in ORDERS:
        archive = ROOT / f'runs/s281_v66_reference/s242c250c_{order}.npz'
        ah = verify_archive_file(archive, archive.parent / 'sha256.remote.txt')
        route = json.loads((ROOT / f'runs/s281_v66_route/{order}.json').read_text())
        scorer.verify_route(route, order, ah, BASES, count)
        with np.load(ROOT / f'runs/chainout/s281_v66p_control_{order}.npz') as c, np.load(archive) as a:
            guard.verify_archive(c, a)
        for base in BASES:
            scorer.load_output(ROOT, 'control', order, base, n, cache)
    remote = scorer.remote_manifest(ROOT / 'runs/representation_inputs/streams.sha256')
    for arm in ('zero_repr', config['arms'][0]):
        prov = json.loads((ROOT / f'runs/newmember/s281_v66p_{arm}_within0_provenance.json').read_text())
        scorer.verify_streams(prov, arm, 'within0', remote)
        with np.load(ROOT / 'runs/newmember/s281_v66p_control_within0_gate.npz') as c, np.load(ROOT / f'runs/newmember/s281_v66p_{arm}_within0_gate.npz') as a:
            guard.verify(dict(c), dict(a), zero=arm == 'zero_repr')
        for base in BASES:
            actual = scorer.load_output(ROOT, arm, 'within0', base, n, cache)
            if arm == 'zero_repr':
                scorer.verify_zero(scorer.load_output(ROOT, 'control', 'within0', base, n, cache), actual)
    return config


def main():
    config = verify_smoke()
    tasks = [(a, o) for a in ('zero_repr', *config['arms']) for o in ORDERS
             if not (o == 'within0' and a in ('zero_repr', config['arms'][0]))]
    queued = subprocess.check_output(['squeue', '-u', 'gfh527', '-h', '-r', '-o', '%i'], text=True).splitlines()
    running = subprocess.check_output(['squeue', '-u', 'gfh527', '-h', '-t', 'R', '-o', '%i'], text=True).splitlines()
    if len(queued) + len(tasks) + 1 > 272:
        raise RuntimeError('insufficient submitted capacity for complete treatment group; nothing released')
    slots = min(4, 32 - len(running))
    if slots < 1:
        raise RuntimeError('no safe running capacity for treatment group; nothing released')
    (ROOT / 'remaining_cells.json').write_text(json.dumps(tasks))
    def submit(*args):
        return subprocess.check_output(['sbatch', '--parsable', *args], text=True).strip().split(';')[0]
    array = submit(f'--array=0-{len(tasks)-1}%{slots}', 'studies/17nov/representation_chain/job.sbatch', 'remaining')
    score = submit(f'--dependency=afterok:{array}', 'studies/17nov/representation_chain/score.sbatch')
    with (ROOT / 'released.json').open('x') as f:
        json.dump(dict(treatments=array, scorer=score, slots=slots, task_count=len(tasks)), f)
    print(f'full-system smoke passed;released{array},scorer{score},concurrency{slots}', flush=True)


if __name__ == '__main__':
    main()
