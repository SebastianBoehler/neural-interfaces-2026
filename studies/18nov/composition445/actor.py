"""Use the pinned §442 genuine/blank/donor held actor with §445's fit contract."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

WORK = Path('/mnt/lustre/work/gehler/gfh527')
path = WORK/'ni26_gain442_v1_20261118/actor.py'
assert hashlib.sha256(path.read_bytes()).hexdigest() == '124cba53568081232ff716e51aaeb5a37cefdf89750d92274b18efbdb7d4e3b5'
spec = importlib.util.spec_from_file_location('memory_actor442', path)
actor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(actor)
ARCH, ROWS, SEEDS, joint, prior = actor.ARCH, actor.ROWS, actor.SEEDS, actor.joint, actor.prior
FITS = ('skew_aux','skew_ce','random_aux','random_ce')
ARMS = ('A6','J_real','frozen') + tuple(f'{a}_{m}' for a in FITS for m in actor.MODES)
SOURCE = '6c54cee5877e453678c1350d0ed42ecfdf719393a266a5d08b29769e6340313c'


def lock():
    joint.pin(Path('fits/source.sha256'),SOURCE)
    records = {}
    for seed in SEEDS:
        matched = None
        partitions = {}
        for arm in FITS:
            path = Path('fits')/f'{arm}_{seed}.json'
            rec = json.loads(path.read_text())
            assert (rec['arm'],rec['seed'],rec['steps'],rec['rows']) == (arm,seed,120,ROWS)
            assert rec['bundle'] == actor.BUNDLE and rec['source'] == SOURCE
            joint.pin(path.with_suffix('.pt'),rec['checkpoint'])
            shared = [rec[k] for k in ('before','frozen_sha','batches')]
            assert matched is None or matched == shared
            matched = shared
            partition, objective = arm.split('_')
            detail = [rec[k] for k in ('first_ce_aux','partition_tv')]
            assert partition not in partitions or partitions[partition] == detail
            partitions[partition] = detail
            assert abs(rec['applied_aux_ratio']-(.35 if objective=='aux' else 0)) < 1e-6
            assert all(v > 0 for v in rec['first_grad'].values()) and rec['aux_norm']['norm']>0
            records[f'{seed}|{arm}'] = rec
        assert partitions['skew'][1] > .6 and partitions['random'][1] < .3
    return records


if __name__ == '__main__':
    if sys.argv[1]=='lock':
        Path('fits.json').write_text(json.dumps(lock(),indent=2))
    else:
        actor.FITS = FITS
        actor.lock = lock
        actor.run(sys.argv[1])
