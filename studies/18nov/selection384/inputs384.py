"""Pin actual §382 actor outputs, complete cap250 CP inputs, and source labels."""
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path('/mnt/lustre/work/gehler/gfh527')
FUSION = BASE/'ni26_fusion383_v1_20261118'
SCHEDULE = BASE/'ni26_refit375_v1_20261117'
sys.path[:0] = [str(FUSION),str(SCHEDULE)]
from fusion_inputs import CHAIN, COHORT, member_bank, pin, sha
from trajectory import INPUT, check, digest, prepare

ORDERS = ('within0','within1','within2','within3','full0','full1','runs0','runs1')
ARMS = ('Z','R','T','S','D','K')
SEEDS = (382,1382)
SOURCE = BASE/'ni26_s306_expert_cpu_v2_20261116'


def setup():
    for parent, h in ((FUSION,'9da18921a7e9d27807c21a511a7c27798c0f7f04f433833ce17419ac462c3522'),
                      (SCHEDULE,'289a7f6d7540ddaa4db1f42db5a1f67a323fcc2626057e94a5bdd76694404b1c')):
        pin(parent/'source.sha256',h)
        for line in (parent/'source.sha256').read_text().splitlines():
            h,n = line.split(maxsplit=1)
            pin(parent/n,h)
    pin(SCHEDULE/'features.npy','fcbd25192a1e1a030c0905f2cc795c21e1984cbb7ff9c1fbd4b5653e21ec9103')
    pin(SOURCE/'source_labels.py','35edab56fe754e6c77a81f54f2010baca7e0912af55af9564043b2a8bde8c4dc')
    sys.path.insert(0,str(SOURCE))
    from source_labels import read_training_labels
    assets, _, cfg = prepare()
    cp, cfg2 = assets.get_stage()
    check(all(cfg[k]==cfg2[k] for k in ('v6_cp_t0','v6_cp_k','v6_cp_w0')), 'stage mismatch')
    with np.load(CHAIN/'data/proteus_windows.npz') as z:
        meta = {k:z[k] for k in ('subject','session','interface')}
    meta['person'] = np.array([s.split('/')[-1] for s in meta['subject']])
    meta['stream'] = np.char.add(np.char.add(meta['subject'],'|'),meta['session'])
    rows = {p:np.flatnonzero(meta['person']==p) for p in COHORT}
    check(all(len(set(meta['stream'][r]))==1 for r in rows.values()), 'single-session cohort changed')
    return assets,cp,cfg,meta,rows,read_training_labels


def banks(seed,rows):
    return {p:member_bank(seed,p,r,ORDERS) for p,r in rows.items()}
