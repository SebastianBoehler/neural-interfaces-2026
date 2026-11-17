"""Pinned native reference/source6 instrument shared with 337; no held labels."""
import sys
from pathlib import Path
PARENT=Path('/mnt/lustre/work/gehler/gfh527/ni26_pretrain337_pilot_v3_20261117')
sys.path.insert(0,str(PARENT))
from common import *
BASE=Path('/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_20261117_v2')
sys.path.insert(0,str(BASE/'experiments'))
import ts_core2 as T
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')
CONTROL=Path('/mnt/lustre/work/gehler/gfh527/ni26_combinations338_v2_20261117/output/source6/A')


def permutations(n):
    with np.load(BASE/'runs/shuffle_perms.npz') as z:
        out={o:z[f'{o[:-1]}|{o[-1]}|all_loso'] for o in ORDERS}
    if any(not np.array_equal(np.sort(p),np.arange(n)) for p in out.values()):raise ValueError('pinned permutation')
    return out
