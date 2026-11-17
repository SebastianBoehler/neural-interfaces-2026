"""Read-only historical control guard when its scheduler ID has expired."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/lustre/work/gehler/gfh527/ni26_s323_chain_20261117')
sys.path.insert(0,str(ROOT/'studies/14nov/newmember'))
import score_s_v66p as s
from s281_guard import verify_archive
from s281_routing import verify_archive_file
with np.load(ROOT/'runs/replay_data/bases.npz',allow_pickle=True) as z:
 sessions=len(np.unique(z['sess|all_loso']))
s.verify_package(ROOT/'runs/sealed/final_zips/v66_cp1_cap250.zip',
                 '9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed')
cache={}
for order in s.ORDERS:
 a=ROOT/f'runs/s281_v66_reference/s242c250c_{order}.npz'
 h=verify_archive_file(a,a.parent/'sha256.remote.txt')
 s.verify_route(json.loads((ROOT/f'runs/s281_v66_route/{order}.json').read_text()),order,h,s.BASES,sessions)
 with np.load(ROOT/f'runs/chainout/s281_v66p_control_{order}.npz') as c,np.load(a) as old:verify_archive(c,old)
 for base in s.BASES:s.load_output(ROOT,'control',order,base,2655,cache)
print('OK: eight archived controls/routes and 48 CP provenance guards',flush=True)
