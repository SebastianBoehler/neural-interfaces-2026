"""Compare a no-score §307 target anchor with its independent §305 reference."""
import json
import os
import sys
from pathlib import Path
import numpy as np

ROOT=Path(os.environ['NI26_WS']).resolve(strict=True)
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'studies/16nov/followups/geometry')]
from config import ARCHIVES
from crossfit import sha
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')


def main(held):
    source=Path(os.environ['S307_S305_ROOT']).resolve(strict=True)
    target=ROOT/f'runs/s307/target/{held}.npz'; ref=source/f'runs/s305_converged_shards/{held}.npz'
    lock=json.loads((source/'runs/s305_converged/manifest.json').read_text())
    if lock['shards'].get(f'{held}.npz') != sha(ref): raise ValueError('§305 verified shard checksum changed')
    with np.load(target,allow_pickle=False) as z, np.load(ref,allow_pickle=False) as q:
        m=json.loads(str(z['metadata'].item())); r=json.loads(str(q['metadata'].item())); rows=z['rows']
        if m['held'] != held or m['held_accuracy_computed'] is not False or r['held'] != held or r['held_accuracy_computed'] is not False or not np.array_equal(rows,q['rows']):
            raise ValueError('target/reference fold mismatch')
        for order in ORDERS:
            path=ROOT/'runs'/ARCHIVES[order]
            if sha(path) != r['archives_sha256'][ARCHIVES[order]]: raise ValueError('original core archive differs')
            with np.load(path,allow_pickle=False) as archive: reference=q[f'Nconverged|{order}']+archive[f'{order}|all_loso'][rows]
            actual=z[f'anchor|{order}']; delta=float(np.max(np.abs(reference-actual)))
            sorted_p=np.sort(reference,axis=1); non_tied=sorted_p[:,-1] > sorted_p[:,-2]
            changed=int(np.count_nonzero((reference.argmax(1)!=actual.argmax(1))&non_tied))
            if delta > 1e-5 or changed: raise ValueError(f'{held}/{order}: target anchor parity {delta=} {changed=}')
            print(held,order,'anchor max difference',delta,'non-tied labels changed',changed,flush=True)
    print(f'{held}: 8-order anchor/zero/reset target guard passed; no score read',flush=True)


if __name__=='__main__': main(sys.argv[1])
