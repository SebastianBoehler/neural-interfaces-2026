"""Pin all ten §310 source-verified target streams as native core deltas."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

ROOT=Path(os.environ['NI26_WS']).resolve(strict=True)
ORDERS=('within0','within1','within2','within3','full0','full1','runs0','runs1')
ARMS={'Anchor':'anchor','Sharpen':'sharpen','Flatten':'flatten'}
ARCHIVES={o:('coremember.npz' if o in ('within0','within1','full0','runs0') else 'coremember_within2_within3_full1_runs1.npz') for o in ORDERS}

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()


def main():
 people=[f'P{i:02d}' for i in range(1,11)]
 with np.load(ROOT/'data/proteus_windows.npz',allow_pickle=False) as z: subject=np.asarray([str(v).rsplit('/',1)[-1] for v in z['subject']]);n=len(subject)
 with np.load(ROOT/'runs/shuffle_perms.npz',allow_pickle=False) as z: perms={o:z[f'{o[:-1]}|{o[-1]}|all_loso'].astype(np.int64) for o in ORDERS}
 out={(a,o):np.full((n,3),np.nan) for a in ('Rzero',*ARMS) for o in ORDERS}; shards={}
 srcsha=sha(ROOT/'s310-src.sha256'); s305=Path(os.environ['S310_S305']).resolve(strict=True)
 manifest=json.loads((Path(os.environ['S310_S305_ASSEMBLER'])/'runs/s305_converged/manifest.json').read_text())
 for held in people:
  path=ROOT/f'runs/s310/target/{held}.npz'; digest=sha(path)
  if path.with_suffix('.cpu-route').read_text().split()!=['CPU',digest,srcsha]: raise ValueError('route hash differs')
  with np.load(path,allow_pickle=False) as z:
   m=json.loads(str(z['metadata'].item()));rows=z['rows']
   if m['held']!=held or m['held_accuracy_computed'] is not False or not np.array_equal(rows,np.flatnonzero(subject==held)):
    raise ValueError('target person mismatch')
   if manifest['shards'].get(f'{held}.npz')!=m['shard_sha256'] or sha(s305/f'runs/s305_converged_shards/{held}.npz')!=m['shard_sha256']:
    raise ValueError('§305 anchor source differs')
   for order in ORDERS:
    if not np.array_equal(np.sort(perms[order]),np.arange(n)) or m['stats'][order]['anchor_maxdiff']>1e-5 or m['stats'][order]['non_tied_changed'] or m['stats'][order]['updates']<=0:
     raise ValueError('anchor, order, or active update guard failed')
    with np.load(ROOT/'runs'/ARCHIVES[order],allow_pickle=False) as ar: ref=ar[f'{order}|all_loso'][rows]
    for arm,key in ARMS.items():
     p=z[f'{key}|{order}']
     if p.shape!=(len(rows),3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-10,rtol=0): raise ValueError('invalid posterior')
     out[arm,order][rows]=p-ref
    out['Rzero',order][rows]=0.
  if sha(path)!=digest: raise ValueError('target changed during collection')
  shards[path.name]=digest
 if any(not np.isfinite(v).all() for v in out.values()): raise ValueError('incomplete target matrix')
 for order in ORDERS:
  for arm in ('Sharpen','Flatten'):
   if np.array_equal(out[arm,order],out['Anchor',order]): raise ValueError('active update is a no-op')
 target=ROOT/'runs/s301_native/deltas';target.mkdir(parents=True,exist_ok=False); files={}
 for (arm,order),delta in out.items():
  name=f'{arm}_{order}.npz';path=target/name
  meta={'arm':arm,'order':order,'order_sha256':hashlib.sha256(perms[order].tobytes()).hexdigest(),'producer_source_sha256':srcsha}
  with path.open('xb') as f:np.savez_compressed(f,delta=delta,metadata=json.dumps(meta,sort_keys=True))
  files[name]=sha(path)
 with (target.parent/'manifest.json').open('x') as f:json.dump({'files':files,'shards':shards,'producer_source_sha256':srcsha},f,sort_keys=True,indent=2)
 print('32 complete entropy-sign delta files pinned; no score',flush=True)


if __name__=='__main__':main()
