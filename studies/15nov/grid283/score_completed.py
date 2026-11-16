"""Score completed §283/288/290 base screens on the §271 protocol.

Usage: python -I studies/15nov/grid283/score_completed.py
Inputs: runs/grid283_completed/{grid,tsm} and the existing §271 controls.
BA is pooled by person, not the official participant/session/interface metric.
The 3-seed candidate has half the per-family weight of a 6-seed base family.
"""
import itertools, os, sys
from pathlib import Path
import numpy as np
repo=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(repo/'studies/14nov/ensemble_curve'))
sys.argv=sys.argv[:1]
import curve as C
root=repo/'runs/grid283_completed'
new=('fb_dist1','fb_dist2','fb_dist4','eegn_dist1','eegn_dist2','eegn_dist4','fbl_coral10_am50','fbw2_dist2','tsm','tsm_lr3e4','tsm_lr3e3')
old=('fb','fbl','sinc','eegn')
cells=list(itertools.product(C.ORD,C.DRAWS)); N={}; ref=None
for o,d in cells:
 for f in old+new:
  for seed in range(6 if f in ('fb','fbl','sinc') else 3):
   base=Path(C.D) if f in old else root/('tsm' if f.startswith('tsm') else 'grid')
   fn=base/f'ttt18{o}{d}_{f}_base-all_loso-s{seed}.npz'
   with np.load(fn) as z:
    p=z['proba']; ix=z['index']
    if ref is None: ref=ix.copy()
    assert np.array_equal(ix,ref) and p.shape==(len(C.y),3) and np.isfinite(p).all(),str(fn)
    assert (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5),str(fn)
    N[o,d,f,seed]=p.astype(np.float64)
assert np.array_equal(ref,np.arange(len(ref))), 'indices not canonical; inspect before scoring'
print(f'Checked {len(N)} inputs: identical canonical index, finite normalized probabilities, ten people')
R=np.random.default_rng(1); ix=[(R.integers(0,len(C.PEOPLE),len(C.PEOPLE)),R.integers(0,len(cells),len(cells))) for _ in range(1000)]
nets=lambda fams,s:[(f,k) for f in fams for k in range(s)]
ba=lambda ns:np.stack([C.subset_ba(N,o,d,ns) for o,d in cells])
def stat(v):
 lo,hi=np.percentile([v[np.ix_(cc,pp)].mean() for pp,cc in ix],[10,90]); pv=v.mean(0)
 return f'{v.mean():+.2f} [{lo:+.2f},{hi:+.2f}] {sum(pv>0)}/10 {sum(v.mean(1)>0)}/4 P04={pv[list(C.PEOPLE).index("P04")]:+.2f} P10={pv[list(C.PEOPLE).index("P10")]:+.2f}'
base3=nets(('fb','fbl','sinc'),3); base6=nets(('fb','fbl','sinc'),6); b3=ba(base3); b6=ba(base6)
print('Base BA b3',b3.mean(),'b6',b6.mean(),'people',C.PEOPLE)
for f in new:
 p=ba(nets([f],3)); fam='fb' if f.startswith('fb_') or f.startswith('fbw') else 'eegn' if f.startswith('eegn_') else 'fbl' if f.startswith('fbl_') else 'tsm'
 print(f,'BA',round(p.mean(),2),'vs',fam,stat(p-ba(nets([fam],3))) if f!=fam else '-', 'add3',stat(ba(base3+nets([f],3))-b3),'add6',stat(ba(base6+nets([f],3))-b6))
print('Paired distillation effect on add-six marginal vs plain eegn')
e0=ba(base6+nets(['eegn'],3))
for f in ('eegn_dist1','eegn_dist2','eegn_dist4'):
 print(f,stat(ba(base6+nets([f],3))-e0))
print('width x2 vs x1 (both T2) alone',stat(ba(nets(['fbw2_dist2'],3))-ba(nets(['fb_dist2'],3))))
print('width x2 vs x1 (both T2) add-six',stat(ba(base6+nets(['fbw2_dist2'],3))-ba(base6+nets(['fb_dist2'],3))))
