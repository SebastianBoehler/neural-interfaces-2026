"""§271 pooled-person base screen for completed B1 and §291. Not chain scoring."""
import itertools, sys
from pathlib import Path
import numpy as np
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'studies/14nov/ensemble_curve'))
sys.argv=sys.argv[:1]
import curve as C
cells=list(itertools.product(C.ORD,C.DRAWS))
control=('fb','fbl','sinc','eegn')
coral=tuple(f'{f}_coral{w}' for f in ('fb','fbl','sinc','eegn') for w in (1,3,10,30,100) if not (f=='fb' and w==1))+tuple(f'{f}_coral10_am50' for f in ('fb','fbl','sinc'))
b1=tuple(f'{f}_am{w}' for f in ('fbl','sinc') for w in (25,50,100))
oldb1=('fb_am25','fb_am','fb_am100','fb_coral')
other=('fb_dist1','fb_dist2','fb_dist4','eegn_dist1','eegn_dist2','eegn_dist4','fbw2_dist2','tsm','tsm_lr3e4','tsm_lr3e3')
new=coral+b1+oldb1+other
N={}; ref=None
for o,d in cells:
 for f in control+new:
  for seed in range(6 if f in ('fb','fbl','sinc','fb_coral3','fb_coral10') else 3):
   if f in ('fb_coral3','fb_coral10') and seed>=3: base=repo/'runs/grid286_287_recovery/s291'
   elif f in coral: base=repo/'runs/grid286_287_recovery/grid'
   elif f in b1: base=repo/'runs/grid286_287_recovery/b1'
   elif f in other: base=repo/'runs/grid283_completed'/('tsm' if f.startswith('tsm') else 'grid')
   else: base=Path(C.D)
   fn=base/f'ttt18{o}{d}_{f}_base-all_loso-s{seed}.npz'
   with np.load(fn,allow_pickle=False) as z:
    p=z['proba']; ix=z['index']
    if ref is None: ref=ix.copy()
    assert np.array_equal(ix,ref) and np.array_equal(ix,np.arange(len(C.y))),fn
    assert p.shape==(len(C.y),3) and np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1,atol=1e-5),fn
    N[o,d,f,seed]=p.astype(np.float64)
print('Validated',len(N),'arrays: canonical indices, finite, nonnegative and normalized; 4 cells, 10 people')
br=np.random.default_rng(1)
idx=[(br.integers(0,len(C.PEOPLE),len(C.PEOPLE)),br.integers(0,len(cells),len(cells))) for _ in range(1000)]
ns=lambda fs,seeds:[(f,s) for f in fs for s in seeds]
ba=lambda nets:np.stack([C.subset_ba(N,o,d,nets) for o,d in cells])
def stat(v):
 lo,hi=np.percentile([v[np.ix_(cc,pp)].mean() for pp,cc in idx],[10,90]); pv=v.mean(0)
 return f'{v.mean():+.3f} [{lo:+.3f},{hi:+.3f}] {sum(pv>0)}/10 {sum(v.mean(1)>0)}/4 P04={pv[list(C.PEOPLE).index("P04")]:+.3f} P10={pv[list(C.PEOPLE).index("P10")]:+.3f}'
base6=ns(('fb','fbl','sinc'),range(6)); b6=ba(base6)
print('B1: own-family difference and add-three to six-seed baseline (80% people/cell intervals)')
for f in b1+oldb1[:3]:
 fam=f.split('_')[0]
 print(f,'alone',stat(ba(ns([f],range(3)))-ba(ns([fam],range(3)))),'add3',stat(ba(base6+ns([f],range(3)))-b6))
print('§291: matched seed sets; add and replacement use full six-seed three-family baseline')
for f in ('fb_coral3','fb_coral10'):
 for seeds in (range(0,3),range(3,6),range(6)):
  seeds=list(seeds); a=ns([f],seeds); plain=ns(['fb'],seeds); replace=[n for n in base6 if n not in plain]+a
  print(f,'seeds',seeds,'alone',stat(ba(a)-ba(plain)),'add',stat(ba(base6+a)-b6),'swap',stat(ba(replace)-b6))
print('Available-arm provisional selection rank by add-three (one per family; see B1 and fb_coral baseline names)')
for rank,f in enumerate(sorted(new,key=lambda f:ba(base6+ns([f],range(3))).mean(),reverse=True),1):
 print(rank,f,stat(ba(base6+ns([f],range(3)))-b6))
print('EEGNet coral/distillation incremental vs adding plain EEGNet')
e0=ba(base6+ns(['eegn'],range(3)))
for f in ('eegn_coral1','eegn_coral3','eegn_dist2'):
 print(f,stat(ba(base6+ns([f],range(3)))-e0))
