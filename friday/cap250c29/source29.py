"""Fit equal-person class covariances from the native source feature operator."""
import importlib.util
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from files29 import ACTOR_SHA,DATA_SHA,NAMES,PEOPLE,WEIGHTS_SHA,sha,stamp,write


PACKAGE=Path("/package")
DATA=Path("/input.npz")


def owner():
    source=PACKAGE/"submission.py"
    config=PACKAGE/"weights.npz"
    assert sha(source)==ACTOR_SHA and sha(config)==WEIGHTS_SHA
    sys.path.insert(0,"/official/tracks/bci_decoding")
    spec=importlib.util.spec_from_file_location("f29_exact_source",source)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with np.load(config,allow_pickle=False) as z:
        cp=module.CausalPers(z)
    assert len(cp.sos)==8 and (cp.t0,cp.k,cp.w0)==(20,20,90)
    return cp


def whiten(classes):
    g=classes.mean(1)
    s,u=np.linalg.eigh(g)
    assert min(s.ravel())>0
    inv=(u*s[:,None,:]**(-.5))@u.transpose(0,2,1)
    p=inv[:,None]@classes@inv[:,None]
    assert np.isfinite(p).all() and np.linalg.eigvalsh(p).min()>0
    np.testing.assert_allclose(p.mean(1),np.broadcast_to(np.eye(41),(len(p),41,41)),atol=2e-11,rtol=0)
    return p


def main(scope):
    assert torch.__version__=="2.14.1+cu130" and sha(DATA)==DATA_SHA
    cp=owner()
    t=perf_counter()
    with np.load(DATA,allow_pickle=False) as z:
        x=z["X"]
        people=np.array([p.rsplit("/",1)[-1] for p in z["subject"]])
        assert x.shape==(2655,41,480) and x.dtype==np.float32 and tuple(sorted(set(people)))==PEOPLE
        if scope=="smoke":
            rows=[int(np.flatnonzero(people==p)[0]) for p in ("P01","P10")]
            cov=[cp.feats(x[row]) for row in rows]
            for row,c in zip(rows,cov,strict=True):
                assert c.shape==(8,41,41) and np.linalg.eigvalsh(c).min()>0
                np.testing.assert_array_equal(c,cp.feats(x[row]))
            digest=write("fit_smoke.json",dict(passed=True,data=DATA_SHA,source_fit=sha("source-fit.sha256"),
                         actor=ACTOR_SHA,weights=WEIGHTS_SHA,rows=rows,features=stamp(np.stack(cov)),
                         source_labels_opened=False))
            print("PASS F-C29 source smoke",digest,flush=True)
            return
        s=json.loads(Path("fit_smoke.json").read_text())
        assert s["passed"] and not s["source_labels_opened"] and s["source_fit"]==sha("source-fit.sha256")
        assert s["data"]==DATA_SHA and s["actor"]==ACTOR_SHA and s["weights"]==WEIGHTS_SHA
        y=z["y"]  # PROTEUS source labels only. Exclusion is applied before every fold-level aggregate.
        assert y.shape==(len(x),) and np.isin(y,(0,1,2)).all()
        counts=np.zeros((10,3),dtype=np.int64)
        sums=np.zeros((10,8,3,41,41),dtype=np.float64)
        for j,name in enumerate(PEOPLE):
            rows=np.flatnonzero(people==name)
            for i in rows:
                c=int(y[i]); sums[j,:,c]+=cp.feats(x[i]); counts[j,c]+=1
            print("F-C29 source person aggregate",name,counts[j].tolist(),flush=True)
    assert counts.sum()==2655 and (counts>0).all()
    means=sums/counts[:,None,:,None,None]
    pins={}
    for name in NAMES:
        allowed=np.array([p!=name for p in PEOPLE])
        i=np.flatnonzero(people!=name).astype("<i8")
        assert len(i)==counts[allowed].sum() and len(i)==len(x)-int(np.count_nonzero(people==name))
        # Each fit uses only the selected source-person blocks. The held block is never an operand.
        p=whiten(means[allowed].mean(0))
        meta=dict(name=name,data=DATA_SHA,actor=ACTOR_SHA,weights=WEIGHTS_SHA,source_rows=len(i),
                  source_rows_hash=stamp(i),source_persons=[p for p in PEOPLE if p!=name],
                  source_classes=counts[allowed].tolist(),scherer_access=False,matrix=stamp(p))
        path=Path(f"prior_{name}.npz")
        with path.open("xb") as f:
            np.savez_compressed(f,prior=p,source_index=i,metadata=json.dumps(meta,sort_keys=True,allow_nan=False))
        path.chmod(0o444)
        pins[name]=sha(path)
    digest=write("source.json",dict(data=DATA_SHA,actor=ACTOR_SHA,weights=WEIGHTS_SHA,
                 source_fit=sha("source-fit.sha256"),smoke=sha("fit_smoke.json"),priors=pins,scherer_access=False,
                 seconds=perf_counter()-t))
    print("PASS F-C29 all source priors",digest,flush=True)


if __name__=="__main__":
    assert len(sys.argv)==2 and sys.argv[1] in ("smoke","full")
    main(sys.argv[1])
