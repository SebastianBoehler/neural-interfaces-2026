"""Apply the archived package count filter and CPU personalization to a new treatment."""
import hashlib
import sys
from pathlib import Path
import numpy as np
from capture import HERE,ORIG,RUN,WORK


def main(name,order,base):
    sys.path[:0]=[str(ORIG/'studies/14nov/rundetect'),str(RUN)]
    import package as K
    import s281_cp as cp
    cp.verify_package_config()
    assert Path(cp.__file__).parent == RUN and np.array_equal(K.ROWS,np.arange(2655))
    feat=WORK/'ni26_refit375_v1_20261117/features.npy'
    with feat.open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()=='fcbd25192a1e1a030c0905f2cc795c21e1984cbb7ff9c1fbd4b5653e21ec9103'
    features=np.load(feat,mmap_mode='r')
    assert features.shape==(2655,8,41,41)
    sessions=[K.C.order_rows(order,s,K.PROTO) for s in K.SESS]
    assert np.array_equal(np.sort(np.concatenate(sessions)),np.arange(2655))
    cache=K.J.hyp_cache(order,'wprior',K.X,K.LAM,sessions=K.SESS,proto=K.PROTO)
    latch=K.latch(order)
    with np.load(HERE/f'{name}_{order}.npz') as z:
        raw=z[f'{base}|V6Fr'].astype(np.float64)
    filtered=K.gate(raw,K.J.filt_joint(raw,cache,'urn'),order,latch)
    result=np.empty_like(raw)
    for rows in sessions:
        model=cp.CP.CausalPers(cp.CONFIG)
        model.feats=features.__getitem__
        for r in rows:
            result[r]=model.step(r,raw[r],filtered[r],False)
        assert model.log['err']==0 and not model.dead and model.stopped is None and model.t==len(rows)
    assert np.isfinite(result).all()
    if name in ('RB','control'):
        with np.load(ORIG/f'runs/s281_v66_cp/{name}_{order}_{base}.npz') as z:
            assert np.array_equal(result.argmax(1),z['scores'].argmax(1)) and np.allclose(result,z['scores'],rtol=0,atol=2e-6)
    np.savez_compressed(HERE/f'final_{name}_{order}_{base}.npz',scores=result)


if __name__=='__main__':
    main(*sys.argv[1:])
