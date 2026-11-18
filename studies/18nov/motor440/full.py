"""Apply a fixed motor or whole-posterior blend before real count filtering and CP."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax

BASE = Path('/mnt/lustre/work/gehler/gfh527')
FIT = BASE/'ni26_motor440_20261118/fit'
OLD = BASE/'ni26_phase435_20261118'
PARENT = BASE/'ni26_priv401_full_v1_20261118'
sys.path.insert(0, str(OLD))
from score379 import PEOPLE, ORDERS, DATA_SHA, ORDER_SHA, source_manifest, sha

assert sha(PARENT/'full401.py') == '46f6576d7a165e1086810d6ed89cff051fe35f7a3292a55d4481a25dd65996bb'
sys.path.insert(0, str(PARENT))
from full401 import cp_run, setup, INPUT, SCHEDULE

ARMS = ('stock', 'real_motor', 'phase_motor', 'real_all', 'phase_all')


def members(people):
    fit_sha = source_manifest(FIT/'source.sha256')
    probs = {a:np.empty((len(people),3)) for a in ('real','phase')}
    proof = {}
    for p in PEOPLE:
        rows = np.flatnonzero(people == p)
        path = FIT/(p+'_logits.npz')
        m = json.loads((FIT/(p+'.json')).read_text())
        assert m['source_sha'] == fit_sha and m['logits'] == sha(path)
        assert set(m['source_people']) == set(PEOPLE)-{p}
        assert m['archive'] == sha(OLD/(p+'.npz'))
        with np.load(path) as z, np.load(OLD/(p+'.npz')) as old:
            assert np.array_equal(z['index'],rows)
            for a in probs:
                v = softmax(z[a].astype(np.float64), axis=1)
                assert np.array_equal(v.argmax(1),old[a]) and np.isfinite(v).all()
                probs[a][rows] = v
        proof[p] = m['logits']
    return probs, proof


def blend(raw, member, mode):
    stock = raw / raw.sum(1,keepdims=True)
    if mode == 'all':
        new = .9*stock + .1*member
    else:
        assert mode == 'motor'
        motor = .9*stock[:,0] + .1*member[:,0]
        pair = stock[:,1:]/stock[:,1:].sum(1,keepdims=True)
        new = np.column_stack((motor,(1-motor[:,None])*pair))
        np.testing.assert_allclose(new[:,1:]/new[:,1:].sum(1,keepdims=True), pair, atol=1e-12, rtol=0)
    assert np.min(new)>0 and np.max(abs(new-raw))>.01
    np.testing.assert_allclose(new.sum(1),1.,atol=1e-14,rtol=0)
    return new


def main(smoke,j):
    assets,cls,cfg = setup()
    data = assets.CHAIN/'data/proteus_windows.npz'
    assert sha(data)==DATA_SHA and sha(assets.CHAIN/'runs/shuffle_perms.npz')==ORDER_SHA
    with np.load(data) as z:
        people = np.array([p.split('/')[-1] for p in z['subject']])
    probs, proof = members(people)
    assert sha(SCHEDULE/'features.npy') == 'fcbd25192a1e1a030c0905f2cc795c21e1984cbb7ff9c1fbd4b5653e21ec9103'
    features = np.load(SCHEDULE/'features.npy',mmap_mode='r')
    package = assets.get_count_filter()
    with np.load(INPUT/'inputs.npz') as z, np.load(INPUT/'outputs.npz') as native:
        assert tuple(z['orders']) == ORDERS
        o = ORDERS[j]
        raw, filt0, gold = [z[o+'|'+k] for k in ('raw','filtered','expected')]
        sessions = [package.C.order_rows(o,s,package.PROTO) for s in package.SESS]
        rr_all = np.split(z[o+'|index'],z[o+'|bounds'][1:-1])
        assert len(sessions)==len(rr_all) and all(np.array_equal(a,b) for a,b in zip(sessions,rr_all))
        assert np.array_equal(np.sort(np.concatenate(sessions)), np.arange(len(people)))
        cache = package.J.hyp_cache(o,'wprior',package.X,package.LAM,sessions=package.SESS,proto=package.PROTO)
        latch = package.latch(o)
        out = {}
        for a in ARMS:
            if a == 'stock':
                new = raw
            else:
                arm,mode = a.split('_')
                new = blend(raw,probs[arm],mode)
            filtered = package.gate(new,package.J.filt_joint(new,cache,'urn'),o,latch)
            if a == 'stock':
                assert np.array_equal(filtered,filt0)
            pred = np.full(len(raw),-1,dtype=np.int8)
            for rr in sessions[:1] if smoke else sessions:
                p = cp_run(cls,cfg,rr,features,new,filtered)
                pred[rr] = p.argmax(1)
                if a == 'stock':
                    assert np.array_equal(p,gold[rr]) and np.array_equal(pred[rr],native[o+'|stock'][rr].argmax(1))
                if smoke and a == 'phase_motor':
                    prefix = cp_run(cls,cfg,rr[:131],features,new,filtered)
                    stop = cp_run(cls,cfg,rr[:131],features,new,filtered,63)
                    assert np.array_equal(prefix,p[:131]) and np.array_equal(stop,filtered[rr[:131]])
            assert smoke or np.all(pred>=0)
            out[a] = pred
    name = 'smoke' if smoke else 'order_'+str(j)
    np.savez_compressed(name+'.npz',**out)
    Path(name+'.json').write_text(json.dumps(dict(source=sha(Path('source.sha256')), folds=proof,
                                                 order=o, output=sha(Path(name+'.npz')))))


if __name__ == '__main__':
    main(sys.argv[1]=='smoke', int(sys.argv[2]))
