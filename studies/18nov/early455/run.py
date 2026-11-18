"""Trust the latent block-count model according to its past predictive evidence."""
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit, logsumexp
import score379 as ref
from assets import get_count_filter, verify_sources
from fusion_inputs import restore

MATCH = ref.WORK/'ni26_match456_stages_20261118'
MANIFEST = 'a66ce105e52714a85d5214307799e13dcd5b989a45f7acd96024c3a2d1f99f1b'


def early(package, p, lp, reset, cap):
    # All urn hypotheses predict the current class using prior deliveries only.
    pred = package.J.urn(p, reset, cap)
    loge = np.log(pred.sum(2))
    past = np.vstack([np.zeros(len(lp)), np.cumsum(loge,axis=0)[:-1]])
    marginal = logsumexp(lp+past, axis=1)
    trust = expit(-np.log(4.)+marginal+np.arange(len(p))*np.log(3.))
    weights = np.exp(lp+past-marginal[:,None])
    original = (weights[:,:,None]*pred).sum(1)
    urn_output = original/original.sum(1,keepdims=True)
    likelihood = np.clip(p,1e-9,1)**2
    prior = original/likelihood
    prior /= prior.sum(1,keepdims=True)
    corrected = p*prior
    corrected /= corrected.sum(1,keepdims=True)
    assert abs(trust[0]-.2)<1e-12 and np.allclose(prior[0],1/3,rtol=0,atol=1e-12)
    return trust, corrected, urn_output


def process(order,smoke):
    assert ref.source_manifest(MATCH/'source.sha256')==MANIFEST
    path=MATCH/f'stages_{order}.npz'
    proof=json.loads(path.with_suffix('.json').read_text())
    assert proof['experiment']==MANIFEST and proof['order']==order and proof['output']==ref.sha(path)
    archive=MATCH/'stock'/(order+'.npz')
    with np.load(path) as z:
        raw, filtered, terminal=[z[k] for k in ('raw','filtered','terminal')]
    assert ref.sha(archive)==proof['archived_stock']
    with np.load(archive) as z:
        assert np.array_equal(z['scores'],terminal)
    verify_sources()
    pkg=get_count_filter()
    assert pkg.J.TAU==2. and pkg.J.FLOOR==.5
    _,_,meta,ages=restore()
    hc=pkg.J.hyp_cache(order,'wprior',pkg.X,pkg.LAM,sessions=pkg.SESS,proto=pkg.PROTO)
    latch=pkg.latch(order)
    staged={k:terminal.copy() for k in ('adaptive','fixed')}
    eligible=np.zeros(len(raw),bool)
    for session,(rows,lp,_,reset,cap) in hc.items():
        assert np.array_equal(ages[order][rows],np.arange(len(rows))) and len(set(meta['stream'][rows]))==1
        p=raw[rows]
        trust,q,urn=early(pkg,p,lp,reset,cap)
        stop=latch[session] if latch[session] is not None else len(rows)
        assert np.array_equal(p[:stop],filtered[rows[:stop]])
        assert np.allclose(urn[stop:],filtered[rows[stop:]],rtol=0,atol=1e-12)
        n=min(stop,90)
        used=rows[:n]
        eligible[used]=True
        assert np.array_equal(terminal[used],raw[used])
        for name,w in (('adaptive',trust[:n,None]),('fixed',.20)):
            staged[name][used]=p[:n]+w*(q[:n]-p[:n])
            # The original prior at the first delivery is uniform exactly.
            staged[name][rows[0]]=terminal[rows[0]]
        if smoke:
            count=min(42,n)
            tr,short,un=early(pkg,p[:count],lp,reset[:count],cap[:count])
            assert np.allclose(tr,trust[:count],atol=1e-14,rtol=0)
            assert np.allclose(short,q[:count],atol=1e-12,rtol=0)
            assert np.allclose(un,urn[:count],atol=1e-12,rtol=0)
    assert np.array_equal((terminal+np.zeros_like(terminal)).argmax(1),terminal.argmax(1))
    assert all(np.array_equal(a[~eligible],terminal[~eligible]) for a in staged.values())
    if not smoke:
        dest=Path(order+'.npz')
        np.savez_compressed(dest,stock=terminal.argmax(1),eligible=eligible,
                            **{k:a.argmax(1) for k,a in staged.items()})
        dest.with_suffix('.json').write_text(json.dumps(dict(source=ref.sha(Path('source.sha256')),
            output=ref.sha(dest),input=proof,eligible=int(eligible.sum()))))


if __name__=='__main__':
    process(sys.argv[1],len(sys.argv)>2)
