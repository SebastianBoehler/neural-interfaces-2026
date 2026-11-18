"""Rebuild the original §431 mixer, shadow and routed tails."""
import json
import numpy as np
from returning_owner import BANK,FIRSTS,ORDER,PROTO,ROOT,SESSIONS,STACK,digest

ARMS=('base','ER','T')


def prepare(ctx,idx,stream):
    assert np.array_equal(ctx.IDX,idx)
    r6=ctx.R6
    oof=r6['oof_rec'](STACK)
    perm=r6['perm'](ORDER)
    with np.load(ROOT/'runs/shuffle_perms.npz') as p:
        assert np.array_equal(idx[perm[PROTO]],p['full|0|seen_session'])
    for s,first in FIRSTS.items():
        delivery=idx[perm[PROTO]]
        assert int(delivery[stream[delivery]==s][0])==first
    ds=r6['member_deltas'](ORDER)
    eyes=[]
    rest=[]
    full0_hash=digest(idx[perm[PROTO]].astype(np.int64))
    for seed in range(9):
        f=ROOT/f'runs/ocushuf_ss/{ORDER}/oc_nomix-{PROTO}-s{seed}.npz'
        with np.load(f) as z:
            assert str(z['order_sha'])==full0_hash
        eyes.append(r6['ld'](str(f)))
    for seed in range(3):
        f=ROOT/f'runs/restshuf/{ORDER}/rest4_sh_r1-{PROTO}-s{seed}.npz'
        rest.append(r6['ld'](str(f)))
    count9=(None,np.log(np.clip(np.mean(eyes,0),1e-3,1)),np.mean(rest,0))
    count3=(None,np.log(np.clip(np.mean(eyes[:3],0),1e-3,1)),count9[2])
    return oof,perm,ds,count9,count3


def fresh(idx,delivery):
    values={}
    for slot,member,n in ((1,'fbcnet_ea_tf',5),(3,'shallow_ea',6)):
        with np.load(BANK/f'{member}_seen_session_orders.npz',allow_pickle=False) as z:
            assert np.array_equal(z['index'],idx) and np.array_equal(z['seeds'],np.arange(n))
            names=z['orders'].tolist()
            k=names.index(ORDER)
            assert len(names)==8 and np.array_equal(z['delivery'][k],delivery)
            causal=z['proba']; recording=z['recording']
            assert causal.shape==(n,8,762,3) and recording.shape==(n,762,3)
            arrays=(np.mean(recording,axis=0),np.mean(causal[:,k],axis=0))
            assert all(np.isfinite(a).all() and np.allclose(a.sum(1),1,atol=2e-6,rtol=0) for a in arrays)
            assert np.max(abs(arrays[0]-arrays[1]))>.05
            values[slot]=arrays
    return ({s:a[0] for s,a in values.items()},{s:a[1] for s,a in values.items()})


def permutation_reset(perm,idx,stream):
    rows=idx[perm[PROTO]]
    order=tuple(dict.fromkeys(stream[rows]))
    assert set(order)==SESSIONS
    reverse=np.concatenate([perm[PROTO][stream[rows]==s] for s in order[::-1]])
    assert np.array_equal(np.sort(reverse),np.arange(len(idx)))
    return {**perm,PROTO:reverse}


def run(ctx,prep,perm,aux,replace_slots,stream):
    oof,_,deltas,_,_=prep
    pp=perm[PROTO]
    reset=(ctx.ns['_BIO'],ctx.A._H,ctx.K.STREAMS,ctx.K.REC,ctx.K.NM)
    for cache in reset:
        cache.clear()
    ctx.S.apply(perm)
    ctx.K.RANK['cur']={p:ctx.K._rank(ctx.K.IDX[p][v]) for p,v in perm.items()}
    ctx.R6['apply_members'](deltas,pp)
    slot_address=f'{STACK}|Ps|{PROTO}'
    arrays=ctx.Z[slot_address]
    expected={}
    for slot in (1,3):
        if replace_slots is not None:
            arrays[slot]=replace_slots[slot][pp]
        expected[slot]=arrays[slot].copy()
    assert set(expected)=={1,3}

    a,gm=ctx.A,ctx.gm
    keys=('sess','GE','pc128')
    tape={k:ctx.Z[f'{k}|{PROTO}'].copy() for k in keys}
    tape['session']=stream[ctx.IDX[pp]]
    assert set(tape['session'])==SESSIONS
    base,_=a.mix(STACK,PROTO,a.FAMS)
    b=a.M.rebal(base,PROTO,None)
    g=gm.gate(b,PROTO)
    e=gm.em_g(b,PROTO,g)
    shadow=gm.tsem_g(e,PROTO,g)
    tape.update(mix=base,rebal=b,gate=g,em=e,shadow=shadow)
    old_mix,old_shadow=a.mix,a.shadow
    count=[0,0]
    def observe_mix(bn,p,fams,extra=(),**kw):
        assert bn==STACK and p==PROTO and fams==a.FAMS
        assert all(np.array_equal(a.members(bn,p)[k],v) for k,v in expected.items())
        count[0]+=1
        return old_mix(bn,p,fams,extra,**kw)

    def observe_shadow(m,p):
        x,y=old_shadow(m,p)
        assert p==PROTO and np.array_equal(base,m)
        assert np.array_equal(x,e) and np.array_equal(y,shadow)
        count[1]+=1
        return x,y

    outputs,routes={},None
    a.mix,a.shadow=observe_mix,observe_shadow
    try:
        for label in ARMS:
            out,rr=ctx.system(STACK,label,pp,oof,aux,None)
            if routes is None:
                routes=rr
            assert routes.keys()==rr.keys()
            for key,(rows,ramp,person) in rr.items():
                original=routes[key]
                assert np.array_equal(rows,original[0]) and np.array_equal(ramp,original[1]) and person==original[2]
                if label=='T':
                    pre=rows[:int(np.flatnonzero(ramp>0)[0])]
                    np.testing.assert_allclose(out[pre],outputs['ER'][pre],atol=1e-12,rtol=0)
                    out[pre]=outputs['ER'][pre]
            assert np.isfinite(out).all()
            outputs[label]=out
    finally:
        a.mix,a.shadow=old_mix,old_shadow
    assert count==[6,3]
    routing={}
    for s in sorted(SESSIONS):
        key=(PROTO,s)
        if key not in routes:
            routing[s]=dict(owner=None,age=None)
            continue
        rows,ramp,person=routes[key]
        assert np.array_equal(np.flatnonzero(tape['session']==s),rows)
        owners,gi,_=ctx.K.NM[key]
        assert np.array_equal(ctx.IDX[pp[rows]],gi)
        routing[s]=dict(owner=person,age=int(np.flatnonzero(ramp>0)[0]))
    inv=np.argsort(pp)
    stored={k:v[inv] for k,v in outputs.items()}
    bank={**{k:v[inv] for k,v in tape.items()},
          **{f'slot{k}':v[inv] for k,v in expected.items()}}
    return stored,bank,routing
