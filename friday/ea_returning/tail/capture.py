"""Observe the genuine §431 T branch and independently verify its consumers."""
from contextlib import contextmanager

import numpy as np

from returning_owner import PROTO, SESSIONS


def same(a,b):
    return np.array_equal(a,b,equal_nan=True) if a.dtype.kind=='f' else np.array_equal(a,b)


def exact(a,b):
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)


class Capture:
    def __init__(self,ctx,idx):
        self.ctx,self.idx=ctx,idx
        self.gates,self.pending,self.slots,self.head_view,self.personal={},{},{},{},{}
        self.prior,self.preown,self.staged=None,None,None
        self.source_by_fold={}
        self.prefix=False

    @contextmanager
    def observe(self):
        ctx=self.ctx
        rr,k=ctx.A.R3,ctx.K
        rmap,bio=rr.rmap,ctx.ns['bio_stream']
        score,gate,slot=ctx.ns['R2'].scores_session,rr.gate_model,rr.slot_mix
        prior,staged,member=ctx.ns['tsem_lik_prior'],k.staged,k.members

        def active():
            return ctx.ns['KAPPA']==20.0

        def scores(x,p,gn,q,*args,**kw):
            value=score(x,p,gn,q,*args,**kw)
            if active() and not self.prefix:
                cut=70
                xx=x[:cut+1].copy(); pp=p[:cut+1].copy()
                xx[cut]=-xx[cut]; pp[cut]=np.roll(pp[cut],1)
                assert not np.array_equal(xx[cut],x[cut])
                assert same(score(xx,pp,gn,q,*args,**kw),value[:cut+1])
                self.prefix=True
            return value

        def prepare_gate(p,f,oof):
            value=gate(p,f,oof)
            if active():
                assert p==PROTO and f in ('P04_s23','P10_s23')
                assert f not in self.source_by_fold
                self.source_by_fold[f]=value[4]
                for s,(X,_,rows) in value[0].items():
                    assert s in SESSIONS and s not in self.head_view
                    self.head_view[s]=(np.array(rows), X[:,:256].copy(), f)
            return value

        def contrast(scores,*args,**kw):
            r,j=rmap(scores,*args,**kw)
            if active():
                assert not self.pending and args==('lin05',)
                valid=j>=0; best=np.full(len(j),np.nan)
                best[valid]=scores[np.flatnonzero(valid),j[valid]]
                self.pending.update(contrast_r=r.copy(),contrast_j=j.copy(),contrast_score=best)
            return r,j

        def identity(p,f,s,rows,names,t0):
            value=bio(p,f,s,rows,names,t0)
            if active():
                assert p==PROTO and t0==10 and self.pending and s not in self.gates
                br,bj,bs=value
                assert self.head_view[s][2]==f and np.array_equal(self.head_view[s][0],rows)
                self.gates[s]=dict(rows=np.array(rows), names=np.array(names), **self.pending,
                                   bio_r=br.copy(), bio_j=bj.copy(), bio_score=bs.copy())
                self.pending.clear()
            return value

        def mix(m,c,H,r,j):
            value=slot(m,c,H,r,j)
            if active():
                p,s=next(key for key,val in ctx.A._H.items() if val is H)
                assert p==PROTO and s not in self.slots and ctx.A.R3.A==1/16 and ctx.A.R3.G3==3
                hit=np.flatnonzero(r>0)
                h=np.zeros_like(m)
                h[hit]=H[j[hit],hit]
                v=r[:,None]/16
                exact((m-v*c+3*v*h)/(1-v+3*v),value)
                self.slots[s]=dict(mix=m.copy(),core=c.copy(),head=h,mixr=value.copy())
            return value

        def pri(mu,kp):
            actual=prior(mu,kp)
            if not active():
                return actual
            assert self.prior is None
            self.prior=dict(personal_prior=mu.copy(), kappa=kp.copy())
            def lik(p,s,x,*args,**kw):
                value=actual(p,s,x,*args,**kw)
                lk,age=value
                self.prior.update(em=p.copy(), tsem_pc=x.copy(), tsem_lik=lk.copy(), tsem_age=age.copy())
                return value
            return lik

        def own(p,s,q):
            values=member(p,s,q)
            if active():
                by_person=self.personal.setdefault(s,{})
                by_person.setdefault(q,values)
            return values

        def stage(inputs):
            value=staged(inputs)
            if active():
                assert self.preown is None
                self.preown=inputs[PROTO].copy()
                self.staged=value[0][PROTO].copy()
            return value

        ctx.ns['R2'].scores_session,rr.gate_model,rr.rmap,ctx.ns['bio_stream']=scores,prepare_gate,contrast,identity
        rr.slot_mix,ctx.ns['tsem_lik_prior'],k.staged,k.members=mix,pri,stage,own
        try:
            yield
        finally:
            ctx.ns['R2'].scores_session,rr.gate_model,rr.rmap,ctx.ns['bio_stream']=score,gate,rmap,bio
            rr.slot_mix,ctx.ns['tsem_lik_prior'],k.staged,k.members=slot,prior,staged,member

    def seal(self,pp):
        ctx=self.ctx; idx=self.idx
        assert self.prefix and not self.pending and self.gates.keys()==self.slots.keys()==self.personal.keys()==SESSIONS
        inv=np.argsort(pp)
        expected={}
        stages={k:v[inv] for k,v in {**self.prior,'preown':self.preown,'staged':self.staged}.items()}
        rank=np.argsort(idx[pp])
        assert np.array_equal(idx[pp][rank],idx)
        for s in sorted(SESSIONS):
            g=self.gates[s]; rows=np.searchsorted(idx,g['rows'])
            assert np.array_equal(idx[rows],g['rows'])
            owner=s.rsplit('/',1)[-1].split('|')[0]
            h=self.head_view[s]
            w=g['bio_r']>g['contrast_r']
            rr=np.where(w,g['bio_r'],g['contrast_r'])
            jj=np.where(w,g['bio_j'],g['contrast_j'])
            rec,j=ctx.K.REC[PROTO,s]
            assert np.array_equal(rr,rec) and np.array_equal(jj,j)
            hit=rr>0; before=np.flatnonzero(hit)[0]
            recognised=np.arange(len(rows))>=before
            gaps=np.flatnonzero(recognised & ~hit)
            names=g['names']; current=names[jj[hit]][0]
            assert str(current).endswith(owner) and (names[jj[hit]]==current).all()
            assert set(self.personal[s])=={str(current)}
            own=self.personal[s][str(current)]
            logs=sum(np.log(np.clip(own[k],1e-3,1)) for k in 'TCS')/3
            raw=stages['preown'][rows]
            combined=np.log(np.clip(raw[hit],1e-9,1))+.10*rr[hit,None]*logs[hit]
            e=np.exp(combined-combined.max(1,keepdims=True))
            expected_out=raw.copy(); expected_out[hit]=e/e.sum(1,keepdims=True)
            exact(expected_out,stages['staged'][rows])
            assert np.array_equal(stages['kappa'][rows],20.*rr)
            prior=stages['personal_prior'][rows]
            per=self.source_by_fold[h[2]][str(current)]
            source_prior=np.mean([np.stack([ctx.A.R3.stdz(X[:,:128])[y==c].mean(0) for c in range(3)]) for X,y in per],0)
            exact(prior[hit][0],source_prior)
            assert not np.any(prior[~hit]) and np.any(prior[hit]) and (prior[hit]==prior[hit][0]).all()
            assert np.any(stages['tsem_age'][rows][hit]>0)
            own_change=float(np.abs(stages['staged'][rows]-raw).max())
            assert own_change>1e-6
            short=s.rsplit('/',1)[-1]; hk=f'{PROTO}|{h[2]}'
            head=ctx.A._HEADS[hk][str(current)]
            assert np.array_equal(head.classes_,np.arange(3))
            expected_head=head.predict_proba(ctx.A.RR.causal_std(h[1]))
            exact(expected_head[hit],self.slots[s]['head'][hit])
            g.update(winner_r=rr,winner_j=jj,recognised=recognised,owner=np.array(str(current)),pc256=h[1],
                     source_head_w=head.coef_,source_head_c=head.intercept_,own_logs=logs)
            g.update(self.slots[s])
            for key,v in g.items():
                stages[f'{short}|{key}']=v
            expected[short]=dict(owner=owner,age=int(before),routed_rows=int(hit.sum()),
                recognised_rows=int(recognised.sum()),post_latch_zero_ages=gaps.tolist(),
                own_max_abs=own_change,head_tsem_view_max_abs=float(np.abs(g['pc256'][:,:128]-stages['tsem_pc'][rows]).max()))
        assert sum(v['recognised_rows'] for v in expected.values())==704
        return stages,expected
