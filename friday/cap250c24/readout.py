"""Keep the native CSP fit and expose a timing contrast to its LDA readout."""

from copy import deepcopy
from hashlib import sha256

import numpy as np
from scipy.signal import sosfiltfilt


ARMS = ("stock", "identity", "temporal", "sham")
SIGN_HASH = "74d2d61251cd13ee5845cacb2998f350919abcd91fc121a97e8c332ed83c333d"


def stamp(a):
    return sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sign_tape():
    seeds=np.random.default_rng(24019).choice(np.array([-1,1],dtype=np.int8),size=256)
    signs=np.stack((seeds,-seeds),1).ravel()
    assert stamp(signs)==SIGN_HASH and (signs.reshape(-1,2).sum(1)==0).all()
    return signs


class Readout:
    def __init__(self,cp,mode):
        assert mode in ARMS
        self.actor=deepcopy(cp)
        self.actor.reset_state()
        assert not self.actor.P and self.actor.t==0
        self.mode=mode
        self.active=mode in ("temporal","sham")
        self.signs=sign_tape() if mode=="sham" else np.ones(512,dtype=np.int8)
        self.parts,self.whole,self.fits,self.likelihood=[],[],[],[]
        self.extra_usage,self.contrast=0,0.
        self._feat=self.actor.feats
        self._fit=self.actor.refit
        self._predict=self.actor.predict
        self.actor.feats=self.observe_features
        self.actor.refit=self.refit
        self.actor.predict=self.predict

    def observe_features(self,x):
        if self.mode=="stock":
            full=self._feat(x)
        else:
            value=np.asarray(x,dtype=np.float64)
            assert value.shape==(41,480)
            filtered=[sosfiltfilt(sos,value,axis=-1) for sos in self.actor.sos]
            full=np.stack([self.actor.cov.transform(w[None])[0] for w in filtered])
            parts=np.stack([self.actor.cov.transform(np.stack((w[:,:240],w[:,240:]))) for w in filtered])
            assert parts.shape==(len(self.actor.sos),2,41,41) and np.isfinite(parts).all()
            self.parts.append(parts)
        self.whole.append(stamp(full))
        return full

    def vector(self,filters,whole,crops,rows):
        n=len(rows)
        body=[self.actor.lv(W,whole[:,b]) for b,W in enumerate(filters)]
        pair=[self.actor.lv(W,crops[:,b].reshape(n*2,41,41)).reshape(n,2,-1) for b,W in enumerate(filters)]
        full=np.concatenate(body,axis=1)
        delta=np.concatenate([v[:,1]-v[:,0] for v in pair],axis=1)
        assert full.shape==delta.shape and np.isfinite(full).all() and np.isfinite(delta).all()
        self.contrast=max(self.contrast,float(np.abs(delta).max()))
        return np.concatenate((full,delta*self.signs[rows,None]),axis=1)

    def refit(self,at):
        native=self._fit(at)
        assert len(self.actor.C)==len(self.actor.P)==at+1
        P=np.asarray(self.actor.P[:at])
        y=P.argmax(1)
        selected=self.actor.select(y,P.max(1))
        assert selected.max()<at and (selected[1:]>selected[:-1]).all()
        filters,lda=native if native is not None else ((),None)
        width=0 if lda is None else int(lda.n_features_in_)
        self.fits.append(dict(age=at,target=stamp(P),selected=stamp(selected),n_rows=len(selected),
                              projection=[stamp(w) for w in filters],whole_width=width,
                              native_classes=[] if lda is None else lda.classes_.tolist()))
        if not self.active or native is None:
            return native
        assert len(self.parts)==at+1
        full=np.stack(self.actor.C[:at])[selected]
        crops=np.stack(self.parts[:at])[selected]
        X=self.vector(filters,full,crops,selected)
        assert X.shape==(len(selected),2*width)
        augmented=self.actor.LDA(solver="lsqr",shrinkage="auto").fit(X,y[selected])
        assert augmented.n_features_in_==X.shape[1] and augmented.coef_.shape[1]==X.shape[1]
        np.testing.assert_array_equal(lda.classes_,augmented.classes_)
        return filters,augmented

    def predict(self,current):
        at=self.actor.t-1
        if not self.active or self.actor.model is None:
            q=self._predict(current)
        else:
            filters,lda=self.actor.model
            assert len(self.parts)==at+1 and current is self.actor.C[at]
            f=self.vector(filters,current[None],self.parts[at][None],np.array([at]))
            assert f.shape==(1,lda.n_features_in_) and f.shape[1]==2*self.fits[-1]["whole_width"]
            q=np.zeros(3)
            q[lda.classes_]=lda.predict_proba(f)[0]
            self.extra_usage+=1
        assert np.isfinite(q).all() and q.shape==(3,) and abs(float(q.sum())-1)<1e-13
        self.likelihood.append((at,q.copy()))
        return q

    def consume(self,eeg,raw,count,recognized):
        a=self.actor
        out=np.asarray([a.step(x,p,o,r) for x,p,o,r in zip(eeg,raw,count,recognized,strict=True)])
        assert not a.dead and not a.log["err"] and a.t==len(eeg) and len(self.fits)==a.refits
        stop=None if not any(recognized) else int(np.flatnonzero(recognized)[0])
        limit=len(eeg) if stop is None else stop
        assert a.stopped==stop and len(self.whole)==limit and len(self.likelihood)==a.fused
        assert len(self.parts)==(0 if self.mode=="stock" else limit)
        assert [r["age"] for r in self.fits]==list(range(a.t0,limit,a.k))
        idx=np.asarray([i for i,_ in self.likelihood],dtype=np.int64)
        if len(idx):
            q=np.stack([p for _,p in self.likelihood])
            weight=a.cap*np.clip((idx-a.w0+1)/a.R,0.,1.)
            check=np.log(np.clip(count[idx],1e-6,1))+weight[:,None]*np.log(np.clip(q,1e-6,1))
            np.testing.assert_array_equal(out[idx],check)
        else:
            q=np.empty((0,3),dtype=np.float64)
        bypass=np.ones(len(eeg),dtype=bool)
        bypass[idx]=False
        np.testing.assert_array_equal(out[bypass],count[bypass])
        return out,idx,q,dict(refits=a.refits,fusions=a.fused,stop=stop,features=limit,
                              fitted=self.fits,feature_hash=self.whole,extra_usage=self.extra_usage,
                              max_contrast=self.contrast)


def evaluate(cp,eeg,raw,count,recognized,expected):
    outcomes={arm:Readout(cp,arm).consume(eeg,raw,count,recognized) for arm in ARMS}
    stock=outcomes["stock"]
    np.testing.assert_array_equal(stock[0],expected)
    np.testing.assert_array_equal(outcomes["identity"][0],stock[0])
    np.testing.assert_array_equal(outcomes["identity"][2],stock[2])
    proof=dict()
    records=stock[3]
    for arm,(out,indices,likelihood,info) in outcomes.items():
        for name in ("refits","fusions","stop","features","fitted","feature_hash"):
            assert info[name]==records[name],(arm,name)
        np.testing.assert_array_equal(indices,stock[1])
        proof[arm]=dict(**info, max_q_change=float(np.abs(likelihood-stock[2]).max(initial=0.)),
                        max_score_change=float(np.abs(out-expected).max(initial=0.)))
        if arm in ("temporal","sham") and info["extra_usage"]:
            assert info["max_contrast"]>.05 and proof[arm]["max_q_change"]>0.
    if records["stop"] is not None:
        for result in outcomes.values():
            np.testing.assert_array_equal(result[0][records["stop"]:],expected[records["stop"]:])
    scores=np.stack([outcomes[arm][0] for arm in ARMS])
    return scores,proof
