"""Independent wiring of owning slot/TailX/OwnStage helpers for component parity."""
import ast
from pathlib import Path

import numpy as np


def helpers():
    ns={'np':np}
    for name,wanted in (('route.py',('causal_std',)),('rlib3.py',('slot_mix',)),('tsem.py',('_lda',)),('t_reference.py',('h','tsem_ref'))):
        path=Path.cwd()/name;tree=ast.parse(path.read_text())
        found=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted]
        if len(found)!=len(wanted):raise ValueError('owning reference helper anchor differs')
        exec(compile(ast.Module(body=found,type_ignores=[]),str(path),'exec'),ns)
    return ns


class Reference:
    def __init__(self,model,z,namespace):
        self.ns=helpers();self.ns.update(A=model.routeB.a,G3=3.)
        self.tail=namespace['TailX'](z);self.own=namespace['OwnStage'](z)
        self.cf=type(model.cf)(z);self.cp=type(model.cp)(z)
        self.U=[];self.W=np.asarray(z['route_ps1_W']);self.c=np.asarray(z['route_ps1_c']);self.prior=np.asarray(z['v6_tpri'])
        self.kappa=float(z['v6_tpri_kappa']);self.softmax=namespace['softmax'];self.tpri_checks=0;self.prior_effect=0.
        self.z=z;self.source_heads=0
        original=self.tail.tsem.step;ts=self.tail.tsem
        def checked(u,q,gate):
            U=np.asarray(ts.U);conf=np.asarray(ts.conf);lab=np.asarray(ts.lab);t=len(U);pr=ts.prior
            out=original(u,q,gate);kp,pm=pr if pr is not None else (0.,self.prior[0])
            expected=self.ns['tsem_ref'](U,conf,lab,u,q,gate,t,kp,pm,z)
            if float(np.abs(out-expected).max())>1e-8:raise ValueError('actual TSEM versus owning corrected independent reference differs')
            if pr is not None and ts.state is not None:
                self.tpri_checks+=1
                neutral=self.ns['tsem_ref'](U,conf,lab,u,q,gate,t,0.,pm,z)
                self.prior_effect=max(self.prior_effect,float(np.abs(expected-neutral).max()))
            return out
        self.tail.tsem.step=checked

    def step(self,x,u,mix,core,extra,contrast,identity,recognized):
        chosen=(-1,0.,0.)
        if contrast is not None:chosen=contrast
        if identity is not None and identity[2]>chosen[2]:chosen=identity
        j,r=int(chosen[0]),float(chosen[2]);self.U.append(u.copy());p=mix.copy()
        if r>0:
            v=self.ns['causal_std'](np.asarray(self.U))[-1]
            H=np.stack([self.softmax(v[None]@w.T+c)[0] for w,c in zip(self.W,self.c)])[:,None,:]
            p=self.ns['slot_mix'](mix[None],core[None],H,np.array([r]),np.array([j]))[0];self.source_heads+=1
        self.tail.tsem.prior=(self.kappa*r,self.prior[j]) if r>0 else None
        out=self.tail.step(x,u[:128],p,extra)
        if not self.own.observe(x):raise ValueError('real engineering window degenerate')
        staged=self.own.apply(out,j,r) if r>0 and self.own.cur is not None else None
        if staged is not None:out=staged[0]
        return out,self.cp.step(x,out,self.cf.step(x,out),recognized)
