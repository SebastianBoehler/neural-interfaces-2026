"""Exact conditional native mix6 geometry for source-only diagnostics."""
import numpy as np
import torch


IDS = tuple(range(47,53))
TOL = 2e-15


def stock_mix(model, mix, members, warm):
    m = mix.copy()
    for family,p in warm.items():
        ids = model.ft.slots[family]
        weights = model.w[1+ids]
        f = (weights[:,None]*np.stack([members[i] for i in ids])).sum(0)
        z = np.clip(f/weights.sum()+p-members[model.ft.member[family]],1e-6,None)
        m = m-f+weights.sum()*z/z.sum()
    return (1-model.w_rest)*m+model.w_rest*model.rest_p


def candidate(stock, frozen, warm, active, weights, rest, pupil):
    a = (weights[:,None]*frozen).sum(0)
    b = weights[0]*frozen[0]+weights[1:].sum()*pupil
    if active:
        m = weights.sum()
        a,b = (np.clip(x/m+warm-frozen[0],1e-6,None) for x in (a,b))
        a,b = m*a/a.sum(),m*b/b.sum()
    return stock+(1-rest)*(b-a)


def differentiable(stock,frozen,warm,active,weights,rest,q):
    a = (weights[:,None]*frozen).sum(1)
    b = weights[0]*frozen[:,0]+weights[1:].sum()*q
    m = weights.sum()

    def route(p):
        z = (p/m+warm-frozen[:,0]).clamp_min(1e-6)
        return torch.where(active[:,None],m*z/z.sum(1,keepdim=True),p)

    return stock+(1-rest)*(route(b)-route(a))


def gradient(packed,weights,rest):
    stock,frozen,warm,active,actual = (packed[k] for k in ("stock_mix","frozen","warm","active","probe48"))
    stock,frozen,warm,active,weights = map(torch.from_numpy,(stock,frozen,warm,active,weights))
    logits = frozen[:,1].double().clamp_min(1e-300).log()
    a,b = (logits.clone().requires_grad_() for _ in range(2))
    q = (a.softmax(1)+b.softmax(1))*.5
    p = differentiable(stock,frozen,warm,active,weights,rest,q)
    if not torch.isfinite(p).all() or (p<=0).any():
        raise ValueError("native row does not support finite class-wise CE gradients")
    np.testing.assert_allclose(p.detach().numpy(),actual,atol=TOL,rtol=0)
    np.testing.assert_allclose(p.detach().sum(1).numpy(),1.,atol=TOL,rtol=0)
    mass = torch.autograd.grad(p.sum(),(a,b),retain_graph=True)
    m = max(float(v.abs().max()) for v in mass)
    if m > TOL:
        raise ValueError("candidate gradient changed total native probability mass")
    norms, jacobian = [], []
    for k in range(3):
        aa,bb = torch.autograd.grad(-p[:,k].log().sum(),(a,b),retain_graph=k<2)
        if not torch.equal(aa,bb) or not aa.isfinite().all():
            raise ValueError("two equal-mass pupils differ in their class-wise gradient")
        norms.append(aa.norm(dim=1).numpy())
        jacobian.append(aa.numpy())
    d = np.stack(norms,axis=1)
    enabled = int(np.count_nonzero(d>0))
    if enabled < .95*d.size:
        raise ValueError("actual pre-RouteB gradient is inactive on over 5% of source cases")
    sigma = np.linalg.svd(np.stack(jacobian,axis=1),compute_uv=False)

    def tangent(rows):
        second = sigma[rows,1]
        if not len(second):
            return None
        return dict(rows=len(second),rank2=int(np.count_nonzero(second>1e-9)),
                    min_sigma2=float(second.min()),median_sigma2=float(np.median(second)),
                    max_sigma2=float(second.max()))

    return dict(cases=int(d.size),active=enabled,min=float(d.min()),median=float(np.median(d)),
                max=float(d.max()),max_mass_gradient=m,mean_seed_gradients_exact=True,
                cold_tangent=tangent(~active.numpy()),warm_tangent=tangent(active.numpy()))
