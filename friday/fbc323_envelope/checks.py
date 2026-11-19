"""Prove phase invariance and active matched trainable waveform branches."""
import torch

from envelope import BASE,EXTRA,FIXED,NATIVE,attach,parameter_hash,trainable
from source_targets import raw_logits


@torch.no_grad()
def install(model,kind,seed,sample):
    from encoder312 import fingerprint

    model.eval()
    before=(torch.random.get_rng_state(),torch.cuda.get_rng_state())
    state=fingerprint(model)
    saved=model.representation(sample)
    logits=raw_logits(model,sample)
    if fingerprint(model)!=state:
        raise ValueError("eval-only source probe changed native buffers")
    branch=attach(model,kind,seed)
    if (not torch.equal(saved,model.representation(sample)[:,:NATIVE]) or
            not torch.equal(logits,raw_logits(model,sample)) or not torch.equal(logits,model(sample)) or
            trainable(model)!=BASE+(EXTRA if branch is not None else 0) or
            not torch.equal(before[0],torch.random.get_rng_state()) or
            not torch.equal(before[1],torch.cuda.get_rng_state())):
        raise ValueError("extra zero path changed native logit or source RNG")
    return branch,{} if branch is None else dict(initial_branch=parameter_hash(branch))


def fit(model,branch,args,expected):
    import model322
    from encoder312 import fingerprint

    q,guards=[],{}
    native=model.task if branch is None else model.task.native
    hooks=[native.weight.register_hook(lambda g:q.append(g[:,-FIXED:].count_nonzero()))]
    if branch is not None:
        for name,param in (("spatial",branch.spatial),("first",branch.first.weight),
                           ("second",branch.second.weight),("readout",model.task.emit.weight)):
            tape=guards[name]=[]
            hooks.append(param.register_hook(lambda g,h=tape:h.append(g.abs().max())))
    try:
        stats=model322.train(model,*args)
    finally:
        for h in hooks:
            h.remove()
    state=fingerprint(model)
    if (len(q)!=expected["steps"] or stats["steps"]!=expected["steps"] or torch.stack(q).sum().item() or
            state==expected["initial"] or branch is None and state!=expected["state"]):
        raise ValueError("source updates, exact hard control or fixed-feature gradients differ")
    for name,history in guards.items():
        actual=torch.stack(history)
        value=float(actual.max())
        if len(history)!=expected["steps"] or not actual.isfinite().all() or value<=1e-8:
            raise ValueError(f"source activity missing from {name}")
        stats[f"{name}_gradient"]=value
    return dict(stats,state=state)


@torch.no_grad()
def evidence(model,branch,sample):
    spec=branch.spectrum(model.body.spectral_filtering(sample))
    analytic,real=branch.powers(spec)
    base=real.sum(-1)
    energy=float(((analytic.sum(-1)-base).abs()/base.clamp_min(1e-8)).max())
    turn=torch.polar(real.new_tensor(1.),real.new_tensor(.7))
    start=tuple(branch.tokens(p) for p in (analytic,real))
    shifted=tuple(branch.tokens(p) for p in branch.powers(spec*turn))
    phase=[float((a-b).abs().max()) for a,b in zip(start,shifted,strict=True)]
    values=[model.task.emit(branch.features(x)) for x in start]
    movement=float((values[0]-values[1]).abs().max())
    if (not 0<=energy<=2e-6 or not 0<=phase[0]<=2e-6 or not phase[1]>=1e-3 or
            not movement>1e-4 or not all(v.isfinite().all() for v in values)):
        raise ValueError(f"real analytic path phase/energy/gradient proof failed: {energy}, {phase}, {movement}")
    return dict(energy_error=energy,analytic_phase_error=phase[0],real_phase_delta=phase[1],
                logit_swap=movement,final_branch=parameter_hash(branch))
