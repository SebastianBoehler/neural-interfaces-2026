"""Verify the raw invariance, exact control and fitted relation path."""
import torch

from covariance import BASE, EXTRAS, FIXED, NATIVE, attach, order, parameter_hash, trainable
from source_targets import raw_logits


@torch.no_grad()
def install(model,kind,seed,sample):
    from encoder312 import fingerprint

    model.eval()
    before=(torch.random.get_rng_state(),torch.cuda.get_rng_state())
    state=fingerprint(model)
    original=model.representation(sample)
    logits=raw_logits(model,sample)
    if fingerprint(model)!=state:
        raise ValueError("zero-lane forward changed native SessionBN state")
    layer=attach(model,kind,seed)
    if (not torch.equal(original,model.representation(sample)[:,:NATIVE]) or
            not torch.equal(logits,raw_logits(model,sample)) or
            not torch.equal(logits,model(sample)) or
            trainable(model)!=BASE+(EXTRAS if layer is not None else 0) or
            not torch.equal(before[0],torch.random.get_rng_state()) or
            not torch.equal(before[1],torch.cuda.get_rng_state())):
        raise ValueError("native path at zero, RNG or parameter count changed")
    return layer,{} if layer is None else dict(initial_basis=parameter_hash(layer))


def fit(model,layer,args,base):
    import model322
    from encoder312 import fingerprint

    q, grads = [], {}
    original=model.task if layer is None else model.task.native
    hooks=[original.weight.register_hook(lambda g:q.append(g[:,-FIXED:].count_nonzero()))]
    if layer is not None:
        for name,param in (("left",layer.left),("right",layer.right),("relation",model.task.relation.weight)):
            records=grads[name]=[]
            hooks.append(param.register_hook(lambda g,h=records:h.append(g.abs().max())))
    try:
        stats=model322.train(model,*args)
    finally:
        for hook in hooks:
            hook.remove()
    n=base["steps"]
    state=fingerprint(model)
    if (stats["steps"]!=n or len(q)!=n or torch.stack(q).sum().item() or state==base["initial"] or
            layer is None and state!=base["state"]):
        raise ValueError("native hard control, source work or zero-feature gradients differ")
    for name,traces in grads.items():
        values=torch.stack(traces)
        value=float(values.max())
        if len(traces)!=n or not values.isfinite().all() or value<=1e-8:
            raise ValueError(f"fitted spatial relation path had no gradient for {name}")
        stats[f"{name}_gradient"]=value
    return dict(stats,state=state)


@torch.no_grad()
def diagnostics(model,layer,sample,kind):
    bands=model.body.spectral_filtering(sample)
    left,right=layer.project(bands)
    active=layer.shuffled(right,layer.permutation)
    alternate=torch.as_tensor(order("permuted" if kind=="aligned" else "aligned"),device=right.device)
    counter=layer.shuffled(right,alternate)
    shape=(len(right),3,4,4,120)
    if not torch.equal(active.reshape(shape).sort(-1).values,counter.reshape(shape).sort(-1).values):
        raise ValueError("permutation changed the actual per-block right waveform marginals")
    energy=float((active.double().square().sum(-1)-counter.double().square().sum(-1)).abs().max())
    actual=layer.describe(left,active)
    swapped=layer.describe(left,counter)
    if actual.shape!=(len(left),240) or max(float(z.abs().max()) for z in (actual,swapped))>1+2e-6:
        raise ValueError("normalization did not bound the Pearson relations")
    gains=torch.tensor([.75,1.5,2.,3.],device=left.device)[None,None,:,None]
    invariant=float((actual-layer.describe(left*gains,active/gains)).abs().max())
    feature_delta=float((actual-swapped).abs().max())
    emitted=model.task.relation(actual)
    residual=float(emitted.abs().max())
    logit_delta=float((emitted-model.task.relation(swapped)).abs().max())
    if (not 0<=energy<=1e-8 or not 0<=invariant<=2e-6 or feature_delta<=.05 or
            min(logit_delta,residual)<=1e-4 or not torch.isfinite(emitted).all()):
        raise ValueError(f"relation geometry, invariance or fitted output not viable: {energy}, {invariant}, {feature_delta}, {logit_delta}")
    return dict(energy_error=energy,gain_error=invariant,feature_delta=feature_delta,
                logit_delta=logit_delta,residual=residual,final_basis=parameter_hash(layer))
