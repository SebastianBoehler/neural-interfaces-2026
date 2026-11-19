"""Perform the fixed three-arm source continuation before any external predictions."""
import argparse

import braindecode
import torch

from anchors import DATA_SHA, sha
from encoder312 import fingerprint
from fit_inputs import HARD_DIGESTS, SEED, WORK, load, pupil, rng_digest
from mixture import differentiable
from produce_distill import publish
from source_targets import raw_logits


ARMS = ("hard","nominal","matched")
MAX_RESERVED = 16_000_000_000


def combination(raw,weights,rest,context,p):
    stock,frozen,warm,active = (u[context] for u in raw)
    return differentiable(stock,frozen,warm,active,weights,rest,p)


def cross_entropy(p,y,w):
    if not torch.isfinite(p).all() or (p<=0).any():
        raise ValueError("invalid source softmax probability")
    return -(p.gather(1,y[:,None])[:,0].log()*w).sum()


def models():
    torch.manual_seed(SEED)
    return [pupil(seed) for seed in (0,1)]


@torch.no_grad()
def behavior(actors,x,sessions):
    worst = 0.
    for m,s,_,_,_ in actors:
        a,b = (x[ix]/s for ix in sessions[:2])
        prefix = raw_logits(m,a[:17])
        raw_logits(m,b[:17])
        if not torch.equal(prefix,raw_logits(m,a[:17])):
            raise ValueError("source actor retained other-session state")
        original = raw_logits(m,a)[:17]
        if not torch.equal(prefix.argmax(1),original.argmax(1)):
            raise ValueError("later source waveform changed held-prefix decisions")
        worst = max(worst,float((prefix-original).abs().max()))
    if worst>2e-5:
        raise ValueError("source probability actor lost bounded prefix")
    return worst


def probe(data):
    x,y,w,pack,weights,rest,sessions,schedule,donor,_ = data
    rows = sessions[schedule[0]]
    with torch.no_grad():
        q = torch.stack([raw_logits(m,x[rows]/s).double().softmax(1) for m,s,*_ in models()]).mean(0)
    q.requires_grad_()
    exact = cross_entropy(combination(pack,weights,rest,rows,q),y[rows],w[rows])
    sham = cross_entropy(combination(pack,weights,rest,donor[rows],q),y[rows],w[rows])
    a = torch.autograd.grad(exact,q)[0]
    b = torch.autograd.grad(sham,q)[0]
    delta = float((a-b).abs().max())
    if not a.isfinite().all() or not b.isfinite().all() or delta<=1e-8:
        raise ValueError("nominal and actual fixed source context gradients are indistinguishable")
    return delta


def fit(arm,data,smoke):
    x,y,w,pack,weights,rest,sessions,schedule,donor,binding = data
    actors = models()
    params = [p for m,*_ in actors for p in m.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=3e-5,weight_decay=.001)
    spatial = [next(p for p in m.body.spatial_conv.parameters() if p.requires_grad and p.ndim>=2)
               for m,*_ in actors]
    initial = [v.detach().clone() for v in spatial]
    losses, max_grad = [],0.
    torch.cuda.reset_peak_memory_stats()
    coefficient = 0. if arm in ("hard","zero") else 4.8
    for i in schedule[:4] if smoke else schedule:
        rows = sessions[i]
        q = [raw_logits(m,x[rows]/s).double().softmax(1) for m,s,*_ in actors]
        target,weight = y[rows],w[rows]
        loss = sum(cross_entropy(v,target,weight) for v in q)*.5
        if arm != "hard":
            use = donor[rows] if arm == "nominal" else rows
            alternate = combination(pack,weights,rest,use,(q[0]+q[1])*.5)
            loss = loss+coefficient*cross_entropy(alternate,target,weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for m,*_ in actors:
            p = m.task.weight.grad
            if p is None or p[:,-287:].count_nonzero() or not p.isfinite().all() or not p[:,:-287].count_nonzero():
                raise ValueError("fixed-feature or native task gradient differs")
        gradient = max(float(p.grad.detach().abs().max()) for p in params if p.grad is not None)
        if not loss.isfinite() or gradient<=0 or not all(p.grad is None or p.grad.isfinite().all() for p in params):
            raise ValueError("inactive or invalid actual source optimizer step")
        max_grad = max(max_grad,gradient)
        optimizer.step()
        losses.append(float(loss.detach()))
    steps = len(losses)
    assert steps == (4 if smoke else 112)
    states = [fingerprint(m) for m,*_ in actors]
    if any(h==old for h,(*_,old) in zip(states,actors,strict=True)):
        raise ValueError("a trained seed was unchanged")
    body_movement = min(float((p.detach()-before).abs().max())
                        for before,p in zip(initial,spatial,strict=True))
    if body_movement<=1e-6:
        raise ValueError("native fitted waveform spatial body failed minimum movement")
    for _,_,layer,moments,_ in actors:
        if any(not torch.equal(a,b) for a,b in zip((layer.running_mean,layer.running_var),moments,strict=True)):
            raise ValueError("frozen bounded SessionBN population moments changed")
    prefix = behavior(actors,x,sessions)
    peak = torch.cuda.max_memory_reserved()
    if peak>MAX_RESERVED:
        raise ValueError("source actor memory exceeds the registered cap")
    state = dict(steps=steps,states=states,loss=losses,max_grad=max_grad,prefix_max=prefix,peak_reserved=peak,
                 fitted_movement=body_movement,rng=rng_digest(),coefficient=coefficient)
    if smoke:
        from io import BytesIO
        import hashlib

        f = BytesIO()
        torch.save(optimizer.state_dict(),f)
        state["optimizer"] = hashlib.sha256(f.getbuffer()).hexdigest()
    else:
        for seed,(m,scale,_,_,old) in enumerate(actors):
            path = WORK/"checkpoints"/f"{arm}_{seed}_all_source.pt"
            scope = dict(kind=arm,held="all_source",seed=seed,scale=scale,data=DATA_SHA,source=sha(WORK/"fit.sha256"),
                         initial=old,origin_checkpoint=HARD_DIGESTS[seed],source_only_labels=True,held_labels_opened=False,
                         normalized_population_immutable=True,**binding,**state)
            scope["state"] = states[seed]
            publish(path,m,scope)
            path.chmod(0o444)
            path.with_suffix(".json").chmod(0o444)
    return state


def main(args):
    import json

    assert str(torch.__version__) == "2.14.1+cu126" and braindecode.__version__ == "1.8.1"
    assert torch.cuda.get_device_name() == "NVIDIA A100-PCIE-40GB"
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    data = load()
    if args.arm == "smoke":
        delta = probe(data)
        report = {a:fit(a,data,True) for a in (*ARMS,"zero")}
        for key in ("states","loss","rng","optimizer"):
            if report["hard"][key]!=report["zero"][key]:
                raise ValueError("zero-dose controls did not reproduce actual hard optimization")
        if len({tuple(report[k]["states"]) for k in ARMS})!=3:
            raise ValueError("actual nominal, hard, or conditional fits share state")
        out = WORK/"fit-smoke.json"
        with out.open("x") as f:
            json.dump(dict(source=sha(WORK/"fit.sha256"),passed=True,nominal_true_gradient_delta=delta,
                           arms=report,bindings=data[-1],held_labels_opened=False),f,sort_keys=True,allow_nan=False)
            f.write("\n")
        out.chmod(0o444)
    else:
        q = json.loads((WORK/"fit-smoke.json").read_text())
        assert q["passed"] and q["source"]==sha(WORK/"fit.sha256") and q["bindings"]==data[-1]
        assert not q["held_labels_opened"]
        fit(args.arm,data,False)
    print("PASS F-C31d source",args.arm,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arm",choices=("smoke",*ARMS))
    main(p.parse_args())
