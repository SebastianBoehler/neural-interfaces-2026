"""Apply matched training-only interventions to S1C's fixed feature vector."""
import copy
import hashlib
from types import MethodType

import torch


KINDS = ("original", "zero", "scaled", "dropout")


def mask_plan(batches, seed, held):
    independent = torch.Generator(device="cpu").manual_seed(15015 + 100 * int(held[1:]) + seed)
    cpu, gpu = torch.random.get_rng_state(), torch.cuda.get_rng_state()
    masks = []
    digest = hashlib.sha256()
    for epoch in batches:
        for rows in epoch:
            if len(rows) % 4:
                raise ValueError("source batch does not support exact quarter masking")
            mask = torch.zeros(len(rows), dtype=torch.bool)
            mask[torch.randperm(len(rows), generator=independent)[:len(rows) // 4]] = True
            digest.update(mask.numpy().tobytes())
            masks.append(mask)
    if not torch.equal(cpu, torch.random.get_rng_state()) or not torch.equal(gpu, torch.cuda.get_rng_state()):
        raise ValueError("mask planning consumed the default CPU or CUDA RNG")
    return masks, digest.hexdigest()


def fixed_features(model, x):
    from encoder312 import moments

    return (moments(x) - model.moment_mean) / model.moment_sd


class TrainingPolicy:
    def __init__(self, kind, masks=()):
        if kind not in ("scaled", "dropout"):
            raise ValueError("training policy requires scaled or dropout")
        self.kind, self.masks, self.calls = kind, masks, 0

    def forward(self, model, x):
        representation = model.representation(x)
        q = fixed_features(model, x)
        if self.kind == "scaled":
            q = 0.75 * q
        else:
            plan = self.masks[self.calls]
            if plan.shape != (len(x),):
                raise ValueError("training mask does not match the scheduled batch")
            q = torch.where(plan[:, None], torch.zeros_like(q), q)
        self.calls += 1
        return model.task(torch.cat((representation, q), dim=1))

    def install(self, model):
        return MethodType(lambda actor, x: self.forward(actor, x), model)


@torch.no_grad()
def verify_real_input(model, x, mask, mu, sd):
    """Compare actual classifier input columns on a real source batch."""
    states, result = {}, {}
    for kind in KINDS:
        actor = copy.deepcopy(model).eval()
        actor.moment_mean.copy_(mu)
        actor.moment_sd.copy_(sd)
        actor.skip = kind != "zero"
        if kind in ("scaled", "dropout"):
            actor.forward = TrainingPolicy(kind, [mask]).install(actor)
        capture = []
        handle = actor.task.register_forward_pre_hook(lambda _, args: capture.append(args[0].detach().clone()))
        result[kind] = actor(x)
        handle.remove()
        if len(capture) != 1:
            raise ValueError("original classifier did not consume exactly one direct input")
        states[kind] = capture[0]
    base = states["original"]
    q = base[:, -287:]
    expected = {"zero": torch.zeros_like(q), "scaled": q * 0.75,
                "dropout": torch.where(mask[:, None], torch.zeros_like(q), q)}
    if not torch.count_nonzero(q).item() or not torch.count_nonzero(q[mask]).item():
        raise ValueError("the real source probe did not exercise the fixed branch")
    for kind, target in expected.items():
        if not torch.equal(base[:, :-287], states[kind][:, :-287]) or not torch.equal(states[kind][:, -287:], target):
            raise ValueError(f"the source intervention also changed the waveform branch or wrong q entries: {kind}")
    if any(not torch.isfinite(output).all() for output in result.values()):
        raise ValueError("real source branch probe emitted a nonfinite output")
