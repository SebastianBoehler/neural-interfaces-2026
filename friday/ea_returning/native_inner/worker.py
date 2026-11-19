"""Rebuild the real conditional §321 returning chain with matched EA inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import benchmark_utils.base_solver
from benchmark_utils.streaming import fresh_model
from inputs import ARM_NAMES, BASE, BASELINES, GROUPS, load, native_ft_proof, sha
from package_guard import frame, instrument
from record import ft_clock_restart, load_model
from returning_pair_guard import state, zero
from trace_inputs import ObserveRefit, ReplaceEA


SMOKE_VISIT = "P10|2"
FIXED_PREFIX = 29


def gate(g):
    return (-1, 0., 0.) if g is None else tuple(float(x) for x in g)


def session(template, name, case, n, observe=None):
    torch.manual_seed(3211701)
    model = fresh_model(template)
    immutable = zero(model)["immutable_sha256"]
    ft_clock_restart(model)
    instrument(model)
    model.nets = ReplaceEA(model.nets, case["ea"].get(name))
    refit = ObserveRefit(model)
    inspect = observe(model) if observe is not None else None
    tape = {k: [] for k in ("scores", "labels", "mix", "shadow", "u", "core", "contrast", "identity", "latch",
                            "out_v5", "outer", "unpatched_ea")}
    for t, row in enumerate(case["x"][:n]):
        refit.age = t
        decision, q, trace = frame(model, row)
        if (model.dead or model.cp.dead or model.cf.dead or model.ft.over or model.log6["err"]
                or model.cp.log["err"] or model.route.log["err"] or model.routeB.log["err"]):
            raise ValueError(f"conditional package fallback {name}/{t}: {model.log6['first']}")
        mem = trace["v6"]["members"]
        if (model.nets.n != t + 1 or mem.keys() != model.nets.after.keys()
                or any(not np.array_equal(mem[i], p) for i, p in model.nets.after.items())):
            raise ValueError("native EA intervention is missing from the actual weighted chain")
        refit.emission(trace)
        if inspect is not None:
            inspect(trace, t)
        for k, v in dict(scores=q, labels=decision, mix=trace["mix"], shadow=trace["shadow"],
                         u=trace["u"], core=trace["core1"], contrast=gate(trace["contrast"]),
                         identity=gate(trace["identity"]), latch=model.recB, out_v5=trace["out_v5"],
                         outer=trace["v6"]["mix6"],
                         unpatched_ea=np.stack([model.nets.before[i] for ids in GROUPS.values() for i in ids])).items():
            tape[k].append(v)
    if state(model)["immutable_sha256"] != immutable:
        raise ValueError("native source members mutated")
    tape = {k: np.asarray(v) for k, v in tape.items()}
    for k in ("scores", "mix", "shadow", "core"):
        v = tape[k]
        if v.shape != (n, 3) or not np.isfinite(v).all() or not np.allclose(v.sum(1), 1, atol=2e-6):
            raise ValueError(f"invalid native posterior tape {k}")
    if not np.array_equal(tape["labels"], tape["scores"].argmax(1)):
        raise ValueError("real predict and final posterior disagree")
    np.testing.assert_allclose(tape["u"], case["u"][:n], atol=1e-8, rtol=0)
    np.testing.assert_allclose(tape["core"], case["core"][:n], atol=1e-8, rtol=0)
    np.testing.assert_array_equal(tape["core"].argmax(1), case["core"][:n].argmax(1))
    positive = (tape["contrast"][:, 2] > 0) | (tape["identity"][:, 2] > 0)
    recognition = int(np.flatnonzero(positive)[0]) if positive.any() else -1
    np.testing.assert_array_equal(tape["latch"], np.logical_or.accumulate(positive))
    if recognition != model.log6["rec"] or len(refit.optimizer) != model.ft.refits or model.core1.n != n:
        raise ValueError("native recognition, warm or TangentCore counter differs")
    stats = dict(recognition=recognition, refit_ages=list(refit.targets), warm_ages=list(refit.warm),
                 steps=model.ft.steps, relabels=model.ft.relabels, cp_stopped=model.cp.stopped,
                 fit_actor_ids={k: int(v) for k, v in model.ft.member.items()})
    return tape, refit, stats


def same_prefix(full, replay, n):
    first, old_ft, _ = full
    short, new_ft, _ = replay
    for key in first:
        np.testing.assert_array_equal(first[key][:n], short[key])
    for name in ("targets", "warm", "optimizer"):
        a, b = getattr(old_ft, name), getattr(new_ft, name)
        if set(k for k in a if k < n) != set(b):
            raise ValueError(f"refit prefix omits {name} ages")
        for age, p in b.items():
            if name == "warm":
                for key in p:
                    np.testing.assert_array_equal(p[key], a[age][key])
            elif name == "optimizer":
                if a[age] != p:
                    raise ValueError("prefix optimizer differs")
            else:
                np.testing.assert_array_equal(p, a[age])


def induction(results, case, n, visit, weights):
    zero_tape, _, zero_stats = results["native_zero"]
    np.testing.assert_array_equal(zero_tape["scores"], case["score"][:n])
    np.testing.assert_array_equal(zero_tape["labels"], case["labels"][:n])
    if zero_stats["recognition"] != BASELINES[visit]:
        raise ValueError("native zero recognition differs from the authenticated original run")
    if zero_stats["refit_ages"] != ([20] if visit == SMOKE_VISIT else []) or zero_stats["steps"] != (
            20 if visit == SMOKE_VISIT else 0):
        raise ValueError("native zero did not run the authenticated stock warm scope")
    error = {}
    ids = [i for group in GROUPS.values() for i in group]
    live_w = weights[np.asarray(ids) + 1]
    for arm in ARM_NAMES[1:]:
        cur = results[arm][0]
        np.testing.assert_array_equal(cur["unpatched_ea"], zero_tape["unpatched_ea"])
        np.testing.assert_array_equal(cur["u"], zero_tape["u"])
        inputs = np.stack([case["ea"][arm][slot][:n] for slot, group in GROUPS.items() for _ in group], 1)
        expected = np.einsum("j,njk->nk", live_w, inputs - zero_tape["unpatched_ea"])
        observed = cur["mix"] - zero_tape["mix"]
        error[arm] = float(np.abs(expected - observed).max())
        if error[arm] > 1e-14:
            raise ValueError(f"native EA mixture is not the declared weighted substitution: {error[arm]}")
    return error


def main(visit, smoke):
    if torch.__version__ != "2.14.1+cu130" or torch.cuda.is_initialized():
        raise ValueError("requires the original clean SIF CPU runtime")
    if "neural-interfaces-2026/.venv/" in Path("/proc/self/maps").read_text():
        raise ValueError("host libraries would compromise the native zero")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    who, day = visit.split("|")
    path, fit, names, weights, channels, cases = load(who)
    model = load_model(path, channels, f"native_inner_{who}", "cpu")
    actor_path = Path(type(model).step.__globals__["__file__"]).resolve()
    if actor_path != (path / "submission.py").resolve():
        raise ValueError("wrong actual conditional worker source")
    np.testing.assert_array_equal(model.w, weights)
    native_ft_proof(model, who)
    if set(model.ft.member.values()) & {i for g in GROUPS.values() for i in g}:
        raise ValueError("EA input replacement would replace a warm actor")
    model.js_bank = None
    with np.load(BASE / "neural-interfaces-2026/data/proteus_windows.npz", allow_pickle=False) as data:
        own_rows = (data["subject"] == f"Dreyer2026Proteus/{who}") & (data["session"] == "1")
        prewarm = data["X"][own_rows][:17]
    # Match the original §321 runtime warm-up, then use independent copied sessions.
    for x in prewarm:
        model.predict(torch.from_numpy(x[None]))
    model.reset_state()
    zero(model)
    case, other = cases[day], cases["3" if day == "2" else "2"]
    n = 31 if smoke else len(case["rows"])
    results = {arm: session(model, arm, case, n) for arm in ARM_NAMES}
    error = induction(results, case, n, visit, weights)
    for arm in ARM_NAMES:
        session(model, arm, other, 9)
        same_prefix(results[arm], session(model, arm, case, FIXED_PREFIX), FIXED_PREFIX)
    record, causal = (results[arm] for arm in ARM_NAMES[1:])
    deltas = {slot: float(np.abs(case["ea"]["recording"][slot][:n] - case["ea"]["causal"][slot][:n]).max()) for slot in GROUPS}
    mix_delta = float(np.abs(record[0]["mix"] - causal[0]["mix"]).max())
    if min(deltas.values()) <= .01 or mix_delta <= .001:
        raise ValueError("matched EA inputs did not activate the native inner mixture")
    a, b = record[1], causal[1]
    scopes_match = set(a.targets) == set(b.targets) and set(a.warm) == set(b.warm)
    target_delta = max((float(np.abs(a.targets[t] - b.targets[t]).max()) for t in a.targets.keys() & b.targets.keys()), default=0.)
    warm_delta = {slot: max((float(np.abs(a.warm[t][slot] - b.warm[t][slot]).max())
                            for t in a.warm.keys() & b.warm.keys()), default=0.) for slot in model.ft.member}
    expected_warm = visit == SMOKE_VISIT
    passes = scopes_match and (set(a.targets) == ({20} if expected_warm else set()))
    passes = passes and (not expected_warm or (target_delta > 1e-4 and max(warm_delta.values()) > 1e-5))
    payload = dict(rows=case["rows"][:n], route_names=np.asarray(names))
    stats = {}
    for arm, (tape, ref, run) in results.items():
        payload.update({f"{arm}|{k}": v for k, v in tape.items() if k != "unpatched_ea"})
        for age, p in ref.targets.items():
            payload[f"{arm}|target|{age}"] = p
            payload[f"{arm}|target_rows|{age}"] = case["rows"][:age]
        for age, slots in ref.warm.items():
            for family, p in slots.items():
                payload[f"{arm}|warm|{age}|{family}"] = p
        contrast, identity = tape["contrast"], tape["identity"]
        chosen = np.where(identity[:, 2:3] > contrast[:, 2:3], identity, contrast)
        run["positive_route_owners"] = sorted({names[int(i)] for i in chosen[chosen[:, 2] > 0, 0]})
        stats[arm] = run
    out = Path("native_inner.npz")
    with out.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    out.chmod(0o444)
    receipt = dict(visit=visit, n=n, source=sha(Path(__file__).resolve().parent / "source.sha256"),
                   fold=fit["fold"], fit_sha=fit["fit_sha"],
                   view_sha=fit["view_sha"], context_sha=fit["context_sha"], native_zero_exact=True,
                   native_weighted_delta_max_abs=error, prefix_reset_exact=True, native_slot_delta=deltas,
                   native_mix_delta=mix_delta, strict_target_delta=target_delta, warm_family_delta=warm_delta,
                   matched_warm_qualified=passes, arms=stats, output=sha(out), no_gold=True,
                   scope="conditional native source fit with production rest, eye, core2, event priors and old-OOF exposure")
    path = Path("native_inner.json")
    with path.open("x") as handle:
        json.dump(receipt, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    path.chmod(0o444)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    if not passes:
        raise SystemExit("native matched warm scope or activity did not qualify")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("visit", choices=BASELINES)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.visit, args.smoke)
