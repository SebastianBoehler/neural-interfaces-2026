"""Replay the authenticated native session with one scoped waveform readout slot."""

import argparse
import importlib.util
import json
from pathlib import Path
from types import FunctionType

import numpy as np
import torch

from assets import meta, sha
from geometry import EEG, waveform_proof
from head import RecognizedHead
from inputs import BASE, BASELINES, load, native_ft_proof
from record import load_model
from returning_pair_guard import zero


INNER = BASE / "ni26_friday_returning_native_inner_v2_20261119"
AGG = BASE / "ni26_friday_returning_native_inner_collect_v1_20261119/returning_native_inner.json"
AGG_SHA = "1e656138b910827defc26de4cc7854b5735908d31bfea4d47d111c73f9675c64"
VISITS = ("P04|2", "P04|3", "P10|2", "P10|3")
ORDER = ("native", "placebo", "population", "own")


def native_module():
    path = INNER / "worker.py"
    spec = importlib.util.spec_from_file_location("frozen_native_inner", path)
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    return run


def fitted(who, package, view, people, days):
    root = Path("/work/fit")
    data = root / f"{who}.npz"
    receipt = meta(data.with_suffix(".json"))
    assert receipt["who"] == who and receipt["file_sha"] == sha(data) and receipt["fold_view"] == view["fit_sha"]
    assert receipt["exact_feature_prefix"] and receipt["exact_native_probabilities"] and receipt["source_labels_only"]
    assert receipt["actor_proof"] == waveform_proof(who, package, people, days)
    with np.load(data, allow_pickle=False) as src:
        values = {k: src[k] for k in src.files}
    for name in ("own_rows", "population_rows", "scale_rows"):
        rows = values[name]
        assert (days[rows] == "1").all() and not np.isin(rows, view["held"]).any()
        is_owner = people[rows] == f"Dreyer2026Proteus/{who}"
        assert is_owner.all() if name == "own_rows" else not is_owner.any()
    assert values["own_rows"].size == values["population_rows"].size == sum(receipt["counts"])
    assert (values["scale"] >= .001).all()
    return values, sha(data), sha(data.with_suffix(".json"))


def call(native, template, case, n, arm, head, names, who):
    if arm in ("native_zero", "native"):
        name = "native_zero" if arm == "native_zero" else "causal"
        return native.session(template, name, case, n), None
    observed = []

    def instrument(model):
        native.instrument(model)
        observed.append(RecognizedHead(model, arm, head, names, who))
        return model

    # Bind one local callback without altering the pinned session or its global module state.
    replay = native.session
    scope = {**replay.__globals__, "instrument": instrument}
    instrumented = FunctionType(replay.__code__, scope, replay.__name__, replay.__defaults__)
    result = instrumented(template, "causal", case, n)
    assert len(observed) == 1
    return result, observed[0]


def archive(native, visit, case, n, baseline, causal):
    assert sha(AGG) == AGG_SHA
    summary = meta(AGG)
    assert not summary["held_labels_opened"] and summary["order"] == "full0" and summary["windows"] == 762
    path = INNER / f"visit{VISITS.index(visit)}"
    saved = meta(path/"native_inner.json")
    assert saved["visit"] == visit and saved["source"] == summary["producer"] and saved["no_gold"]
    assert saved["output"] == sha(path/"native_inner.npz")
    with np.load(path/"native_inner.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["rows"][:n], case["rows"][:n])
        for name, (tape, ft, stats) in (("native_zero", baseline), ("causal", causal)):
            for key, values in tape.items():
                if key != "unpatched_ea":
                    np.testing.assert_array_equal(z[f"{name}|{key}"][:n], values)
            for age, values in ft.targets.items():
                np.testing.assert_array_equal(z[f"{name}|target|{age}"], values)
                np.testing.assert_array_equal(z[f"{name}|target_rows|{age}"], case["rows"][:age])
            for age, values in ft.warm.items():
                for member, a in values.items():
                    np.testing.assert_array_equal(z[f"{name}|warm|{age}|{member}"], a)
            for k in ("recognition", "steps", "refit_ages", "warm_ages", "cp_stopped"):
                assert stats[k] == saved["arms"][name][k], (k, stats)
    np.testing.assert_array_equal(baseline[0]["labels"], case["labels"][:n])
    np.testing.assert_array_equal(baseline[0]["scores"], case["score"][:n])


def protected(baseline, challenger, visit, n):
    b, bf, bs = baseline
    a, af, ass = challenger
    assert bs == ass and bs["recognition"] == BASELINES[visit]
    assert bs["refit_ages"] == ([20] if visit == "P10|2" else [])
    assert bs["warm_ages"] == (list(range(20, 28)) if visit == "P10|2" else [])
    assert bs["steps"] == (20 if visit == "P10|2" else 0)
    for key, reference in b.items():
        if key not in ("labels", "scores"):
            np.testing.assert_array_equal(a[key], reference, err_msg=key)
    t = bs["recognition"]
    assert 0 <= t < n
    np.testing.assert_array_equal(a["scores"][:t], b["scores"][:t])
    np.testing.assert_array_equal(a["labels"][:t], b["labels"][:t])
    assert af.optimizer == bf.optimizer
    for age, labels in bf.targets.items():
        np.testing.assert_array_equal(af.targets[age], labels)
    for age, families in bf.warm.items():
        for member, values in families.items():
            np.testing.assert_array_equal(af.warm[age][member], values)
    return max(float(np.abs(a["scores"][t:]-b["scores"][t:]).max()), 0.)


def main(visit, smoke):
    assert torch.__version__ == "2.14.1+cu130" and not torch.cuda.is_initialized()
    assert "neural-interfaces-2026/.venv/" not in Path("/proc/self/maps").read_text()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    who, day = visit.split("|")
    native = native_module()
    package, view, names, weights, channels, cases = load(who)
    model = load_model(package, channels, f"prototype_replay_{who}", "cpu")
    assert Path(type(model).step.__globals__["__file__"]).resolve() == (package/"submission.py").resolve()
    np.testing.assert_array_equal(model.w, weights)
    native_ft_proof(model, who)
    with np.load(EEG, allow_pickle=False) as f:
        people, days = f["subject"], f["session"]
        rows = (people == f"Dreyer2026Proteus/{who}") & (days == "1")
        warmup = f["X"][rows][:17]
    head, fit_sha, fit_proof = fitted(who, package, view, people, days)
    model.js_bank = None
    for row in warmup:
        model.predict(torch.from_numpy(row[None]))
    model.reset_state()
    zero(model)
    case, other = cases[day], cases["2" if day == "3" else "3"]
    n = 39 if smoke else len(case["rows"])
    if smoke:
        assert visit == "P10|2" and case["rows"][0] == 1238
    full = lambda name, rows, count: call(native, model, rows, count, name, head, names, who)
    baseline, _ = full("native_zero", case, n)
    parent, _ = full("native", case, n)
    archive(native, visit, case, n, baseline, parent)
    canonical = {k: v for k, v in parent[0].items() if k in ("scores", "labels", "latch", "contrast", "identity")}
    outputs = {"rows": case["rows"][:n]}
    meta_arms = {}
    for arm in ORDER:
        trajectory, actor = (parent, None) if arm == "native" else full(arm, case, n)
        delta = protected(parent, trajectory, visit, n)
        if arm == "placebo":
            assert delta == 0
            np.testing.assert_array_equal(trajectory[0]["labels"], parent[0]["labels"])
        full(arm, other, 9)
        short, prefix_actor = full(arm, case, 29)
        native.same_prefix(trajectory, short, 29)
        changed = int(np.count_nonzero(trajectory[0]["labels"] != parent[0]["labels"]))
        status = dict(changed=changed, max_posterior_delta=delta, native=trajectory[2])
        if actor is not None:
            np.testing.assert_array_equal(np.asarray(actor.ramps[:29]), prefix_actor.ramps)
            np.testing.assert_array_equal(np.asarray(actor.inputs[:29]), prefix_actor.inputs)
            status.update(actor.receipt())
            if arm != "placebo":
                assert actor.local_effect > 1e-4 and delta > 1e-4
        meta_arms[arm] = status
        for k in canonical:
            outputs[f"{arm}|{k}"] = trajectory[0][k]
        print(f"F-C16 {visit} {arm} qualified: {status}", flush=True)
    path = Path("result.npz")
    with path.open("xb") as f:
        np.savez_compressed(f, **outputs)
    path.chmod(0o444)
    receipt = dict(visit=visit, n=n, arms=meta_arms, locked_native_parity=True, exact_prefix=True, fit=fit_sha,
                   fit_proof=fit_proof, case_fit=view["fit_sha"], no_held_gold=True, source=sha("/work/replay.sha256"),
                   conditional_source="actual §321/D-EArN causal EA with production auxiliary exposure",
                   parents=AGG_SHA, artifact=sha(path), scope="smoke" if smoke else "complete")
    out = Path("result.json")
    with out.open("x") as f:
        json.dump(receipt, f, sort_keys=True, allow_nan=False)
        f.write("\n")
    out.chmod(0o444)
    print(f"PASS F-C16 {visit} {sha(out)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("visit", choices=VISITS)
    p.add_argument("--smoke", action="store_true")
    opts = p.parse_args()
    main(opts.visit, opts.smoke)
