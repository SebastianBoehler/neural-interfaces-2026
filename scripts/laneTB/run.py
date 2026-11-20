"""Stream one real fold through the packaged live mixer, or verify the pinned P01 smoke."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "cleanslate/harness"), str(ROOT / "submissions/final_stack/team_b")]
import data as D
from stream import load_solver
from build import ARMS, bank, fold, safe_source, sha

DEVICE = "cuda:0"


def timed_load(pkg, data, device):
    start = time.perf_counter()
    model = load_solver(pkg, data["ch_names"], device)
    sync(device)
    return model, time.perf_counter() - start


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def stream_session(template, signals, rows, device, capture=False):
    _, official = D.official()
    sync(device)
    start = time.perf_counter()
    model = official.fresh_model(template)
    labels, taps = [], []
    for i, row in enumerate(rows):
        result = model.predict(signals[row:row + 1].to(device))
        if result.dtype != torch.int64 or result.shape != (1,) or not 0 <= int(result[0]) <= 2:
            raise ValueError("invalid official Solver label")
        if model.dead or model.log6["err"]:
            raise ValueError(model.log6)
        labels.append(int(result[0]))
        if capture:
            taps.append(np.r_[model.trace["v6"]["mix6"], model.trace["out"]])
        if i == 0:
            sync(device)
            warm_start = time.perf_counter()
    sync(device)
    secs = time.perf_counter() - warm_start
    meta = dict(setup_s=warm_start - start, warm_s=secs, recognition=model.log6["rec"])
    if model.log6["v6"] != len(rows):
        raise ValueError(f"live path missed windows: {model.log6}")
    return np.array(labels, np.int8), np.array(taps), meta


def stage_packages(source, work, trained):
    out = {}
    blobs = {3: bank(trained, 3), 1: bank(trained, 1)}
    for arm, (weight, members) in ARMS.items():
        pkg = work / arm / "pkg"
        fold(source, pkg, weight, members, blobs[members] if weight else b"")
        out[arm] = pkg
    return out


def verify_encoder(enc, original, pack_path, data, signals, rows, members, device):
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("team_b_oracle", original)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    pack = torch.load(pack_path, map_location="cpu", weights_only=True)
    pack["nets"] = [pack["nets"][i + j] for i in (0, 3, 6) for j in range(members)]
    oracle = mod.Model(pack, device)
    enc.reset()
    start = time.perf_counter()
    for r in rows:
        oracle.predict(signals[r:r + 1].to(device))
        q = enc.posterior(signals[r].to(device))
        expected = oracle.last[0][-1]
        torch.testing.assert_close(q, expected, rtol=2e-6, atol=2e-7)
        assert int(q.argmax()) == oracle.arms["enc"][-1]
    enc.reset()
    print(f"encoder oracle {3 * members} members, {len(rows)} windows passed in {time.perf_counter() - start:.2f}s",
          flush=True)


def time_encoder(enc, signals, rows, device):
    enc.reset()
    enc.posterior(signals[rows[0]].to(device)).cpu()
    sync(device)
    start = time.perf_counter()
    for r in rows[1:]:
        enc.posterior(signals[r].to(device)).cpu().numpy()
    sync(device)
    msec = 1000 * (time.perf_counter() - start) / (len(rows) - 1)
    enc.reset()
    return msec


def check_zero_mixer(model):
    """Check exact identity against the unmodified mixer in the same call."""
    cls = type(model)
    parent = cls.__mro__[1]
    original_parent, original_zero = parent.mix6, cls.mix6

    def reference(self, *args):
        mix = original_parent(self, *args)
        self._zero_reference = mix.copy()
        return mix

    def verify(self, *args):
        mix = original_zero(self, *args)
        if not np.array_equal(mix, self._zero_reference):
            raise ValueError("blend zero changed the mixer output")
        return mix

    parent.mix6, cls.mix6 = reference, verify


def smoke(args, data, signals, pkgs):
    plan = D.plans(data, 71000, ["P01|1"])
    rows = plan["P01|1"]
    with np.load(args.source.parent.parent / "pred_new_P01.npz", allow_pickle=False) as stored:
        expected = stored["labels_71000_1"]
        np.testing.assert_array_equal(rows, stored["rows_71000_1"])
    safe = args.safe
    safe_text = (safe / "submission.py").read_text()
    assert safe_source((args.source / "submission.py").read_text()) == safe_text
    models = {}
    timing = dict(device=torch.cuda.get_device_name(), windows=len(rows), timings={})
    for arm, path in {"stock": args.stock, "safe": safe, **pkgs}.items():
        models[arm], load_s = timed_load(path, data, DEVICE)
        timing["timings"][arm] = dict(load_s=load_s, runs=[])
    check_zero_mixer(models["base"])
    for arm in ("nine15", "three15"):
        model = models[arm]
        verify_encoder(model.team_b, args.weights.with_name("submission.py"), args.weights,
                       data, signals, rows[:68], model.TEAM_B_MEMBERS, DEVICE)
        timing["timings"][arm]["encoder_ms"] = [time_encoder(model.team_b, signals, rows, DEVICE)
                                               for _ in range(2)]
    guarded, repeated = {}, {}
    counterbalanced = [arm for arm in reversed(models) if arm not in {"nine25", "three25"}]
    for round_index, order in enumerate([list(models), counterbalanced]):
        for arm in order:
            labels, taps, meta = stream_session(models[arm], signals, rows, DEVICE, capture=True)
            timing["timings"][arm]["runs"].append(meta)
            if arm in {"safe", "base"}:
                np.testing.assert_array_equal(labels, expected)
            (guarded if round_index == 0 else repeated)[arm] = (labels, taps)
    assert models["base"].team_b is None and models["base"].TEAM_B_MIX == 0
    # GPU fine-tuning can introduce small run-to-run posterior drift.
    drift = {}
    for before, after, key in [
        (guarded["safe"], repeated["safe"], "safe_repeat"),
        (guarded["base"], repeated["base"], "zero_repeat"),
        (guarded["safe"], guarded["base"], "safe_vs_zero"),
    ]:
        np.testing.assert_array_equal(before[0], after[0])
        drift[key] = float(np.max(np.abs(before[1] - after[1])))
        print(f"Maximum absolute GPU posterior difference {key}: {drift[key]:.3g}", flush=True)
        np.testing.assert_allclose(before[1], after[1], rtol=0, atol=2e-6)
    timing["zero_posterior_drift"] = drift
    for arm in ARMS.keys() - {"base"}:
        if np.array_equal(guarded[arm][1][:, :3], guarded["safe"][1][:, :3]):
            raise ValueError(f"blend {arm} did not reach the live mixer")
    (args.out / "smoke.json").write_text(json.dumps(timing, indent=2) + "\n")
    print(f"PASS pinned order and blend-zero equality on {len(rows)} windows; live blend reached all positive arms",
          flush=True)


def scored(args, data, signals, pkgs):
    sessions = sorted(set(data["session"][data["person"] == args.held]))
    returning = args.mode == "ret"
    people = D.PEOPLE if returning else [p for p in D.PEOPLE if p != args.held]
    pack = torch.load(args.weights, map_location="cpu", weights_only=True)
    assert pack["people"] == people and pack["first_only"] == ([args.held] if returning else [])
    if returning:
        assert args.held in D.RETURNING and set(sessions) == {"1", "2", "3"}
        sessions = ["2", "3"]
    orders = [int(s) for s in args.orders.split(",")]
    for arm, path in pkgs.items():
        template, _ = timed_load(path, data, DEVICE)
        res = {"source_sha": sha((args.source / "submission.py").read_bytes()),
               "weights_sha": sha(args.weights.read_bytes())}
        for order in orders:
            for stream, rows in D.plans(data, order, [f"{args.held}|{s}" for s in sessions]).items():
                labels, _, meta = stream_session(template, signals, rows, DEVICE)
                suffix = f"{order}_{stream.split('|')[1]}"
                res.update({f"rows_{suffix}": rows, f"labels_{suffix}": labels})
                print(arm, args.mode, args.held, suffix, len(rows), meta, flush=True)
        np.savez_compressed(args.out / arm / f"pred_{args.mode}_{args.held}.npz", **res)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "weights", "data", "out", "stock", "safe"):
        p.add_argument("--" + name, type=Path, required=name not in {"stock", "safe"})
    p.add_argument("--mode", choices=("new", "ret", "smoke"), required=True)
    p.add_argument("--held", required=True)
    p.add_argument("--orders", default="71120,71121,71122,71123")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Leonardo A100 required")
    data = D.load(args.data)
    signals = torch.from_numpy(data["X"])
    pkgs = stage_packages(args.source, args.out, args.weights)
    if args.mode == "smoke":
        smoke(args, data, signals, pkgs)
    else:
        scored(args, data, signals, pkgs)


if __name__ == "__main__":
    main()
