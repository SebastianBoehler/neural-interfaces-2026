"""Score every terminal arm from the same live Solver and fresh session."""
import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "scripts/laneTB"), str(ROOT / "submissions/final_stack")]
from fit import DATA_SHA, D
from fixture import HELD, ORDER_PIN, SCREEN, TEAM_B_PIN, stage, visits
from patch_terminal import MIXES
from team_b_speed_bench import check_features
from verify import digest, timed_load

DEVICE = "cuda:0"


def stream(template, signals, rows, arms, *, observe, captures=False):
    _, official = D.official()
    model = official.fresh_model(template)
    labels = {a: [] for a in ("lead", *arms)}
    taps = []
    for row in rows:
        x = signals[row:row + 1].to(DEVICE)
        p = model.predict(x)
        if p.shape != (1,) or p.dtype != torch.long or p.item() not in (0, 1, 2):
            raise ValueError("invalid official label")
        out = model.terminal_labels if observe else {"lead": p.item()}
        if set(out) != labels.keys() or out[(arms or ("lead",))[-1]] != p.item():
            raise ValueError("invalid terminal decision trace")
        for a in labels:
            labels[a].append(out[a])
        if model.dead or model.log6["err"]:
            raise ValueError(f"live Solver degraded: {model.log6}")
        if captures:
            taps.append(model.terminal_native.copy())
    n = len(rows)
    if (model.log6["v6"] != n or not model.routed or model.routeB.log["err"] or
            model.cf.dead or model.cp.dead or model.cf.log["err"] or model.cp.log["err"]):
        raise ValueError(f"route, count or CSP failed: {model.log6}, {model.cf.log}, {model.cp.log}")
    if len(model.team_b.windows) != n or template.team_b.windows:
        raise ValueError("Team B did not process every row with session-local state")
    check_features(model.team_b, n)
    spectral = getattr(model, "spectral", None)
    if spectral is not None and (spectral.n != n or template.spectral.n):
        raise ValueError("spectral geometry did not process every row with session-local state")
    cb = getattr(model, "terminal_cb", None)
    if cb is not None and (cb.n != n or cb._forward_graph.replays != n or
                           cb._forward_graph.graph is None or template.terminal_cb.n):
        raise ValueError("CBraMod did not process every row with a session-local graph")
    return {a: np.asarray(actual, dtype=np.int8) for a, actual in labels.items()}, np.asarray(taps)


def load(run, fold, data, folder, arms, observe=True):
    receipt = stage(run, data, fold, folder, arms, observe=observe)
    model, _ = timed_load(folder, data, DEVICE)
    if type(model).__name__ != ("BreakthroughStack" if observe else "TeamBlendStack"):
        raise ValueError("unexpected score model")
    if len(model.team_b.nets) != 9:
        raise ValueError("unexpected Team B count")
    for net in model.team_b.nets[:6]:
        if getattr(net.body, "fused", None) is None:
            raise ValueError("Team B fused convolution is not active")
    return model, receipt


def smoke(run, fold, data, signals, tmp):
    s = visits(data, fold)[0]
    rows = D.plans(data, ORDER_PIN, [s])[s]
    native, _ = load(run, fold, data, tmp / "native", (), observe=False)
    ref, _ = stream(native, signals, rows, (), observe=False)
    with np.load(TEAM_B_PIN / fold / "nine25" / f"pred_{fold}_{ORDER_PIN}.npz", allow_pickle=False) as pin:
        key = f"{ORDER_PIN}_{s.split('|')[1]}"
        np.testing.assert_array_equal(pin[f"rows_{key}"], rows)
        np.testing.assert_array_equal(pin[f"labels_{key}"], ref["lead"])
    zero, _ = load(run, fold, data, tmp / "zero", ())
    z, native_scores = stream(zero, signals, rows, (), observe=True, captures=True)
    np.testing.assert_array_equal(z["lead"], ref["lead"])
    positive, _ = load(run, fold, data, tmp / "all", tuple(MIXES))
    out, traces = stream(positive, signals, rows, tuple(MIXES), observe=True, captures=True)
    np.testing.assert_array_equal(out["lead"], z["lead"])
    np.testing.assert_allclose(traces, native_scores, atol=2e-5, rtol=0)
    limit = 64
    for a in MIXES:
        if not np.any(out[a] != out["lead"]):
            raise ValueError(f"{a} does not affect any smoke label")
        single, _ = load(run, fold, data, tmp / a, (a,))
        picked, _ = stream(single, signals, rows[:limit], (a,), observe=True)
        for name in picked:
            np.testing.assert_array_equal(picked[name], out[name][:limit])
    print(f"PASS {fold} pinned fused lead and zero, all {len(MIXES)} active arms, "
          f"paired full {len(rows)} and fresh prefix {limit}", flush=True)


def scored(run, fold, data, signals, tmp, block, chunk):
    if block == "screen":
        arms, orders = tuple(MIXES), SCREEN
    else:
        decision = json.loads((run / "screen.json").read_text())
        if decision["selected"] is None:
            print("No candidate met the frozen screen. No held-out scores.")
            return
        arms, orders = (decision["selected"],), HELD
    selected_orders = orders[2 * chunk:2 * chunk + 2]
    if len(selected_orders) != 2:
        raise ValueError("invalid order chunk")
    model, receipt = load(run, fold, data, tmp / "fold", arms)
    payload = {"receipt": np.asarray(json.dumps(receipt, sort_keys=True))}
    for o in selected_orders:
        for s, rows in D.plans(data, o, visits(data, fold)).items():
            key = f"{o}_{s.split('|')[1]}"
            out, _ = stream(model, signals, rows, arms, observe=True)
            payload[f"rows_{key}"] = rows
            payload.update({f"{a}_{key}": labels for a, labels in out.items()})
            print(f"PASS {fold} {o} {s} {len(rows)}", flush=True)
    target = run / block / fold
    target.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target) as scratch:
        name = f"{selected_orders[0]}.npz"
        path = Path(scratch) / name
        np.savez_compressed(path, **payload)
        path.rename(target / name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("block", choices=("smoke", "screen", "held"))
    p.add_argument("run", type=Path)
    p.add_argument("data", type=Path)
    p.add_argument("fold")
    p.add_argument("chunk", nargs="?", type=int)
    a = p.parse_args()
    if not torch.cuda.is_available() or digest(a.data) != DATA_SHA:
        raise ValueError("CUDA and the exact PROTEUS export are required")
    data = D.load(a.data)
    signals = torch.from_numpy(data["X"])
    with TemporaryDirectory() as scratch:
        tmp = Path(scratch)
        if a.block == "smoke":
            smoke(a.run, a.fold, data, signals, tmp)
        else:
            scored(a.run, a.fold, data, signals, tmp, a.block, a.chunk)


if __name__ == "__main__":
    main()
