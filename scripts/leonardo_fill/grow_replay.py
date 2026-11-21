"""Run pinned graph parity, bank comparisons, and paired full-Solver timing."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from dose import DATA_SHA, PIN, V, live_check, replay, sessions
from grow_fixture import ARMS, HELD, REFERENCE, SCREEN, open_fold, prepare_zero, stage
from run import check_zero_mixer
from team_b_speed_bench import check_features
from check_ba_terminal import play as check_terminal_blend

TIMING_FOLDS = ("new_P01", "ret_P04")


def check(model, windows):
    live_check(model, windows)
    if model.routeB.log["err"] or model.cp.nfail:
        raise ValueError(f"failed routing or CSP stage: {model.routeB.log}, {model.cp.nfail}")
    if model.team_b is not None:
        check_features(model.team_b, windows)
    team_a = getattr(model, "team_a", None)
    if team_a is not None and (team_a.age != windows or len(team_a.adapter.features) != windows or
                               model.team_a_graph.replays != windows or model.team_a_graph.graph is None):
        raise ValueError("MANAS did not consume each window on the graph path")


def load(data, pkg, arm, receipt):
    model, secs = V.timed_load(pkg / arm, data, V.DEVICE)
    expected = ["TeamBlendStack", "HardWaveEnsemble"]
    if arm == "ba_six":
        expected.insert(0, "TerminalTeamAStack")
    if [c.__name__ for c in type(model).__mro__[:len(expected)]] != expected:
        raise ValueError(f"{arm}: unexpected Solver class")
    if (model.TEAM_B_MIX != 0.25 or len(model.team_b.nets) != sum(receipt["members"]) or
            (arm == "ba_six" and (model.TEAM_A_MIX != 0.10 or model.team_a is None))):
        raise ValueError(f"{arm}: incorrect live blend or roster")
    return model, secs


def smoke(data, x, fold, pkg, receipts):
    order = 71124
    stream = sessions(data, fold)[0]
    rows = V.D.plans(data, order, [stream])[stream]
    suffix = f"{order}_{stream.split('|')[1]}"
    pinned = {}
    for arm in ("base", REFERENCE):
        with np.load(PIN / fold / arm / f"pred_{fold}_{order}.npz", allow_pickle=False) as tape:
            np.testing.assert_array_equal(tape[f"rows_{suffix}"], rows)
            pinned[arm] = tape[f"labels_{suffix}"]
    with TemporaryDirectory() as tmp:
        target = Path(tmp) / "zero"
        prepare_zero(pkg / REFERENCE, target)
        zero, _ = V.timed_load(target, data, V.DEVICE)
        assert zero.team_b is None
        check_zero_mixer(zero)
        labels, _, _ = V.stream_session(zero, x, rows, V.DEVICE, check=check)
        np.testing.assert_array_equal(labels, pinned["base"])
    reference = None
    for arm in ARMS:
        model, _ = load(data, pkg, arm, receipts[arm])
        labels, taps, _ = V.stream_session(model, x, rows, V.DEVICE, capture=True, check=check)
        prefix, _, _ = V.stream_session(model, x, rows[:24], V.DEVICE, check=check)
        np.testing.assert_array_equal(prefix, labels[:24])
        if model.team_b.windows:
            raise ValueError("Team B template changed across sessions")
        if arm == REFERENCE:
            np.testing.assert_array_equal(labels, pinned[REFERENCE])
            reference = taps[:, :3]
        elif np.max(np.abs(reference - taps[:, :3])) <= 1e-4:
            raise ValueError(f"{arm}: the live mixer did not move")
        if arm == "ba_six":
            check_terminal_blend(model, x[rows].to(V.DEVICE), capture=True)
        print(f"{fold} {arm}: {len(rows)} live windows and reset/prefix passed", flush=True)
    print("PASS exact zero and lead labels, changed banks, graph and full terminal path", flush=True)


def timing(data, x, fold, pkg, receipts, arm, out):
    plans = [V.D.plans(data, order, sessions(data, fold)) for order in (71124, 71125)]
    rows = [next(iter(plans[0].values())), list(plans[1].values())[-1]]
    arms = (REFERENCE, arm)
    models, loads = {}, {}
    for candidate in arms:
        models[candidate], loads[candidate] = load(data, pkg, candidate, receipts[candidate])
        V.stream_session(models[candidate], x, rows[0][:72], V.DEVICE, check=check)
    pairs = []
    for r, sequence in zip(rows, (arms, arms[::-1]), strict=True):
        pair = {"windows": len(r)}
        for candidate in sequence:
            _, _, meta = V.stream_session(models[candidate], x, r, V.DEVICE, check=check)
            pair[candidate] = meta
        pairs.append(pair)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(device=torch.cuda.get_device_name(), fold=fold, arm=arm,
                                   load_s=loads, pairs=pairs), indent=1) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("smoke", "prepare", "screen", "held", "timing"))
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--fold", required=True)
    p.add_argument("--chunk", type=int)
    p.add_argument("--arm", choices=ARMS[1:])
    a = p.parse_args()
    assert torch.cuda.is_available()
    if V.digest(a.data) != DATA_SHA:
        raise ValueError("PROTEUS data digest differs")
    data = V.D.load(a.data)
    if a.phase in ("smoke", "prepare"):
        stage(a.run, a.fold, data, a.data)
        if a.phase == "prepare":
            return
    if a.phase == "held":
        decision = json.loads((a.run / "selection.json").read_text())
        if decision["chosen"] is None:
            print("No candidate met the screen; no held-out scoring")
            return
    pkg, receipts = open_fold(a.run, a.fold)
    signals = torch.from_numpy(data["X"])
    if a.phase == "smoke":
        smoke(data, signals, a.fold, pkg, receipts)
    elif a.phase == "timing":
        assert a.fold in TIMING_FOLDS and a.arm in ARMS[1:]
        timing(data, signals, a.fold, pkg, receipts, a.arm, a.run / "timing" / f"{a.fold}-{a.arm}.json")
    else:
        arms, order_bank = (ARMS, SCREEN) if a.phase == "screen" else ((REFERENCE, decision["chosen"]), HELD)
        orders = order_bank[2 * a.chunk:2 * a.chunk + 2]
        assert len(orders) == 2
        models = ((arm, load(data, pkg, arm, receipts[arm])[0], receipts[arm]) for arm in arms)
        replay(data, signals, a.fold, models, orders, a.run / a.phase, check=check)


if __name__ == "__main__":
    main()
