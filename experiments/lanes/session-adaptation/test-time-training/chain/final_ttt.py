"""Final-stack arms (stock, hard, cb, hardcb) in the corrected Friday chain: raw-slot replacement, then the real count/CSP tail.

usage: final22.py ORDER (smoke|complete) "cbbank_ORDER_s*.npz"   (ORDER full0 or within0; the three cb_bank.py seed streams, mean = the pupil)
Runs under the c25d worker environment (PYTHONPATH as in worker.sbatch); a copy of matrix22/cross22 with the FBL2/joint arms
removed, the CBraMod pupil added, and MASS set per arm: hard 5/6, cb 1/6, hardcb 5/6 with proposal 0.8*hard + 0.2*cb."""
import argparse
import json
import os
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np

import outer as outer_mod
from bridge_orders import write_npz, write_receipt
from bridge_owner import pin, sha, start_module, stamp
from complete_zero import FEAT, HASHES
from matrix22 import parents
from outer import full_parent, replace, within_parent
from parents22 import C20, F32, PUPIL_SOURCE

WORK = Path(__file__).resolve().parent
ARMS = ("stock", "identity", "hard", "cb", "hardcb", "ftctl", "ftlight", "ftheavy")
FT_ARMS = {"ftctl": "ctl", "ftlight": "light", "ftheavy": "heavy"}
FAMS = ("fbraw", "fb", "sinc", "fbl")
BANKS = Path(os.environ["LANET_BANKS"])  # arms/ORDER_ARM/members/FAMILY_s0.npz from laneT/fit.sbatch
MASS = {"hard": 5 / 6, "cb": 1 / 6, "hardcb": 5 / 6}
FULL_MASS = 5 / 6


def ft_bank(order, arm):
    out = {}
    for fam in FAMS:
        with np.load(BANKS / f"{order}_{arm}" / "members" / f"{fam}_s0.npz", allow_pickle=False) as z:
            assert np.array_equal(z["index"], np.arange(2655)), (arm, fam)
            out[fam] = z["causal"].astype(np.float64)
        assert np.isfinite(out[fam]).all() and np.allclose(out[fam].sum(1), 1, atol=2e-6)
    return out


def helper(name, path):
    return start_module(name, path, path.read_text())


@contextmanager
def run_arms(order, pupils):
    owner, load = (full_parent, full_parent.full0) if order == "full0" else (within_parent, within_parent.rebuilt)
    base = owner.candidate
    raw, traces = {}, {}

    def visit(ctx, archive, patches, incumbent):
        assert patches and not raw
        stock, deltas = base(ctx, archive, patches, incumbent)
        raw["stock"] = stock
        active = np.ones(len(stock), dtype=bool)
        try:
            for name, values in (("identity", None), ("hard", pupils["hard"]), ("cb", pupils["cb"]),
                                 ("hardcb", 0.8 * pupils["hard"] + 0.2 * pupils["cb"])):
                outer_mod.MASS = MASS.get(name, FULL_MASS)
                raw[name], used, traces[name] = replace(base, ctx, archive, patches, incumbent, values, active)
                assert used == deltas
        finally:
            outer_mod.MASS = FULL_MASS
        # lane T: the same final-stack proposal (hardcb, MASS 5/6) with the four FT family posteriors CU17[family, 0] swapped
        saved = {f: ctx.CU17[f, 0] for f in FAMS}
        for name, arm in FT_ARMS.items():
            bank = ft_bank(order, arm)
            if arm == "ctl":  # zero-change: the unmodified stream must give the released member banks
                traces["ctl_vs_release"] = {f: float(np.abs(bank[f] - saved[f]).max()) for f in FAMS}
            try:
                for f in FAMS:
                    ctx.CU17[f, 0] = bank[f]
                outer_mod.MASS = FULL_MASS
                raw[name], _, traces[name] = replace(base, ctx, archive, patches, incumbent, 0.8 * pupils["hard"] + 0.2 * pupils["cb"], active)
            finally:
                for f in FAMS:
                    ctx.CU17[f, 0] = saved[f]
                outer_mod.MASS = FULL_MASS
            if arm == "ctl":
                traces["ctl_final_bitwise"] = bool(np.array_equal(raw[name], raw["hardcb"]))
                traces["ctl_final_maxdiff"] = float(np.abs(raw[name] - raw["hardcb"]).max())
            else:
                assert not np.array_equal(raw[name], raw["hardcb"]), arm
        np.testing.assert_array_equal(raw["identity"], stock)
        assert traces["identity"]["slot_movement"] == 0
        for name in ("hard", "cb", "hardcb"):
            assert traces[name]["slot_movement"] > 0, name
            assert all(traces[name][s] != traces["identity"][s] for s in ("mix", "rebalance", "em", "tsem")), name
        assert traces["hardcb"]["mix"] != traces["hard"]["mix"] != traces["cb"]["mix"]
        return stock, deltas

    owner.candidate = visit
    try:
        with load() as ctx:
            assert set(raw) == set(ARMS)
            control, zero, parent = ((ctx["matched"]["stock"], ctx["zero"], ctx["proof"]) if order == "full0"
                                     else (ctx["reference"], ctx["archive_zero"], ctx["parent"]))
            np.testing.assert_array_equal(raw["stock"], control["raw"])
            yield dict(raw=raw, reference=control, zero=zero, parent=parent, traces=traces,
                       **{k: ctx[k] for k in ("count", "final", "windows", "delivery")})
    finally:
        owner.candidate = base


def run(order, smoke, cb_path):
    begin = perf_counter()
    source = sha(WORK / "source.sha256")
    pin(FEAT, HASHES[FEAT])
    with np.load(FEAT, allow_pickle=False) as z:
        person = np.asarray([p.rsplit("/", 1)[-1] for p in z["subject"]])
        visits = z["session"]
    assert len(set(person)) == 10 and len(person) == 2655
    _, receipts = parents(order)
    pupils, own_pins = helper("c25_f32_inputs", F32 / "consume32.py").inputs(order, person, PUPIL_SOURCE)
    run_tail = helper("c22_parent_tail", C20 / "worker.py").run_tail
    seeds = sorted(Path(cb_path).parent.glob(Path(cb_path).name))  # e.g. cbbank_full0_s*.npz: the three seed streams
    assert len(seeds) == 3, f"expected three CBraMod seed banks, got {seeds}"
    posts = []
    for f in seeds:
        with np.load(f, allow_pickle=False) as z:
            assert str(z["order"]) == order
            posts.append(z["post"])
            cb_delivery = z["delivery"]
    cb = np.mean(posts, axis=0)
    assert cb.shape == (2655, 3) and np.isfinite(cb).all() and (cb >= 0).all()
    np.testing.assert_allclose(cb.sum(1), 1.0, atol=1e-6, rtol=0)
    pupils = dict(hard=pupils["hard"], cb=cb)
    with run_arms(order, pupils) as ctx:
        np.testing.assert_array_equal(ctx["delivery"], cb_delivery)
        assert ctx["parent"] == receipts["raw"]["parent"] == receipts["fbl"]["anchor"]
        assert ctx["zero"] == receipts["raw"]["archive_zero"] == receipts["fbl"]["zero"]
        count = ctx["count"]
        sessions = tuple(count.SESS)
        assert len(sessions) == 14
        rows = {s: count.C.order_rows(order, s, count.PROTO) for s in sessions}
        p01 = next(s for s, r in rows.items() if set(person[r]) == {"P01"} and set(visits[r]) == {"1"})
        latch = count.latch(order)
        peer = next(s for s in sessions if s != p01 and latch[s] is not None and latch[s] < 120)
        prefix = (p01, peer, p01)
        selected = (p01, peer) if smoke else sessions
        delivered = np.concatenate([rows[s] for s in selected])
        if not smoke:
            np.testing.assert_array_equal(delivered, ctx["delivery"])
        stages, work = {}, {}
        for arm in ARMS:
            arrays, work[arm] = run_tail(ctx, order, ctx["raw"][arm], selected, prefix)
            stages[arm] = arrays
            assert work[arm]["count_effect"] > 0 and work[arm]["csp_refits"] > 0
            if not smoke:
                assert (work[arm]["csp_refits"], work[arm]["csp_fusions"]) == (126, 1395)
            print("PASS final-stack count/CSP", order, arm, work[arm]["csp_refits"], flush=True)
        with np.load(F32 / f"{order}.npz", allow_pickle=False) as hard_bank:
            for key in ("raw", "filtered", "terminal"):
                stock = stages["stock"][key][delivered]
                np.testing.assert_array_equal(stages["identity"][key][delivered], stock)
                np.testing.assert_array_equal(hard_bank[f"stock|{key}"][delivered], stock)
                np.testing.assert_array_equal(hard_bank[f"hard|{key}"][delivered], stages["hard"][key][delivered])
        for key in stages["stock"]:
            np.testing.assert_array_equal(stages["stock"][key][delivered], ctx["reference"][key][delivered])
        proof = dict(passed=True, scope="smoke" if smoke else "complete", order=order, source=source, cb_pupil=[sha(f) for f in seeds],
                     delivery=stamp(ctx["delivery"]), rows=len(delivered), sessions=len(selected), parent=ctx["parent"],
                     zero=ctx["zero"], pupils=own_pins, work=work, mix=ctx["traces"], held_labels_opened=False,
                     strict_prefix=True, seconds=perf_counter() - begin)
        if not smoke:
            arrays = dict(index=np.arange(len(delivered)), delivery=delivered)
            arrays.update({f"{a}|{k}": v for a, parts in stages.items() for k, v in parts.items()})
            proof["output"] = write_npz(WORK / f"{order}.npz", arrays)
        path = WORK / ("smoke.json" if smoke else f"{order}.json")
        write_receipt(path, **proof)
    print("PASS final-stack", order, proof["scope"], sha(path), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("order", choices=("full0", "within0"))
    p.add_argument("mode", choices=("smoke", "complete"))
    p.add_argument("cb_pupil", type=Path)
    a = p.parse_args()
    assert a.mode != "smoke" or a.order == "full0"
    run(a.order, a.mode == "smoke", a.cb_pupil)
