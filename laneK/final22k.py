"""Lane K test 3 (diverse source subsets per seed). Copy of final22.py with proposal arms built from B5 banks.
Original docstring: Final-stack arms (stock, hard, cb, hardcb) in the corrected Friday chain: raw-slot replacement, then the real count/CSP tail.

usage: final22.py ORDER (smoke|complete) "cbbank_ORDER_s*.npz"   (ORDER full0 or within0; the three cb_bank.py seed streams, mean = the pupil)
Runs under the c25d worker environment (PYTHONPATH as in worker.sbatch); a copy of matrix22/cross22 with the FBL2/joint arms
removed, the CBraMod pupil added, and MASS set per arm: hard 5/6, cb 1/6, hardcb 5/6 with proposal 0.8*hard + 0.2*cb."""
import argparse
import json
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
B5 = Path("/mnt/lustre/work/gehler/gfh527/ni26_laneB5_20261120/banks")
NEW = ("n9", "n5s", "n5d", "n9cb", "n5scb", "n5dcb", "n5dTcb")
ARMS = ("stock", "identity", "hard", "hardcb") + NEW
FULL_MASS = 5 / 6   # every proposal arm replaces 5/6 of the raw slot (B5 swap form)


def entropy(p):
    return float(-(p * np.log(np.clip(p, 1e-12, 1))).sum(1).mean())


def temper(p, target):
    """Power-scale posteriors p**t (renormalised) so the mean entropy equals target (bisection on t)."""
    lo, hi = 0.2, 5.0
    for _ in range(60):
        t = (lo + hi) / 2
        q = p ** t; q /= q.sum(1, keepdims=True)
        lo, hi = (lo, t) if entropy(q) < target else (t, hi)  # entropy falls as t grows
    assert abs(entropy(q) - target) < 1e-3, (entropy(q), target)
    return q


def b5_pupils(delivery):
    """n9 = N9 3-seed ensemble; n5s = N5 subset 0 (shared subset, 3 seeds); n5d = seed s trained on subset s (s=0,1,2), mean."""
    z = {k: np.load(B5 / f"bank_{k}.npz") for k in ("N9s0", "N5s0", "N5s1", "N5s2")}
    for v in z.values():
        np.testing.assert_array_equal(v["full0_delivery"], delivery)
    n5d = np.mean([z[f"N5s{s}"]["full0_members"][s] for s in range(3)], axis=0)
    n9, n5s = z["N9s0"]["full0"], z["N5s0"]["full0"]
    assert not np.allclose(n5d, n5s) and np.isfinite(n5d).all()
    return dict(n9=n9, n5s=n5s, n5d=n5d, n5dT=temper(n5d, entropy(n9)))


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
            hc = lambda x: 0.8 * x + 0.2 * pupils["cb"]
            plan = (("identity", None), ("hard", pupils["hard"]), ("hardcb", hc(pupils["hard"])),
                    ("n9", pupils["n9"]), ("n5s", pupils["n5s"]), ("n5d", pupils["n5d"]),
                    ("n9cb", hc(pupils["n9"])), ("n5scb", hc(pupils["n5s"])), ("n5dcb", hc(pupils["n5d"])),
                    ("n5dTcb", hc(pupils["n5dT"])))
            for name, values in plan:
                outer_mod.MASS = FULL_MASS
                raw[name], used, traces[name] = replace(base, ctx, archive, patches, incumbent, values, active)
                assert used == deltas
        finally:
            outer_mod.MASS = FULL_MASS
        np.testing.assert_array_equal(raw["identity"], stock)
        assert traces["identity"]["slot_movement"] == 0
        for name in ARMS[2:]:
            assert traces[name]["slot_movement"] > 0, name
            assert all(traces[name][s] != traces["identity"][s] for s in ("mix", "rebalance", "em", "tsem")), name
        assert len({str(traces[n]["mix"]) for n in ARMS[2:]}) == len(ARMS) - 2  # every arm acts differently
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
    pupils.update(b5_pupils(cb_delivery))
    print("mean entropy", {k: round(entropy(v), 4) for k, v in pupils.items()}, flush=True)
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
        proof = dict(passed=True, scope="smoke" if smoke else "complete", order=order, source=source, cb_pupil=[sha(f) for f in seeds], b5=[sha(B5 / f"bank_{k}.npz") for k in ("N9s0", "N5s0", "N5s1", "N5s2")],
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
    assert a.order == "full0"  # B5 banks are full0 only
    run(a.order, a.mode == "smoke", a.cb_pupil)
