"""Lane K test 2 (segment-dispersion correction after RouteB). Copy of final22.py with proposal arms built from B5 banks.
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
NEW = ("whole_g", "disp_g", "whole_d", "disp_d")
ARMS = ("stock", "identity", "hardcb") + NEW
FULL_MASS = 5 / 6
SEG = Path("/mnt/lustre/work/gehler/gfh527/ni26_laneB5_20261120/laneK/seg")  # seg_{seed}_{P}.npz, member seed 0, N9
SEED = 0


def seg_rows(person):
    """(2655 x 4 x 3) segment contributions in FEAT row order, and the delivery they were streamed in."""
    out = np.zeros((len(person), 4, 3)); dl = []
    for p in sorted(set(person)):
        z = np.load(SEG / f"seg_{SEED}_{p}.npz")
        out[z["rows"]] = z["seg"]; dl.append(z["delivery"])
    return out


def correct_arms(raw, delivery, person, seg):
    """Apply segcorr to raw['hardcb'] per person in delivery order; pseudo-labels = frozen stock raw argmax; donor = next person's stream (same age)."""
    import segcorr
    labels = raw["stock"].argmax(1)
    people = sorted(set(person))
    res, info = {k: raw["hardcb"].copy() for k in NEW}, {}
    rows = {p: delivery[person[delivery] == p] for p in people}
    for i, p in enumerate(people):
        q = people[(i + 1) % len(people)]
        for kind in ("whole", "disp"):
            xo = segcorr.features(seg[rows[p]], kind)
            xq = segcorr.features(seg[rows[q]], kind)
            m = min(len(xo), len(xq))
            xd = np.concatenate([xq[:m], xo[m:]])  # donor history at the same age; own features beyond the donor's length are not used as history
            for tag in ("g", "d"):
                hist, lab = (None, labels[rows[p]]) if tag == "g" else (xd, np.concatenate([labels[rows[q]][:m], labels[rows[p]][m:]]))
                o, act = segcorr.correct(raw["hardcb"][rows[p]], lab, xo, hist)
                res[f"{kind}_{tag}"][rows[p]] = o
                info[f"{kind}_{tag}|{p}"] = dict(active=int(act.sum()), changed=int((o.argmax(1) != raw["hardcb"][rows[p]].argmax(1)).sum()))
    return res, info


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
            for name, values in (("identity", None), ("hardcb", 0.8 * pupils["hard"] + 0.2 * pupils["cb"])):
                outer_mod.MASS = FULL_MASS
                raw[name], used, traces[name] = replace(base, ctx, archive, patches, incumbent, values, active)
                assert used == deltas
            corr, info = correct_arms(raw, pupils["delivery"], pupils["person"], pupils["seg"])
            for k, v in corr.items():
                raw[k], traces[k] = v, traces["hardcb"]
            raw["_info"] = info
        finally:
            outer_mod.MASS = FULL_MASS
        np.testing.assert_array_equal(raw["identity"], stock)
        assert traces["identity"]["slot_movement"] == 0
        assert traces["hardcb"]["slot_movement"] > 0
        assert all(raw[k].argmax(1).tolist() != raw["hardcb"].argmax(1).tolist() for k in NEW), "a correction arm equals hardcb"
        return stock, deltas

    owner.candidate = visit
    try:
        with load() as ctx:
            assert set(raw) == set(ARMS) | {"_info"}
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
    pupils["person"], pupils["seg"], pupils["delivery"] = person, seg_rows(person), cb_delivery
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
        for key in stages["stock"]:
            np.testing.assert_array_equal(stages["stock"][key][delivered], ctx["reference"][key][delivered])
        info = ctx["raw"].pop("_info")
        proof = dict(passed=True, scope="smoke" if smoke else "complete", order=order, source=source, cb_pupil=[sha(f) for f in seeds], seg=[sha(SEG / f"seg_{SEED}_{p}.npz") for p in sorted(set(person))], corr=info,
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
