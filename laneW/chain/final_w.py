"""Lane W chain (widened post-mixer correction family; copy of laneP/chain/final_p.py with arm set W)
Lane P chain (copy of laneB1/chain final_b.py; arm set P): B1 chain: the final stack (hardcb) and arms around it in the corrected Friday chain (raw-slot replacement, then the real count/CSP tail).

usage: final_p.py ORDER (smoke|complete) "cbbank_ORDER_s*.npz"
Arm sets (always stock, identity, hard, hardcb as guards/controls):
  T5 / T20  lane T fitting forms, one job each: ftctl (unmodified FT stream = zero-change) and ftlite5 | ftsparse20 (banks from laneT/fit.sbatch)
  S  seed spread: hardcb with a single CBraMod seed s0/s1/s2 in the 0.2 share (instead of their mean)
  X  B6 arm X raw family (3-seed ensemble bank ensemble_X_held_new.npz, full0 only): hcX = 0.4 hard + 0.4 X + 0.2 cb (X shares the
     raw slot with the hard pair, primary), hXc = 0.8 hard + 0.2 X (X replaces CBraMod), and hcX with each single X seed
Copy of laneT/chain/final_ttt.py (hash-pinned release) with a selectable arm set; MASS 5/6 for every arm."""
import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np

import outer as outer_mod
from cross22 import doubling  # F-C21 FBL2: doubles the live FBLight member (final-stack release, hash-checked)
from bridge_orders import write_npz, write_receipt
from bridge_owner import PROTO, STACK, pin, sha, start_module, stamp
from complete_zero import FEAT, HASHES
from matrix22 import parents
from outer import full_parent, replace, within_parent
from parents22 import C20, F32, PUPIL_SOURCE

WORK = Path(__file__).resolve().parent
SET = "W"
EXTRA = {"T5": ("ftctl", "ftlite5"), "T20": ("ftctl", "ftsparse20"), "S": ("hcb_s0", "hcb_s1", "hcb_s2"),
         "X": ("hcX", "hXc", "hcX_s0", "hcX_s1", "hcX_s2"), "P": ("ftctl", "ftlite5", "hcF", "hcF_l5"), "W": ("hcF_l5",)}
ARMS = ("stock", "identity", "hard", "hardcb") + EXTRA[SET]
FT_ARMS = {"ftctl": "ctl", "ftlite5": "lite5", "ftsparse20": "sparse20"}
X_BANK = Path(os.environ.get("B1_XBANK", "/mnt/lustre/work/gehler/gfh527/ni26_laneB6_20261120/ensemble_X_held_new.npz"))
FAMS = ("fbraw", "fb", "sinc", "fbl")
BANKS = Path(os.environ["LANET_BANKS"])  # arms/ORDER_ARM/members/FAMILY_s0.npz from laneT/fit.sbatch
FULL_MASS = 5 / 6
SEG = Path(os.environ["LANEP_SEG"])  # seg_{order}_{seed}_{P}.npz from laneP/seg_export.py (member seed 0, N9); seeds 1, 2 in LANEW_SEG1
import corrw  # lane W correction family (arm specs, features, causal correct)

SEG1 = Path(os.environ["LANEW_SEG1"])  # seeds 1, 2 of the stand-in member (laneB7 export)
CORR = tuple(corrw.arms(os.environ.get("LANEW_SET", "a")))
ALL = ARMS + CORR


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
    raw, traces, mem = {}, {}, {}

    def visit(ctx, archive, patches, incumbent):
        assert patches and not raw
        stock, deltas = base(ctx, archive, patches, incumbent)
        raw["stock"] = stock
        active = np.ones(len(stock), dtype=bool)
        try:
            hc = 0.8 * pupils["hard"] + 0.2 * pupils["cb"]
            slot = {"identity": None, "hard": pupils["hard"], "hardcb": hc}
            for k in range(3):
                slot[f"hcb_s{k}"] = 0.8 * pupils["hard"] + 0.2 * pupils["cb_seeds"][k]
            if "X" in pupils:
                slot["hcX"] = 0.4 * pupils["hard"] + 0.4 * pupils["X"] + 0.2 * pupils["cb"]
                slot["hXc"] = 0.8 * pupils["hard"] + 0.2 * pupils["X"]
                for k in range(3):
                    slot[f"hcX_s{k}"] = 0.4 * pupils["hard"] + 0.4 * pupils["X_seeds"][k] + 0.2 * pupils["cb"]
            for name, values in slot.items():
                if name not in ARMS:
                    continue
                outer_mod.MASS = FULL_MASS
                raw[name], used, traces[name] = replace(base, ctx, archive, patches, incumbent, values, active)
                assert used == deltas
        finally:
            outer_mod.MASS = FULL_MASS
        # lane T: the same final-stack proposal (hardcb, MASS 5/6) with the four FT family posteriors CU17[family, 0] swapped
        saved = {f: ctx.CU17[f, 0] for f in FAMS}
        for name, arm in FT_ARMS.items():
            if name not in ARMS:
                continue
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
        # FBL2 on the final-stack proposal, with the stock FT streams (hcF) and with the lite5 FT streams (hcF_l5)
        for name, arm in (("hcF", None), ("hcF_l5", "lite5")):
            if name not in ARMS:
                continue
            if arm:
                bank = ft_bank(order, arm)
            try:
                if arm:
                    for f in FAMS:
                        ctx.CU17[f, 0] = bank[f]
                outer_mod.MASS = FULL_MASS
                if arm:
                    mem["l5"] = corrw.dump_members(ctx, archive, outer_mod, PROTO, STACK)
                with doubling(ctx, archive) as boost:
                    raw[name], used, traces[name] = replace(base, ctx, archive, patches, incumbent, 0.8 * pupils["hard"] + 0.2 * pupils["cb"], active)
                assert arm or used == deltas, name  # FT-swapped arms move the deltas by design (as ftlite5)
                assert boost["share"] == 1 / 17 and boost["warm_rows"] == 2375, (name, boost.get("share"), boost.get("warm_rows"))  # the doubled mix fired; trace mix is post-MASS so it is not compared
            finally:
                for f in FAMS:
                    ctx.CU17[f, 0] = saved[f]
                outer_mod.MASS = FULL_MASS
            assert not np.array_equal(raw[name], raw["hardcb"]), name
        mem["stock"] = corrw.dump_members(ctx, archive, outer_mod, PROTO, STACK)
        np.testing.assert_array_equal(raw["identity"], stock)
        assert traces["identity"]["slot_movement"] == 0
        for name in ARMS:
            if name in ("stock", "identity") or name in FT_ARMS:
                continue
            assert traces[name]["slot_movement"] > 0, name
            assert all(traces[name][s] != traces["identity"][s] for s in ("mix", "rebalance", "em", "tsem")), name
            assert name == "hardcb" or traces[name]["mix"] != traces["hardcb"]["mix"], name  # every arm really moved the slot
        return stock, deltas

    owner.candidate = visit
    try:
        with load() as ctx:
            assert set(raw) == set(ARMS)
            control, zero, parent = ((ctx["matched"]["stock"], ctx["zero"], ctx["proof"]) if order == "full0"
                                     else (ctx["reference"], ctx["archive_zero"], ctx["parent"]))
            np.testing.assert_array_equal(raw["stock"], control["raw"])
            yield dict(raw=raw, mem=mem, reference=control, zero=zero, parent=parent, traces=traces,
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
    pupils = dict(hard=pupils["hard"], cb=cb, cb_seeds=posts)
    if SET == "X":
        assert order == "full0", "the X bank exists for full0 only"
        with np.load(X_BANK, allow_pickle=False) as z:
            np.testing.assert_array_equal(z["index"], np.arange(2655))
            x_delivery, x = z["full0_delivery"], z["full0"]
            x_seeds = z["full0_members"]
        assert x.shape == (2655, 3) and np.isfinite(x).all() and (x >= 0).all() and x_seeds.shape == (3, 2655, 3)
        np.testing.assert_allclose(x.sum(1), 1.0, atol=2e-6, rtol=0)
        np.testing.assert_allclose(x_seeds.mean(0), x, atol=1e-9, rtol=0)
        pupils.update(X=x, X_seeds=list(x_seeds))
    with run_arms(order, pupils) as ctx:
        np.testing.assert_array_equal(ctx["delivery"], cb_delivery)
        if SET == "X":
            np.testing.assert_array_equal(ctx["delivery"], x_delivery)
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
        corr_info = corrw.run(order, ctx, rows, person, pupils, os.environ.get("LANEW_SET", "a"), SEG, SEG1)
        stages, work = {}, {}
        for arm in ALL:
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
                     arm_set=SET, corr=corr_info, delivery=stamp(ctx["delivery"]), rows=len(delivered), sessions=len(selected), parent=ctx["parent"],
                     zero=ctx["zero"], pupils=own_pins, work=work, mix=ctx["traces"], held_labels_opened=False,
                     strict_prefix=True, seconds=perf_counter() - begin)
        if not smoke:
            arrays = dict(index=np.arange(len(delivered)), delivery=delivered)
            arrays.update({f"{a}|{k}": v for a, parts in stages.items() for k, v in parts.items()})
            proof["output"] = write_npz(WORK / f"{order}_{SET}{os.environ.get('LANEW_SET', 'a')}.npz", arrays)
        path = WORK / (f"smoke_{SET}.json" if smoke else f"{order}_{SET}{os.environ.get('LANEW_SET', 'a')}.json")
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
