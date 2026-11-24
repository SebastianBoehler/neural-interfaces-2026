"""Lane B7 chain (copy of laneP/chain/final_p.py): seed robustness of the lane P arms hardcb / ftlite5 / hcF_l5 / ds_l5 / ds_all.
usage: final_b7.py ORDER (smoke|complete) "cbbank_ORDER_s*.npz" SETNAME
A SET (table SETS) picks one alternative seed per factor: hard pair member (None = pair mean, as lane P), CBraMod seed in the 0.2 share (None = three-seed mean),
lite5 fit seed (lite5 = seed 0, lite5f1, lite5f2) and the dispersion member (seg export seed). Set B0 = all as lane P and must reproduce laneP full0_P/within0_P terminal bitwise.
Everything else (chain, constants, MASS 5/6, correction rule) is the lane P code unchanged."""
import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np

import outer as outer_mod
from cross22 import doubling  # F-C21 FBL2: doubles the live FBLight member (final-stack release, hash-checked)
from bridge_orders import write_npz, write_receipt
from bridge_owner import pin, sha, start_module, stamp
from complete_zero import FEAT, HASHES
from matrix22 import parents
from outer import full_parent, replace, within_parent
from parents22 import C20, F32, PUPIL_SOURCE

C20_PUPILS = F32.parent / "fc20_pupils_v2"  # as locked32.C20_PUPILS (CAMP / fc20_pupils_v2)

WORK = Path(__file__).resolve().parent
SETS = {  # name: (hard seed or None, cb seed or None, lite5 bank name, seg seed)
    "B0": (None, None, "lite5", 0),
    "cb0": (None, 0, "lite5", 0), "cb1": (None, 1, "lite5", 0), "cb2": (None, 2, "lite5", 0),
    "h0": (0, None, "lite5", 0), "h1": (1, None, "lite5", 0),
    "l5f1": (None, None, "lite5f1", 0), "l5f2": (None, None, "lite5f2", 0),
    "seg1": (None, None, "lite5", 1), "seg2": (None, None, "lite5", 2),
    "J0": (0, 0, "lite5", 0), "J1": (1, 1, "lite5f1", 1), "J2": (0, 2, "lite5f2", 2),
}
SETNAME, HARD, CBK, L5, SEGSEED = "B0", None, None, "lite5", 0
REF = os.environ.get("LANEB7_REF")  # laneP chain dir with full0_P.npz / within0_P.npz (zero-change check of B0)
ARMS = ("stock", "identity", "hard", "hardcb", "ftlite5", "hcF_l5")
FAMS = ("fbraw", "fb", "sinc", "fbl")
BANKS = [Path(x) for x in os.environ["LANET_BANKS"].split(":")]  # arms/ORDER_ARM/members/FAMILY_s0.npz from laneT/fit.sbatch (lane T release, lane B7 release)
FULL_MASS = 5 / 6
SEGS = [Path(x) for x in os.environ["LANEP_SEG"].split(":")]  # seg_{order}_{seed}_{P}.npz from laneP/seg_export.py (N9 member, seed SEGSEED)
CORR = ("ds_l5", "ds_all")  # segcorr on hardcb (sess = history resets each session; p = person history, laneK control) and on ftlite5
ALL = ARMS + CORR


def seg_file(order, p):
    hits = [d / f"seg_{order}_{SEGSEED}_{p}.npz" for d in SEGS if (d / f"seg_{order}_{SEGSEED}_{p}.npz").exists()]
    assert len(hits) == 1, (order, SEGSEED, p, hits)
    return hits[0]


def seg_rows(order, person):
    out = np.zeros((len(person), 4, 3))
    for p in sorted(set(person)):
        z = np.load(seg_file(order, p))
        out[z["rows"]] = z["seg"]
    return out


def correct_arms(order, ctx, rows, person, seg):
    """Add CORR arms to ctx['raw'] (plain dict). Pseudo-labels = frozen stock raw argmax; densities see only earlier windows of the same session
    (sess) or of the same person (p, laneK form, crosses sessions: control only)."""
    import segcorr
    raw = ctx["raw"]
    labels = raw["stock"].argmax(1)
    deliv = ctx["delivery"]
    for p in sorted(set(person)):  # the seg stream order is the chain's delivery order
        np.testing.assert_array_equal(np.load(seg_file(order, p))["delivery"], deliv[person[deliv] == p])
    groups = {"s": list(rows.values())}
    spec = {"ds_l5": ("ftlite5", "disp", "s"), "ds_all": ("hcF_l5", "disp", "s")}
    info = {}
    for name, (base, kind, g) in spec.items():
        out, act = raw[base].copy(), 0
        for ix in groups[g]:
            o, a = segcorr.correct(raw[base][ix], labels[ix], segcorr.features(seg[ix], kind))
            out[ix] = o; act += int(a.sum())
        raw[name] = out
        info[name] = dict(active=act, changed=int((out.argmax(1) != raw[base].argmax(1)).sum()))
        assert info[name]["changed"] > 0 and act > 0, name
    return info


def ft_bank(order, arm):
    out = {}
    for fam in FAMS:
        hits = [b / f"{order}_{arm}" / "members" / f"{fam}_s0.npz" for b in BANKS if (b / f"{order}_{arm}" / "members" / f"{fam}_s0.npz").exists()]
        assert len(hits) == 1, (order, arm, fam, hits)
        with np.load(hits[0], allow_pickle=False) as z:
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
            hc = 0.8 * pupils["hard"] + 0.2 * pupils["cb"]
            slot = {"identity": None, "hard": pupils["hard"], "hardcb": hc}
            for name, values in slot.items():
                if name not in ARMS:
                    continue
                outer_mod.MASS = FULL_MASS
                raw[name], used, traces[name] = replace(base, ctx, archive, patches, incumbent, values, active)
                assert used == deltas
        finally:
            outer_mod.MASS = FULL_MASS
        saved = {f: ctx.CU17[f, 0] for f in FAMS}
        proposal = 0.8 * pupils["hard"] + 0.2 * pupils["cb"]
        # ftlite5 = hardcb with the four FT family posteriors CU17[family, 0] swapped (L5 = the lite5 fit of this seed set);
        # hcF_l5 = the same plus the FBL2 doubling of the live FBLight member
        for name, doubled in (("ftlite5", False), ("hcF_l5", True)):
            bank = ft_bank(order, L5)
            try:
                for f in FAMS:
                    ctx.CU17[f, 0] = bank[f]
                outer_mod.MASS = FULL_MASS
                if doubled:
                    with doubling(ctx, archive) as boost:
                        raw[name], _, traces[name] = replace(base, ctx, archive, patches, incumbent, proposal, active)
                    assert boost["share"] == 1 / 17 and boost["warm_rows"] == 2375, (name, boost.get("share"), boost.get("warm_rows"))
                else:
                    raw[name], _, traces[name] = replace(base, ctx, archive, patches, incumbent, proposal, active)
            finally:
                for f in FAMS:
                    ctx.CU17[f, 0] = saved[f]
                outer_mod.MASS = FULL_MASS
            assert not np.array_equal(raw[name], raw["hardcb"]), name
        np.testing.assert_array_equal(raw["identity"], stock)
        assert traces["identity"]["slot_movement"] == 0
        for name in ARMS:
            if name in ("stock", "identity") or name in ("ftlite5", "hcF_l5"):
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
            yield dict(raw=raw, reference=control, zero=zero, parent=parent, traces=traces,
                       **{k: ctx[k] for k in ("count", "final", "windows", "delivery")})
    finally:
        owner.candidate = base


def run(order, smoke, cb_path, setname):
    global SETNAME, HARD, CBK, L5, SEGSEED
    SETNAME = setname
    HARD, CBK, L5, SEGSEED = SETS[setname]
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
    hard = pupils["hard"]
    seeds_hard = np.full((2, 2655, 3), np.nan)  # the two hard-pair seeds, read exactly as consume32.inputs does (mean over seeds = pupils["hard"])
    for p in sorted(set(person)):
        wanted = np.flatnonzero(person == p)
        for k in (0, 1):
            with np.load(C20_PUPILS / "pupils" / f"{p}_s{k}.npz", allow_pickle=False) as z:
                np.testing.assert_array_equal(z["index"], wanted)
                seeds_hard[k, wanted] = z[f"hard|{order}"]
    assert np.isfinite(seeds_hard).all()
    np.testing.assert_array_equal(seeds_hard.mean(0), hard)  # zero-change: the pair mean is the lane P hard input
    if HARD is not None:
        assert not np.array_equal(seeds_hard[HARD], hard)
        hard = seeds_hard[HARD]
    if CBK is not None:
        assert not np.array_equal(posts[CBK], cb)
        cb = posts[CBK]
    pupils = dict(hard=hard, cb=cb, cb_seeds=posts)
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
        corr_info = correct_arms(order, ctx, rows, person, seg_rows(order, person))
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
                if HARD is None:
                    np.testing.assert_array_equal(hard_bank[f"hard|{key}"][delivered], stages["hard"][key][delivered])
        if SETNAME == "B0" and not smoke:  # zero-change against the lane P receipt: every shared arm equal bitwise
            assert REF, "LANEB7_REF must name the lane P chain directory for B0"
            with np.load(Path(REF) / f"{order}_P.npz", allow_pickle=False) as ref:
                for a in ALL:
                    np.testing.assert_array_equal(ref[f"{a}|terminal"], stages[a]["terminal"], err_msg=a)
            print("PASS B0 equals lane P bitwise", order, ALL, flush=True)
        for key in stages["stock"]:
            np.testing.assert_array_equal(stages["stock"][key][delivered], ctx["reference"][key][delivered])
        proof = dict(passed=True, scope="smoke" if smoke else "complete", order=order, source=source, cb_pupil=[sha(f) for f in seeds],
                     arm_set=SETNAME, seeds=dict(hard=HARD, cb=CBK, l5=L5, seg=SEGSEED), corr=corr_info, delivery=stamp(ctx["delivery"]), rows=len(delivered), sessions=len(selected), parent=ctx["parent"],
                     zero=ctx["zero"], pupils=own_pins, work=work, mix=ctx["traces"], held_labels_opened=False,
                     strict_prefix=True, seconds=perf_counter() - begin)
        if not smoke:
            arrays = dict(index=np.arange(len(delivered)), delivery=delivered)
            arrays.update({f"{a}|{k}": v for a, parts in stages.items() for k, v in parts.items()})
            proof["output"] = write_npz(WORK / f"{order}_{SETNAME}.npz", arrays)
        path = WORK / (f"smoke_{SETNAME}.json" if smoke else f"{order}_{SETNAME}.json")
        write_receipt(path, **proof)
    print("PASS final-stack", order, proof["scope"], sha(path), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("order", choices=("full0", "within0"))
    p.add_argument("mode", choices=("smoke", "complete"))
    p.add_argument("cb_pupil", type=Path)
    p.add_argument("setname", choices=sorted(SETS))
    a = p.parse_args()
    assert a.mode != "smoke" or a.order == "full0"
    run(a.order, a.mode == "smoke", a.cb_pupil, a.setname)
