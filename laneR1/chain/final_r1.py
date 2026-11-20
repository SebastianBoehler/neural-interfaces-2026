"""Lane R1 big stack: ds_all (hardcb + lite5 + FBL2 + per-session dispersion correction) with a re-weightable mixer, in the corrected Friday chain.

usage: final_r1.py ORDER (smoke|complete) JOB   (JOB = w1:i/n coordinate arms part i of n | file:arms.json  | base)
Each arm = one real chain pass: mixer (stock weights x multipliers, optional extra members, optional per-person multipliers) -> rebal -> EM -> TSEM
-> dispersion correction (frozen stock-raw pseudo-labels, per-session history) -> count/CSP tail. Arm spec: {name, mult:{key:f}, add:{bank:units}, W:{person:{key:f}}, dump}.
Mixer items (stock weights): c1 1, hb 1, 1 .5, 2 .5, 3 1, 4 1, 10 5 (teacher), 9 1, sc_fbraw sc_fb sc_sinc sc_fbl sc_rest 1 each (sum 16). Extra members: reve, prat, cbx (full0 only), xfam.
Zero-change guards: the mixer rebuilt from the tapped members equals the real mix (<1e-12); arm `base` equals lane P ds_all terminal bitwise; stock equals the release."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

import outer as outer_mod
from cross22 import doubling
from bridge_orders import write_npz, write_receipt
from bridge_owner import pin, sha, start_module, stamp
from complete_zero import FEAT, HASHES
from matrix22 import parents
from outer import full_parent, replace, within_parent
from parents22 import C20, F32, PUPIL_SOURCE

WORK = Path(__file__).resolve().parent
FAMS = ("fbraw", "fb", "sinc", "fbl")
BANKS = Path(os.environ["LANET_BANKS"])
SEG = Path(os.environ["LANEP_SEG"])
PREF = Path(os.environ.get("R1_PREF", "/mnt/lustre/work/gehler/gfh527/ni26_laneP_20261120/chain"))
FM = Path("/mnt/lustre/work/gehler/gfh527/ni26_fm_banks_20261119")
X_BANK = Path("/mnt/lustre/work/gehler/gfh527/ni26_laneB6_20261120/ensemble_X_held_new.npz")
KEYS = ("c1", "hb", "1", "2", "3", "4", "10", "9", "sc_fbraw", "sc_fb", "sc_sinc", "sc_fbl", "sc_rest")
EXTRAS = ("reve", "prat", "cbx", "xfam")
FULL_MASS = 5 / 6


def wave1():
    arms = [dict(name="base", dump=True)]
    for k in KEYS:
        for f in (0.5, 2.0):
            arms.append(dict(name=f"m_{k}_x{f}", mult={k: f}))
    for e in EXTRAS:
        for u in (1.0, 3.0):
            arms.append(dict(name=f"a_{e}_{u}", add={e: u}))
    return arms


def specs_for(job, order):
    if job == "base":
        arms = [dict(name="base", dump=True)]
    elif job.startswith("w1:"):
        i, n = (int(x) for x in job[3:].split("/"))
        arms = [a for j, a in enumerate(wave1()) if j % n == i]
    elif job.startswith("file:"):
        arms = json.loads(Path(job[5:]).read_text())
    else:
        raise SystemExit(job)
    if order != "full0":
        arms = [a for a in arms if not a.get("add")]  # extra banks exist for the full orders only
    names = [a["name"] for a in arms]
    assert len(set(names)) == len(names) and "stock" not in names
    return arms


def extra_banks(order):
    if order != "full0":
        return {}
    out = {}
    for tag, path in (("reve", FM / "reve"), ("prat", FM / "pratibha"), ("cbx", FM / "cbramod")):
        with np.load(path / "ensemble_C_held_new.npz", allow_pickle=False) as z:
            out[tag] = (z["full0"].astype(np.float64), z["full0_delivery"])
    with np.load(X_BANK, allow_pickle=False) as z:
        out["xfam"] = (z["full0"].astype(np.float64), z["full0_delivery"])
    for tag, (v, _) in out.items():
        assert v.shape == (2655, 3) and np.isfinite(v).all() and np.allclose(v.sum(1), 1, atol=2e-6), tag
    return out


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


def seg_rows(order, person):
    out = np.zeros((len(person), 4, 3))
    for p in sorted(set(person)):
        z = np.load(SEG / f"seg_{order}_0_{p}.npz")
        out[z["rows"]] = z["seg"]
    return out


def make_mix(ctx, spec, person, extras, taps):
    perm = ctx.Pperm
    pdel = person[perm]
    real = ctx.A.mix

    def mix(bn, proto, families, extra):
        out = real(bn, proto, families, extra)
        items = [(str(k), float(w)) for f in families for k, w in ctx.A.FAM[f]] + [(str(k), float(w)) for k, w in extra]
        assert tuple(k for k, _ in items) == KEYS, [k for k, _ in items]
        members = ctx.A.members(bn, proto)
        w0 = np.array([w for _, w in items])
        cols = [members[k] for k, _ in items]
        plain = sum(w * c for w, c in zip(w0, cols)) / w0.sum()
        taps["mixdiff"] = max(taps.get("mixdiff", 0.0), float(np.abs(plain - out[0]).max()))
        assert taps["mixdiff"] < 1e-12, taps["mixdiff"]
        if spec.get("dump"):
            taps["members"] = {k: members[k][np.argsort(perm)].astype(np.float32) for k, _ in items}
        mult, add, wp = spec.get("mult", {}), spec.get("add", {}), spec.get("W", {})
        if not (mult or add or wp):
            return out
        assert set(mult) <= set(KEYS) and set(add) <= set(extras), (mult, add)
        n = len(pdel)
        W = np.tile(w0, (n, 1)) * np.array([mult.get(k, 1.0) for k in KEYS])
        if wp:
            for p in set(pdel):
                m = pdel == p
                W[m] *= np.array([wp[p].get(k, 1.0) for k in KEYS])
        mats = cols + [extras[e][0][perm] for e in add]
        W = np.concatenate([W, np.tile([add[e] for e in add], (n, 1))], 1) if add else W
        acc = sum(W[:, i:i + 1] * c for i, c in enumerate(mats)) / W.sum(1, keepdims=True)
        assert np.allclose(acc.sum(1), 1, atol=1e-9) and np.isfinite(acc).all()
        return acc, out[1]
    return mix


def build_raws(order, pupils, specs, person, extras):
    owner, load = (full_parent, full_parent.full0) if order == "full0" else (within_parent, within_parent.rebuilt)
    base = owner.candidate
    raw, info = {}, {}

    def visit(ctx, archive, patches, incumbent):
        assert patches and not raw
        stock, deltas = base(ctx, archive, patches, incumbent)
        raw["stock"] = stock
        active = np.ones(len(stock), dtype=bool)
        saved = {f: ctx.CU17[f, 0] for f in FAMS}
        bank = ft_bank(order, "lite5")
        pup = 0.8 * pupils["hard"] + 0.2 * pupils["cb"]
        for spec in specs:
            taps = {}
            old_mix = ctx.A.mix
            try:
                for f in FAMS:
                    ctx.CU17[f, 0] = bank[f]
                outer_mod.MASS = FULL_MASS
                ctx.A.mix = make_mix(ctx, spec, person, extras, taps)
                with doubling(ctx, archive) as boost:
                    raw[spec["name"]], _, tr = replace(base, ctx, archive, patches, incumbent, pup, active)
                assert boost["share"] == 1 / 17 and boost["warm_rows"] == 2375, (spec["name"], boost)
            finally:
                ctx.A.mix = old_mix
                for f in FAMS:
                    ctx.CU17[f, 0] = saved[f]
                outer_mod.MASS = FULL_MASS
            info[spec["name"]] = dict(mixdiff=taps.get("mixdiff"), slot_movement=tr["slot_movement"], members=taps.get("members"))
            print("raw", order, spec["name"], flush=True)
        return stock, deltas

    owner.candidate = visit
    try:
        with load() as ctx:
            control = ctx["matched"]["stock"] if order == "full0" else ctx["reference"]
            np.testing.assert_array_equal(raw["stock"], control["raw"])
            yield_ = dict(raw=raw, info=info, reference=control, parent=ctx["proof"] if order == "full0" else ctx["parent"],
                          zero=ctx["zero"] if order == "full0" else ctx["archive_zero"], **{k: ctx[k] for k in ("count", "final", "windows", "delivery")})
            yield yield_
    finally:
        owner.candidate = base


def disp_correct(order, ctx, rows, person, seg, raw):
    import segcorr
    labels = raw["stock"].argmax(1)
    deliv = ctx["delivery"]
    for p in sorted(set(person)):
        np.testing.assert_array_equal(np.load(SEG / f"seg_{order}_0_{p}.npz")["delivery"], deliv[person[deliv] == p])
    for name in [k for k in raw if k != "stock"]:
        out = raw[name].copy()
        for ix in rows.values():
            out[ix], _ = segcorr.correct(raw[name][ix], labels[ix], segcorr.features(seg[ix], "disp"))
        raw[name + "|d"] = out


def run(order, smoke, job):
    begin = perf_counter()
    source = sha(WORK / "source.sha256")
    pin(FEAT, HASHES[FEAT])
    with np.load(FEAT, allow_pickle=False) as z:
        person = np.asarray([p.rsplit("/", 1)[-1] for p in z["subject"]])
        visits, y = z["session"], z["y"].astype(int)
    assert len(set(person)) == 10 and len(person) == 2655
    _, receipts = parents(order)
    pupils, _ = helper("c25_f32_inputs", F32 / "consume32.py").inputs(order, person, PUPIL_SOURCE)
    run_tail = helper("c22_parent_tail", C20 / "worker.py").run_tail
    posts = []
    for s in range(3):
        with np.load(Path(os.environ["R1_CB"]) / f"cbbank_{order}_s{s}.npz", allow_pickle=False) as z:
            assert str(z["order"]) == order
            posts.append(z["post"]); cb_delivery = z["delivery"]
    pupils = dict(hard=pupils["hard"], cb=np.mean(posts, axis=0))
    specs = specs_for(job, order)
    if smoke:
        specs = [a for a in specs if a["name"] == "base"] + [a for a in specs if a.get("mult") or a.get("add")][:1]
    extras = extra_banks(order)
    for tag, (_, d) in extras.items():
        np.testing.assert_array_equal(d, cb_delivery)
    gen = build_raws(order, pupils, specs, person, extras)
    ctx = next(gen)
    try:
        np.testing.assert_array_equal(ctx["delivery"], cb_delivery)
        count = ctx["count"]
        sessions = tuple(count.SESS)
        rows = {s: count.C.order_rows(order, s, count.PROTO) for s in sessions}
        p01 = next(s for s, r in rows.items() if set(person[r]) == {"P01"} and set(visits[r]) == {"1"})
        latch = count.latch(order)
        peer = next(s for s in sessions if s != p01 and latch[s] is not None and latch[s] < 120)
        prefix = (p01, peer, p01)
        selected = (p01, peer) if smoke else sessions
        delivered = np.concatenate([rows[s] for s in selected])
        raw = ctx["raw"]
        disp_correct(order, ctx, rows, person, seg_rows(order, person), raw)
        out, work = {}, {}
        for arm in ["stock"] + [s["name"] + "|d" for s in specs]:
            out[arm], work[arm] = run_tail(ctx, order, raw[arm], selected, prefix)
            assert work[arm]["count_effect"] > 0 and work[arm]["csp_refits"] > 0
            print("PASS tail", order, arm, flush=True)
        for key in out["stock"]:
            np.testing.assert_array_equal(out["stock"][key][delivered], ctx["reference"][key][delivered])
        guard = {}
        if not smoke and (PREF / f"{order}_P.npz").exists():
            with np.load(PREF / f"{order}_P.npz", allow_pickle=False) as z:
                np.testing.assert_array_equal(z["delivery"], ctx["delivery"])
                guard["stock_vs_P"] = float(np.abs(z["stock|terminal"] - out["stock"]["terminal"]).max())
                if "base|d" in out:
                    guard["base_vs_P_ds_all"] = float(np.abs(z["ds_all|terminal"] - out["base|d"]["terminal"]).max())
            print("GUARD", guard, flush=True)
        arrays = dict(index=np.arange(len(delivered)), delivery=delivered, y=y, person=person)
        for a, parts in out.items():
            arrays[f"{a}|terminal"] = parts["terminal"]
        for s in specs:
            for k, v in (ctx["info"][s["name"]].get("members") or {}).items():
                arrays[f"member|{k}"] = v
        meta = dict(source=source, order=order, job=job, specs=specs, guard=guard, scope="smoke" if smoke else "complete",
                    mixdiff={k: v["mixdiff"] for k, v in ctx["info"].items()}, delivery=stamp(ctx["delivery"]), rows=len(delivered),
                    seconds=perf_counter() - begin, held_labels_opened=False, parent=ctx["parent"], zero=ctx["zero"], work=work)
        tag = "".join(c if c.isalnum() or c in "_-" else "_" for c in (Path(job[5:]).stem if job.startswith("file:") else job))
        if not smoke:
            meta["output"] = write_npz(WORK / f"{order}_{tag}.npz", arrays)
        write_receipt(WORK / (f"smoke_{tag}.json" if smoke else f"{order}_{tag}.json"), passed=True, **meta)
    finally:
        gen.close()
    print("PASS r1", order, tag, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("order", choices=("full0", "within0"))
    p.add_argument("mode", choices=("smoke", "complete"))
    p.add_argument("job")
    a = p.parse_args()
    assert a.mode != "smoke" or a.order == "full0"
    run(a.order, a.mode == "smoke", a.job)
