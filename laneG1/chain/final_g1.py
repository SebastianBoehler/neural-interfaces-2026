"""Lane G1: class-coupled SessionEM (anchors from other people + one graph edge) in the lane S2 chain (copy of final_s2.py; see evidence/wave_20nov/laneG1_emgraph.md).

usage: final_g1.py ORDER (smoke|complete) JOB   (JOB = g1:all | file:arms.json)
Only the em stage changes (mode zero|anchor|shared|edge in g1_em.py); everything below is the S2 chain.
S2 header follows.

Systems: hardcb (final stack: raw slot 0.8 hard + 0.2 CBraMod, MASS 5/6) and ds_all (hardcb + lite5 FT banks + FBL2 doubling + per-session dispersion correction).
One real chain pass per distinct (system, rebalance, EM, TSEM) setting: mixer -> rebal -> EM -> TSEM (gate, FT banks, shadow stay the frozen incumbent, as lane Z)
-> [dispersion for ds_all] -> count filter -> CSP. Arm = {name, sys, set:{stage:{const:value}}, P:{person:{stage:{const:value}}}} (P = per-person settings for the nested forms).
Stages: rebal(alpha,k,W) em(lam,n0,m0,shrink,bem,nmin) tsem(rho,shrink,m0,n0,lam,bts,nmin) csp(on,k,q,cap) count(on). DEF = shipped values.
Not tunable here (stated): SessionEM feature bins and event prior k=5 (precomputed GE features), gate lo/hi, count thresholds, urn prior.
Zero-change guards: base arms equal lane P hardcb / ds_all terminal bitwise; stock equals the release; a default-parameter stage wrapper equals the shipped function bitwise."""
import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

import outer as outer_mod
from bridge_orders import write_npz, write_receipt
from bridge_owner import pin, sha, start_module, stamp
from complete_zero import FEAT, HASHES
from cross22 import doubling
from matrix22 import parents
from outer import full_parent, replace, within_parent
from parents22 import C20, F32, PUPIL_SOURCE

WORK = Path(__file__).resolve().parent
FAMS = ("fbraw", "fb", "sinc", "fbl")
BANKS = Path(os.environ["LANET_BANKS"])
SEG = Path(os.environ["LANEP_SEG"])
PREF = Path(os.environ.get("S2_PREF", "/mnt/lustre/work/gehler/gfh527/ni26_laneP_20261120/chain"))
FULL_MASS = 5 / 6
SYSTEMS = ("hardcb", "ds_all")
from consts_g1 import ARMS, DEF  # noqa: E402
import g1_em  # noqa: E402
import time as _time  # noqa: E402
ALIGN = []
TIMING_KEY = ["", ""]
TIMING = {}  # (sys, arm) -> seconds spent inside the em stage
ANCHOR_SECONDS = []


def specs_for(job):
    if job == "g1:all":
        return [dict(name=n, sys=sy, set=dict(em=dict(v)) if v else {}) for sy in SYSTEMS for n, v in ARMS.items()]
    if job.startswith("file:"):
        return json.loads(Path(job[5:]).read_text())
    raise SystemExit(job)


def resolve(spec, people):
    """per-person full settings {person: {stage: {const: v}}} from DEF, the arm's global set and the per-person overrides."""
    out = {}
    for p in people:
        cfg = {st: dict(c) for st, c in DEF.items()}
        for src in (spec.get("set", {}), spec.get("P", {}).get(p, {})):
            for st, cs in src.items():
                assert st in cfg and set(cs) <= set(cfg[st]), (st, cs)
                cfg[st].update(cs)
        out[p] = cfg
    return out


def key(cfg, stages):
    return json.dumps({st: cfg[st] for st in stages}, sort_keys=True)


def stage_fns(gm, pdel, ydel):
    def rebal(p):
        L = gm.L
        def f(mix, proto, C):
            assert C is None
            return L.rebal(mix, L.Z[f"sess|{proto}"], alpha=p["alpha"], k=float(p["k"]), W=int(p["W"]))
        return f

    EMc = g1_em.coupled_cls(gm.EMg)
    EMc.mods = (gm.SE, gm.M.h)
    anchor_cache = {}

    def anchors_for(proto, person):
        if person not in anchor_cache:
            Z, t0 = gm.L.Z, _time.perf_counter()
            if f"person|{proto}" in Z:
                assert np.array_equal(Z[f"person|{proto}"], pdel), "tape person order differs from the delivered order"
            anchor_cache[person] = g1_em.anchors(Z[f"GE|{proto}"], ydel, Z[f"sess|{proto}"], pdel, person)
            ANCHOR_SECONDS.append(_time.perf_counter() - t0)
        return anchor_cache[person]

    def em(p):
        def f(b, proto, g):
            t0 = _time.perf_counter()
            F, sess = gm.L.Z[f"GE|{proto}"], gm.L.Z[f"sess|{proto}"]
            if p["mode"] is not None:  # label alignment guard: the stage input (rebalanced mix) must agree with the delivered-order labels
                acc = float((b.argmax(1) == ydel).mean()); ALIGN.append(acc)
                assert acc > 0.6, ("labels not aligned with the delivered tape", acc)
            out = np.empty_like(b)
            for s in dict.fromkeys(sess):
                rows = np.flatnonzero(sess == s)
                if p["mode"] is None:
                    e = gm.EMg(p["bem"], lam=p["lam"], n0=p["n0"], m0=p["m0"], shrink=p["shrink"], nmin=int(p["nmin"]))
                else:
                    assert len(set(pdel[rows])) == 1
                    e = EMc(p["bem"], lam=p["lam"], n0=p["n0"], m0=p["m0"], shrink=p["shrink"], nmin=int(p["nmin"]))
                    e.cfg = dict(mode=p["mode"], edge=tuple(p["edge"]) if p["edge"] else None, ew=p["ew"],
                                 A=None if p["mode"] == "zero" else anchors_for(proto, pdel[rows[0]]))
                    assert p["mode"] != "edge" or p["edge"]
                for r in rows:
                    e.g = g[r]; out[r] = e.step(F[r], b[r])
            TIMING[tuple(TIMING_KEY)] = TIMING.get(tuple(TIMING_KEY), 0.0) + _time.perf_counter() - t0
            return out
        return f

    def tsem(p):
        def f(P, proto, g, bts=None):
            Lk, tt = gm.L.G.tsem_lik(P, gm.L.Z[f"sess|{proto}"], gm.L.Z[f"pc128|{proto}"], rho=p["rho"], shrink=p["shrink"], m0=p["m0"], nmin=int(p["nmin"]))
            out = P.copy(); m = ~np.isnan(Lk[:, 0])
            e = p["lam"] * tt[m] / (tt[m] + p["n0"]) * (1 + p["bts"] * (gm.M.h(tt[m], 30) * g[m]))
            q = P[m] * Lk[m] ** e[:, None]; out[m] = q / q.sum(1, keepdims=True)
            return out
        return f
    return dict(rebal=rebal, em=em, tsem=tsem)


def merged(fns, name, cfgs, pdel, pre=None):
    """stage wrapper with per-person settings: run the stage once per distinct setting on all rows, take each person's rows from its own setting (the stages are per-session independent)."""
    groups = {}
    for p, cfg in cfgs.items():
        groups.setdefault(json.dumps(cfg[name], sort_keys=True), []).append(p)

    def f(inp, proto, *args):
        out = None
        for k, ps in groups.items():
            o = fns[name](json.loads(k))(inp, proto, *args)
            if out is None:
                out = o.copy()
            rows = np.isin(pdel, ps)
            out[rows] = o[rows]
        return out
    return f


def bank_ft(order, arm):
    out = {}
    for fam in FAMS:
        with np.load(BANKS / f"{order}_{arm}" / "members" / f"{fam}_s0.npz", allow_pickle=False) as z:
            assert np.array_equal(z["index"], np.arange(2655)), (arm, fam)
            out[fam] = z["causal"].astype(np.float64)
    return out


def helper(name, path):
    return start_module(name, path, path.read_text())


def build(order, pupils, specs, person, ylab):
    owner, load = (full_parent, full_parent.full0) if order == "full0" else (within_parent, within_parent.rebuilt)
    base = owner.candidate
    raws, info = {}, {}
    people = sorted(set(person))

    def visit(ctx, archive, patches, incumbent):
        assert patches and not raws
        stock, deltas = base(ctx, archive, patches, incumbent)
        raws["stock"] = stock
        active = np.ones(len(stock), dtype=bool)
        saved = {f: ctx.CU17[f, 0] for f in FAMS}
        bank = bank_ft(order, "lite5")
        pup = 0.8 * pupils["hard"] + 0.2 * pupils["cb"]
        pdel = person[ctx.Pperm]
        orig = (ctx.gm.M.rebal, ctx.gm.em_g, ctx.gm.tsem_g)
        fns = stage_fns(ctx.gm, pdel, ylab[ctx.Pperm])
        cache = {}
        for spec in specs:
            cfgs = resolve(spec, people)
            ck = (spec["sys"], json.dumps({p: key(c, ("rebal", "em", "tsem")) for p, c in cfgs.items()}, sort_keys=True))
            if ck not in cache:
                TIMING_KEY[:] = [spec["sys"], spec["name"]]
                try:
                    ctx.gm.M.rebal, ctx.gm.em_g, ctx.gm.tsem_g = (merged(fns, n, cfgs, pdel) for n in ("rebal", "em", "tsem"))
                    outer_mod.MASS = FULL_MASS
                    if spec["sys"] == "ds_all":
                        for f in FAMS:
                            ctx.CU17[f, 0] = bank[f]
                        with doubling(ctx, archive) as boost:
                            r, used, tr = replace(base, ctx, archive, patches, incumbent, pup, active)
                        assert boost["share"] == 1 / 17 and boost["warm_rows"] == 2375
                    else:
                        r, used, tr = replace(base, ctx, archive, patches, incumbent, pup, active)
                        assert used == deltas
                finally:
                    ctx.gm.M.rebal, ctx.gm.em_g, ctx.gm.tsem_g = orig
                    for f in FAMS:
                        ctx.CU17[f, 0] = saved[f]
                    outer_mod.MASS = FULL_MASS
                cache[ck] = r
                print("chain", order, spec["sys"], spec["name"], flush=True)
            raws[(spec["sys"], spec["name"])] = cache[ck]
        return stock, deltas

    owner.candidate = visit
    try:
        with load() as ctx:
            control = ctx["matched"]["stock"] if order == "full0" else ctx["reference"]
            np.testing.assert_array_equal(raws["stock"], control["raw"])
            yield dict(raws=raws, reference=control, **{k: ctx[k] for k in ("count", "final", "windows", "delivery")},
                       parent=ctx["proof"] if order == "full0" else ctx["parent"], zero=ctx["zero"] if order == "full0" else ctx["archive_zero"])
    finally:
        owner.candidate = base


def seg_rows(order, person):
    out = np.zeros((len(person), 4, 3))
    for p in sorted(set(person)):
        z = np.load(SEG / f"seg_{order}_0_{p}.npz")
        out[z["rows"]] = z["seg"]
    return out


def disperse(order, ctx, rows, person, seg, raw, stock):
    import segcorr
    for p in sorted(set(person)):
        np.testing.assert_array_equal(np.load(SEG / f"seg_{order}_0_{p}.npz")["delivery"], ctx["delivery"][person[ctx["delivery"]] == p])
    labels, out = stock.argmax(1), raw.copy()
    for ix in rows.values():
        out[ix], _ = segcorr.correct(raw[ix], labels[ix], segcorr.features(seg[ix], "disp"))
    return out


def csp_session(final, windows, raw, filtered, ix, p):
    """tails.personalize with the schedule constants of the arm (cadence k, selected share q, dose cap), the strict-prefix asserts of the original kept where they do not depend on them."""
    cp = final.CP.CausalPers(final.CONFIG)
    assert (cp.t0, cp.k, cp.w0) == (20, 20, 90)
    assert (cp.q, cp.cap) == (0.5, 0.1), (cp.q, cp.cap)  # shipped values the analysis interpolates from
    cp.k = int(p["k"])
    if p["q"] is not None:
        cp.q = float(p["q"])
    if p["cap"] is not None:
        cp.cap = float(p["cap"])
    out = np.stack([cp.step(windows[r], raw[r], filtered[r], False) for r in ix])
    assert np.isfinite(out).all() and cp.log["err"] == 0 and not cp.dead and cp.stopped is None
    np.testing.assert_array_equal(out[:90], filtered[ix[:90]])
    return out, cp.refits


def tail(ctx, order, raw, selected, cfgs, tm, person):
    count, final, windows = (ctx[k] for k in ("count", "final", "windows"))
    rows = {s: count.C.order_rows(order, s, count.PROTO) for s in selected}
    pers = {s: person[ix][0] for s, ix in rows.items()}
    assert all(len(set(person[ix])) == 1 for ix in rows.values())
    if any(cfgs[p]["count"]["on"] for p in pers.values()):
        with tm.scopes(count, selected):
            filtered, _ = tm.count_filter(count, raw, order)
        filtered = filtered.copy()
    else:
        filtered = raw.copy()
    scores = filtered.copy()
    for s, ix in rows.items():
        c = cfgs[pers[s]]
        if not c["count"]["on"]:
            filtered[ix] = raw[ix]; scores[ix] = raw[ix]
        if c["csp"]["on"]:
            scores[ix], _ = csp_session(final, windows, raw, filtered, ix, c["csp"])
    return scores


def run(order, smoke, job):
    begin = perf_counter()
    source = sha(WORK / "source.sha256")
    pin(FEAT, HASHES[FEAT])
    with np.load(FEAT, allow_pickle=False) as z:
        person = np.asarray([p.rsplit("/", 1)[-1] for p in z["subject"]])
        ylab = np.asarray(z["y"]).astype(int)  # source labels: used only for the other nine people (held person excluded in g1_em.anchors)
        visits = z["session"]
    assert len(set(person)) == 10 and len(person) == 2655
    _, receipts = parents(order)
    pupils, _ = helper("c25_f32_inputs", F32 / "consume32.py").inputs(order, person, PUPIL_SOURCE)
    tm = helper("c22_parent_tail", C20 / "worker.py")
    posts = []
    for s in range(3):
        with np.load(Path(os.environ["S2_CB"]) / f"cbbank_{order}_s{s}.npz", allow_pickle=False) as z:
            assert str(z["order"]) == order
            posts.append(z["post"]); cb_delivery = z["delivery"]
    pupils = dict(hard=pupils["hard"], cb=np.mean(posts, axis=0))
    specs = specs_for(job)
    names = [(a["sys"], a["name"]) for a in specs]
    assert len(set(names)) == len(names) and all(a["sys"] in SYSTEMS for a in specs)
    if smoke and job == "g1:all":
        specs = [a for a in specs if a["name"] in ("base", "zero", "anchor", "cog") and (a["sys"] == "hardcb" or a["name"] in ("base", "cog"))]
    people = sorted(set(person))
    gen = build(order, pupils, specs, person, ylab)
    ctx = next(gen)
    try:
        np.testing.assert_array_equal(ctx["delivery"], cb_delivery)
        count = ctx["count"]
        sessions = tuple(count.SESS)
        rows = {s: count.C.order_rows(order, s, count.PROTO) for s in sessions}
        p01 = next(s for s, r in rows.items() if set(person[r]) == {"P01"} and set(visits[r]) == {"1"})
        selected = (p01,) if smoke else sessions
        delivered = np.concatenate([rows[s] for s in selected])
        seg, stock = seg_rows(order, person), ctx["raws"]["stock"]
        out = {}
        for spec in specs:
            raw = ctx["raws"][(spec["sys"], spec["name"])]
            if spec["sys"] == "ds_all":
                raw = disperse(order, ctx, rows, person, seg, raw, stock)
            out[f"{spec['sys']}|{spec['name']}"] = tail(ctx, order, raw, selected, resolve(spec, people), tm, person)
            print("PASS tail", order, spec["sys"], spec["name"], flush=True)
        moved = {}
        for sy in SYSTEMS:  # G1 guards: zero == base bitwise; every other arm differs from base and from every other arm on the delivered rows (silent no-op guard)
            b = out.get(f"{sy}|base")
            if b is None:
                continue
            z = out.get(f"{sy}|zero")
            if z is not None:
                np.testing.assert_array_equal(z[delivered], b[delivered])
            live = {a["name"]: out[f"{sy}|{a['name']}"][delivered] for a in specs if a["sys"] == sy and a["name"] not in ("base", "zero")}
            for n, v in live.items():
                assert not np.array_equal(v, b[delivered]), ("no-op", sy, n)
                moved[f"{sy}|{n}"] = float(np.abs(v - b[delivered]).max())
            ns = sorted(live)
            for i, n in enumerate(ns):
                for m in ns[i + 1:]:
                    assert not np.array_equal(live[n], live[m]), ("arms identical", sy, n, m)
        guard = {}
        if (PREF / f"{order}_P.npz").exists():
            with np.load(PREF / f"{order}_P.npz", allow_pickle=False) as z:
                if not smoke:
                    np.testing.assert_array_equal(z["delivery"], ctx["delivery"])
                for s in SYSTEMS:
                    if f"{s}|base" in out:
                        guard[f"{s}_base_vs_P"] = float(np.abs(z[f"{s}|terminal"][delivered] - out[f"{s}|base"][delivered]).max())
                        assert guard[f"{s}_base_vs_P"] == 0.0, guard
            print("GUARD", guard, flush=True)
        arrays = dict(index=np.arange(len(delivered)), delivery=delivered)
        arrays.update({f"{k}|terminal": v.astype(np.float32) for k, v in out.items()})
        meta = dict(label_alignment_acc=ALIGN[:2], timing_em_seconds={f"{k[0]}|{k[1]}": v for k, v in TIMING.items()}, anchor_seconds=ANCHOR_SECONDS, moved_max_abs=moved, source_labels="labels of the nine other people read as source anchors only; held person excluded (asserted)", source=source, order=order, job=job, specs=specs, guard=guard, scope="smoke" if smoke else "complete", rows=len(delivered),
                    seconds=perf_counter() - begin, held_labels_opened=False, delivery=stamp(ctx["delivery"]), parent=ctx["parent"], zero=ctx["zero"])
        tag = "".join(c if c.isalnum() or c in "_-" else "_" for c in (Path(job[5:]).stem if job.startswith("file:") else job))
        if not smoke:
            meta["output"] = write_npz(WORK / f"{order}_{tag}.npz", arrays)
        write_receipt(WORK / (f"smoke_{tag}.json" if smoke else f"{order}_{tag}.json"), passed=True, **meta)
    finally:
        gen.close()
    print("PASS s2", order, tag, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("order", choices=("full0", "within0"))
    p.add_argument("mode", choices=("smoke", "complete"))
    p.add_argument("job")
    a = p.parse_args()
    assert a.mode != "smoke" or a.order == "full0"
    run(a.order, a.mode == "smoke", a.job)
