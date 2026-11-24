"""Lane K2: ideas 4, 5, 6 of research2_codex.md in the corrected Friday chain (copy of lane S2 machinery; registered in evidence/wave_20nov/laneK2_ideas456.md).

usage: final_k2.py ORDER (smoke|complete) JOB   (JOB = k:i/n arm part i of n | file:arms.json)
Systems: hardcb (final stack) and ds_l5 (hardcb + lite5 FT banks + per-session dispersion correction; lane P P4).
Arms: base; prior.* (idea 5: responsibilities of the SessionEM template fit, chain pass); x4*/x6* (ideas 4 and 6: injection on the post-TSEM raw posterior, before count filter + CSP).
Zero-change guards: base arms equal lane P hardcb / ds_l5 terminal bitwise; weight-0 injection arms equal base bitwise; a non-zero arm must change the output."""
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
import prior_k2

WORK = Path(__file__).resolve().parent
FAMS = ("fbraw", "fb", "sinc", "fbl")
BANKS = Path(os.environ["LANET_BANKS"])
SEG = Path(os.environ["LANEP_SEG"])
PREF = Path(os.environ.get("S2_PREF", "/mnt/lustre/work/gehler/gfh527/ni26_laneP_20261120/chain"))
FULL_MASS = 5 / 6
SYSTEMS = ("hardcb", "ds_l5")
from consts_k2 import CLIP4, DEF, PRIORS, SCALES, W4, W6  # noqa: E402
STATE = {}  # per chain pass: pre-rebalance mix (captured by the rebalance stage), labels and person in delivery order, prior diagnostics


def grid():
    arms = [dict(name="base", sys=s, set={}) for s in SYSTEMS]
    for s in SYSTEMS:
        arms += [dict(name="zero4", sys=s, set={}, X=dict(kind="x4c", s=0.0), zero=True), dict(name="zero6", sys=s, set={}, X=dict(kind="x6g", s=0.0), zero=True)]
        arms += [dict(name=f"prior.{k}", sys=s, set={"em": {"prior": k}}) for k in PRIORS]
        for sc in SCALES:
            for kind in ("x4c", "x4m", "x4p0", "x4p1", "x4p2", "x6g", "x6a"):
                arms.append(dict(name=f"{kind}_s{sc}", sys=s, set={}, X=dict(kind=kind, s=sc)))
        arms.append(dict(name="x4o_s1.0", sys=s, set={}, X=dict(kind="x4o", s=1.0)))
    return arms


def specs_for(job):
    if job.startswith("k:"):
        i, n = (int(x) for x in job[2:].split("/"))
        names = sorted({a["name"] for a in grid() if a["name"] != "base" and not a.get("zero")})
        mine = {x for j, x in enumerate(names) if j % n == i} | {"base"} | ({"zero4", "zero6"} if i == 0 else set())
        return [a for a in grid() if a["name"] in mine]
    if job.startswith("file:"):
        return json.loads(Path(job[5:]).read_text())
    raise SystemExit(job)


def inject(raw, X, KF, y):
    """ideas 4 and 6 on the post-TSEM raw posterior (N,3). x4c/x4m/x4p*: cross-view class evidence on the top two classes, 0.15 x scale x centred ll, clipped +-0.5; x6g/x6a: 0.10 x scale x centred log-evidence; x4o: oracle ceiling."""
    s = X["s"]
    if s == 0.0:
        return raw.copy()
    lg, r = np.log(np.clip(raw, 1e-12, None)), np.arange(len(raw))
    o = np.argsort(-raw, 1); a, b = o[:, 0], o[:, 1]
    if X["kind"] == "x4o":  # labels: true class among the top two gets +0.5, the other candidate -0.5
        hit = (a == y) | (b == y)
        d = np.where(hit, 0.5, 0.0); other = np.where(a == y, b, a)
        lg[r, y] += np.where(hit, d, 0.0); lg[r, other] -= np.where(hit, d, 0.0)
    elif X["kind"].startswith("x4"):
        L = KF[{"x4c": "ll4c", "x4m": "ll4m", "x4p0": "ll4p0", "x4p1": "ll4p1", "x4p2": "ll4p2"}[X["kind"]]]
        c = np.stack([L[r, a], L[r, b]], 1); c = c - c.mean(1, keepdims=True)
        d = np.clip(W4 * s * c, -CLIP4, CLIP4)
        lg[r, a] += d[:, 0]; lg[r, b] += d[:, 1]
    else:
        lp = KF[{"x6g": "x6g", "x6a": "x6a"}[X["kind"]]]
        lg += W6 * s * (lp - lp.mean(1, keepdims=True))
    out = np.exp(lg - lg.max(1, keepdims=True))
    return out / out.sum(1, keepdims=True)


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


def stage_fns(gm):
    def rebal(p):
        L = gm.L
        def f(mix, proto, C):
            assert C is None
            STATE["mix"] = np.array(mix, copy=True)  # pre-rebalance posterior, used by the prior arms as responsibilities
            return L.rebal(mix, L.Z[f"sess|{proto}"], alpha=p["alpha"], k=float(p["k"]), W=int(p["W"]))
        return f

    def em(p):
        def f(b, proto, g):
            F, sess = gm.L.Z[f"GE|{proto}"], gm.L.Z[f"sess|{proto}"]
            out = np.empty_like(b)
            pr = p["prior"]
            if pr is not None:
                mix, y, pdel = STATE["mix"], STATE["y"], STATE["pdel"]
                assert mix.shape == b.shape and len(y) == len(b) == len(pdel)
                stats, diag = {}, []
            for s in dict.fromkeys(sess):
                e = gm.EMg(p["bem"], lam=p["lam"], n0=p["n0"], m0=p["m0"], shrink=p["shrink"], nmin=int(p["nmin"]))
                rr = np.flatnonzero(sess == s)
                if pr is not None:
                    who = pdel[rr[0]]; assert (pdel[rr] == who).all()
                    if who not in stats:
                        stats[who] = prior_k2.source_stats(mix, y, pdel != who)
                    truth = np.bincount(y[rr], minlength=3) / len(rr)
                    if pr == "oracle":
                        pi, status = np.tile(truth, (len(rr), 1)), np.full(len(rr), prior_k2.ACTIVE)
                    elif pr == "uniform":
                        pi, status = np.tile(prior_k2.U, (len(rr), 1)), np.zeros(len(rr), dtype=int)
                    else:
                        pi, status = prior_k2.session_prior(mix[rr], stats[who], PRIORS[pr])
                    resp = prior_k2.responsibilities(mix[rr], pi)
                    diag += [(i, status[i], np.abs(pi[i] - truth).sum(), np.abs(prior_k2.U - truth).sum()) for i in range(len(rr))]
                for j, r in enumerate(rr):
                    e.g = g[r]; out[r] = e.step(F[r], b[r])
                    if pr is not None:
                        e.B[-1] = resp[j]  # the template fit sees the prior-corrected responsibility; the output posterior stays stock
            if pr is not None:
                STATE.setdefault("diag", {})[pr] = prior_k2.diagnose(diag)
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


def build(order, pupils, specs, person, y):
    owner, load = (full_parent, full_parent.full0) if order == "full0" else (within_parent, within_parent.rebuilt)
    base = owner.candidate
    raws, info = {}, {}
    diags = info.setdefault("prior", {})
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
        fns = stage_fns(ctx.gm)
        cache = {}
        for spec in specs:
            cfgs = resolve(spec, people)
            ck = (spec["sys"], json.dumps({p: key(c, ("rebal", "em", "tsem")) for p, c in cfgs.items()}, sort_keys=True))
            if ck not in cache:
                STATE.clear(); STATE.update(y=y[ctx.Pperm], pdel=pdel)
                try:
                    ctx.gm.M.rebal, ctx.gm.em_g, ctx.gm.tsem_g = (merged(fns, n, cfgs, pdel) for n in ("rebal", "em", "tsem"))
                    outer_mod.MASS = FULL_MASS
                    if spec["sys"] == "ds_l5":
                        for f in FAMS:
                            ctx.CU17[f, 0] = bank[f]
                        r, used, tr = replace(base, ctx, archive, patches, incumbent, pup, active)
                    else:
                        r, used, tr = replace(base, ctx, archive, patches, incumbent, pup, active)
                        assert used == deltas
                finally:
                    ctx.gm.M.rebal, ctx.gm.em_g, ctx.gm.tsem_g = orig
                    for f in FAMS:
                        ctx.CU17[f, 0] = saved[f]
                    outer_mod.MASS = FULL_MASS
                cache[ck] = r
                if "diag" in STATE:
                    diags[f"{spec['sys']}|{spec['name']}"] = STATE["diag"]
                print("chain", order, spec["sys"], spec["name"], flush=True)
            raws[(spec["sys"], spec["name"])] = cache[ck]
        return stock, deltas

    owner.candidate = visit
    try:
        with load() as ctx:
            control = ctx["matched"]["stock"] if order == "full0" else ctx["reference"]
            np.testing.assert_array_equal(raws["stock"], control["raw"])
            yield dict(raws=raws, reference=control, info=info, **{k: ctx[k] for k in ("count", "final", "windows", "delivery")},
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
        visits = z["session"]
        y = z["y"].astype(np.int64)
    KF = dict(np.load(WORK / "feat_k2.npz", allow_pickle=False))
    assert np.array_equal(KF["y"], y) and np.array_equal(KF["person"], person)
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
    if smoke:
        keep = ("base", "zero4", "zero6", "prior.uniform", "prior.feas", "prior.oracle", "x4c_s1.0", "x4o_s1.0", "x6g_s1.0")
        specs = [a for a in grid() if a["name"] in keep]
    people = sorted(set(person))
    gen = build(order, pupils, specs, person, y)
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
            if spec["sys"] == "ds_l5":
                raw = disperse(order, ctx, rows, person, seg, raw, stock)
            if spec.get("X"):
                raw = inject(raw, spec["X"], KF, y)
            out[f"{spec['sys']}|{spec['name']}"] = tail(ctx, order, raw, selected, resolve(spec, people), tm, person)
            print("PASS tail", order, spec["sys"], spec["name"], flush=True)
        for spec in specs:  # a move must change the output (silent no-op guard); per-person forms may equal base
            b = out.get(f"{spec['sys']}|base")
            same = b is not None and np.array_equal(out[f"{spec['sys']}|{spec['name']}"][delivered], b[delivered])
            if spec.get("zero"):
                assert same, ("zero-change arm differs from base", spec["name"])
            elif b is not None and (spec.get("set") or spec.get("X")):
                assert not same, ("no-op", spec["name"])
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
        meta = dict(source=source, order=order, job=job, specs=specs, guard=guard, scope="smoke" if smoke else "complete", rows=len(delivered),
                    seconds=perf_counter() - begin, labels_used="LOSO source fits and oracle arms (dev instrument)", prior_diag=ctx["info"]["prior"], delivery=stamp(ctx["delivery"]), parent=ctx["parent"], zero=ctx["zero"])
        tag = "".join(c if c.isalnum() or c in "_-" else "_" for c in (Path(job[5:]).stem if job.startswith("file:") else job))
        if not smoke:
            meta["output"] = write_npz(WORK / f"{order}_{tag}.npz", arrays)
        write_receipt(WORK / (f"smoke_{tag}.json" if smoke else f"{order}_{tag}.json"), passed=True, **meta)
    finally:
        gen.close()
    print("PASS k2", order, tag, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("order", choices=("full0", "within0"))
    p.add_argument("mode", choices=("smoke", "complete"))
    p.add_argument("job")
    a = p.parse_args()
    assert a.mode != "smoke" or a.order == "full0"
    run(a.order, a.mode == "smoke", a.job)
