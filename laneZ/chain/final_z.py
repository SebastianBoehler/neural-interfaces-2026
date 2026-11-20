"""Lane Z chain profile: stage taps, leave-one-stage-out, injection (transmission) and mixer levers in the corrected Friday chain.

usage: Z_JOB=<group> final_z.py ORDER (smoke|complete) "cbbank_ORDER_s*.npz"
Every arm is one pass of the real chain (mix -> rebal -> SessionEM -> TSEM -> count filter -> CSP/LDA) with a spec:
  system  stock (cap250) | hardcb (raw slot = 0.8 hard + 0.2 CBraMod, MASS 5/6, as laneB1)
  ft      FT deltas on (default) | off (frozen V6r path, FT refits unreadable)
  bypass  subset of rebal, em, tsem, geb0 (EM/TSEM gate g = 0), count, csp (identity stages, rest kept)
  wmul    {mixer slot name: factor}  (slot share lever)
  inj     (target slot|mix, mode rand|wrong, frac, alpha): (1-alpha) p + alpha onehot(y) on a fixed window subset (synthetic gain)
  mix_fn  callable(members, items, perm, rows) -> mixture (learned-weight lever)
The gate, FT banks, SessionEM/TSEM shadow and refit targets stay the frozen incumbent ones (replay design): injected signal reaches them only through
the live mixer path, never through FT labels or the gate. Zero-change guard: the arm 'base' must equal the released stock raw bitwise (stock) and
FT-off must equal the released frozen V6r path. Copy of laneB1/chain/final_b.py machinery."""
import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import numpy as np

from bridge_orders import write_npz, write_receipt
from bridge_owner import pin, sha, start_module, stamp
from complete_zero import FEAT, HASHES
from matrix22 import parents
from parents22 import C20, F32, PUPIL_SOURCE

from outer import full_parent, within_parent  # noqa: E402
from upstream import UnreadableWarm, frozen_path  # noqa: E402

WORK = Path(__file__).resolve().parent
FAMILY, SLOT, MASS = "fbraw", "sc_fbraw", 5 / 6
STAGES = ("slot", "mix", "rebal", "em", "tsem", "filtered", "terminal")
ALPHAS = (0.1, 0.3, 1.0)


def helper(name, path):
    return start_module(name, path, path.read_text())


def arm_table(job):
    """name -> spec. 'prof' = baseline + leave-one-stage-out; 'tx_<target>_<mode>' = injections; 'lev_*' = levers."""
    out = {}
    systems = ("stock", "hardcb")
    if job == "prof":
        for s in systems:
            out[f"{s}|base"] = dict(system=s)
            for b in ("rebal", "em", "tsem", "geb0", "count", "csp"):
                out[f"{s}|no_{b}"] = dict(system=s, bypass={b})
            out[f"{s}|no_emtsem"] = dict(system=s, bypass={"em", "tsem"})
            out[f"{s}|no_stages"] = dict(system=s, bypass={"rebal", "em", "tsem", "count", "csp"})
            out[f"{s}|no_ft"] = dict(system=s, ft=False)
            out[f"{s}|no_ft_stages"] = dict(system=s, ft=False, bypass={"rebal", "em", "tsem", "count", "csp"})
    elif job.startswith("tx_"):
        _, target, mode = job.split("_")
        for s in systems:
            out[f"{s}|base"] = dict(system=s)
            for frac in (0.1, 0.2):
                for a in ALPHAS:
                    out[f"{s}|{target}_{mode}{int(frac*100)}_a{a}"] = dict(system=s, inj=(target, mode, frac, a))
    else:
        raise SystemExit(f"unknown job {job}")
    return out


def inj_masks(y, stock_terminal, n):
    rng = np.random.default_rng(7001)
    u, v = rng.random(n), rng.random(n)
    wrong = np.flatnonzero(stock_terminal.argmax(1) != y)
    order_w = wrong[np.argsort(v[wrong])]
    return u, order_w


def make_mask(mode, frac, u, order_w, n):
    if mode == "rand":
        return u < frac
    m = np.zeros(n, dtype=bool)
    m[order_w[: int(round(frac * n))]] = True
    return m


def onehot_mix(p, m_deliv, y_deliv, alpha):
    out = p.copy()
    oh = np.eye(3)[y_deliv]
    out[m_deliv] = (1 - alpha) * p[m_deliv] + alpha * oh[m_deliv]
    return out


def run_chain(ctx, archive, patches, incumbent, spec, pup, y, masks, order_w_ctx):
    """one pass of the corrected chain with the spec; returns (raw tsem output in recording layout, stage taps in recording layout, weights)."""
    from bridge_owner import PROTO, STACK
    perm, inv = ctx.Pperm, np.argsort(ctx.Pperm)
    ft = spec.get("ft", True)
    bypass = set(spec.get("bypass", ()))
    taps = {}
    apply, mix, gate = ctx.S.apply, ctx.A.mix, ctx.gm.gate
    old_rebal, old_em, old_tsem = ctx.gm.M.rebal, ctx.gm.em_g, ctx.gm.tsem_g
    previous_members = ctx.A.EXTRA.copy()
    saved_cu = ctx.CU17

    def apply_slots(orders):
        assert "applied" not in taps and np.array_equal(orders[PROTO], perm)
        apply(orders)
        values = ctx.Z[f"{STACK}|Ps|{PROTO}"]
        for slot, corrected in patches.items():
            values[slot] = corrected[perm]
        taps["applied"] = True

    def slot_value(chain_value):
        value = chain_value
        if spec["system"] == "hardcb":
            base = [ctx.FR17[FAMILY, k] for k in range(6)]
            other = np.mean(base[1:], axis=0)
            stock = archive[5] + np.mean([b - ctx.FR6[FAMILY, i] for i, b in enumerate(base)], axis=0)
            if ft:
                stock = stock + (ctx.CU17[FAMILY, 0] - ctx.FT_BASE[FAMILY, 0])
            value = ctx.norm(stock + MASS * (pup - other))[perm]
            if not ft:  # the unmodified slot must be what the frozen V6r chain built
                np.testing.assert_array_equal(ctx.norm(stock)[perm], chain_value)
        if "inj" in spec and spec["inj"][0] == "slot":
            _, mode, frac, alpha = spec["inj"]
            value = onehot_mix(value, make_mask(mode, frac, *masks)[perm], y[perm], alpha)
        return value

    def observe_mix(bn, proto, families, extra):
        assert (bn, proto) == (STACK, PROTO) and "mix" not in taps
        chain_value = ctx.A.EXTRA[SLOT][PROTO]
        value = slot_value(chain_value)
        wm = spec.get("wmul", {})
        extra = [(k, w * wm.get(k, 1.0)) for k, w in extra]
        try:
            ctx.A.EXTRA[SLOT][PROTO] = value
            members = ctx.A.members(bn, proto)
            items = [item for f in families for item in ctx.A.FAM[f]] + list(extra)
            taps["members"] = {str(k): v[inv].astype(np.float32) for k, v in members.items()}
            taps["items"] = [(str(k), float(w)) for k, w in items]
            taps["slot"] = value[inv]
            if "mix_fn" in spec:
                out = (spec["mix_fn"](members, items, perm), 0.0)
            else:
                out = mix(bn, proto, families, extra)
        finally:
            ctx.A.EXTRA[SLOT][PROTO] = chain_value
        m = out[0]
        if "inj" in spec and spec["inj"][0] == "mix":
            _, mode, frac, alpha = spec["inj"]
            m = onehot_mix(m, make_mask(mode, frac, *masks)[perm], y[perm], alpha)
        taps["mix"] = m[inv]
        return m, out[1]

    def stage(kind, fun, name):
        def run(inp, proto, *args):
            out = inp if name in bypass else fun(inp, proto, *args)
            taps[kind] = out[inv]
            return out
        return run

    def geb(values, proto):
        out = gate(values, proto)
        return np.zeros_like(out) if "geb0" in bypass else out

    ctx.S.apply, ctx.A.mix, ctx.gm.gate = apply_slots, observe_mix, geb
    ctx.gm.M.rebal = stage("rebal", old_rebal, "rebal")
    ctx.gm.em_g = stage("em", old_em, "em")
    ctx.gm.tsem_g = stage("tsem", old_tsem, "tsem")
    arm = "V6Fr" if ft else "V6r"
    if not ft:
        ctx.CU17 = UnreadableWarm()
    try:
        ctx.chain(STACK, arm, perm)
    finally:
        ctx.S.apply, ctx.A.mix, ctx.gm.gate = apply, mix, gate
        ctx.gm.M.rebal, ctx.gm.em_g, ctx.gm.tsem_g = old_rebal, old_em, old_tsem
        ctx.CU17 = saved_cu
        ctx.A.EXTRA.clear()
        ctx.A.EXTRA.update(previous_members)
    raw = ctx.OUTS[f"{STACK}|{arm}"].astype(np.float64)
    np.testing.assert_array_equal(raw.astype(np.float32), taps["tsem"].astype(np.float32))
    return raw, taps


def tail(ctx, order, raw, selected, bypass, tm):
    count, final, windows = (ctx[k] for k in ("count", "final", "windows"))
    rows = {s: count.C.order_rows(order, s, count.PROTO) for s in selected}
    if "count" in bypass:
        filtered = raw.copy()
    else:
        with tm.scopes(count, selected):
            filtered, _ = tm.count_filter(count, raw, order)
    scores = filtered.copy()
    if "csp" not in bypass:
        for s, ix in rows.items():
            scores[ix], _ = tm.personalize(final, windows, raw, filtered, ix)
    return filtered, scores


B1REF = Path(os.environ.get("Z_B1REF", "/mnt/lustre/work/gehler/gfh527/ni26_b1_20261120/chain"))


@contextmanager
def session(order, runner):
    """open the real chain context; runner(ctx, archive, patches, incumbent, stock) runs inside the owner's candidate hook (as laneB1 does)."""
    owner, load = (full_parent, full_parent.full0) if order == "full0" else (within_parent, within_parent.rebuilt)
    base = owner.candidate
    seen = {}

    def visit(ctx, archive, patches, incumbent):
        assert not seen
        stock, deltas = base(ctx, archive, patches, incumbent)
        seen["stock"] = stock
        runner(ctx, archive, patches, incumbent, stock)
        return stock, deltas

    owner.candidate = visit
    try:
        with load() as ctx:
            control = ctx["matched"]["stock"] if order == "full0" else ctx["reference"]
            np.testing.assert_array_equal(seen["stock"], control["raw"])
            yield ctx, control
    finally:
        owner.candidate = base


def run(order, smoke, cb_path, job):
    begin = perf_counter()
    source = sha(WORK / "source.sha256")
    pin(FEAT, HASHES[FEAT])
    with np.load(FEAT, allow_pickle=False) as z:
        person = np.asarray([p.rsplit("/", 1)[-1] for p in z["subject"]])
        visits, y = z["session"], z["y"].astype(int)
    assert len(set(person)) == 10 and len(person) == 2655 and set(y) == {0, 1, 2}
    _, receipts = parents(order)
    pupils, own_pins = helper("c25_f32_inputs", F32 / "consume32.py").inputs(order, person, PUPIL_SOURCE)
    tm = helper("c22_parent_tail", C20 / "worker.py")
    seeds = sorted(Path(cb_path).parent.glob(Path(cb_path).name))
    assert len(seeds) == 3, seeds
    posts = []
    for f in seeds:
        with np.load(f, allow_pickle=False) as z:
            assert str(z["order"]) == order
            posts.append(z["post"])
            cb_delivery = z["delivery"]
    cb = np.mean(posts, axis=0)
    np.testing.assert_allclose(cb.sum(1), 1.0, atol=1e-6, rtol=0)
    pup_hc = 0.8 * pupils["hard"] + 0.2 * cb
    arms = arm_table(job)
    if smoke:
        keep = [k for k in arms if k.endswith("|base") or "no_ft" in k or "no_csp" in k or "a0.3" in k]
        arms = {k: arms[k] for k in keep[:5]}
    out, times = {}, {}
    holder = {}

    def runner(ctx, archive, patches, incumbent, stock):
        masks = inj_masks(y, holder["control"]["terminal"], len(y))
        for name, spec in arms.items():
            t0 = perf_counter()
            raw, taps = run_chain(ctx, archive, patches, incumbent, spec, pup_hc if spec["system"] == "hardcb" else None, y, masks, None)
            out[name] = dict(raw=raw, taps=taps, spec=spec)
            times[name] = perf_counter() - t0
            print("chain", order, name, round(times[name], 1), flush=True)
        if "stock|base" in out:
            np.testing.assert_array_equal(out["stock|base"]["raw"], stock)
        if "stock|no_ft" in out:
            np.testing.assert_array_equal(out["stock|no_ft"]["raw"], frozen_path(ctx, archive, patches, incumbent, "V6r"))
            assert not np.array_equal(out["stock|no_ft"]["raw"], stock)

    # the stock terminal that defines the 'wrong' windows is the released one: read it before the chains run
    holder["control"] = released_control(order)
    with session(order, runner) as (ctx, control):
        for k in ("raw", "filtered", "terminal"):
            np.testing.assert_array_equal(control[k], holder["control"][k])
        np.testing.assert_array_equal(ctx["delivery"], cb_delivery)
        count = ctx["count"]
        sessions = tuple(count.SESS)
        assert len(sessions) == 14
        rows = {s: count.C.order_rows(order, s, count.PROTO) for s in sessions}
        p01 = next(s for s, r in rows.items() if set(person[r]) == {"P01"} and set(visits[r]) == {"1"})
        selected = tuple(sessions) if not smoke else (p01,)
        delivered = np.concatenate([rows[s] for s in selected])
        arrays = dict(index=np.arange(len(delivered)), delivery=delivered, y=y)
        info = {}
        for name, o in out.items():
            spec = o["spec"]
            t0 = perf_counter()
            filtered, terminal = tail(ctx, order, o["raw"], selected, set(spec.get("bypass", ())), tm)
            info[name] = dict(chain_s=times[name], tail_s=perf_counter() - t0)
            taps = o["taps"]
            for k in ("slot", "mix", "rebal", "em", "tsem"):
                arrays[f"{name}|{k}"] = np.asarray(taps[k], dtype=np.float32)
            arrays[f"{name}|filtered"], arrays[f"{name}|terminal"] = filtered.astype(np.float32), terminal.astype(np.float32)
            if name.endswith("|base"):
                for k, v in taps["members"].items():
                    arrays[f"{name}|member|{k}"] = v
                info[name]["items"] = taps["items"]
            print("tail", order, name, round(info[name]["tail_s"], 1), flush=True)
            if name == "stock|base" and not smoke:
                for k in ("filtered", "terminal"):
                    np.testing.assert_array_equal({"filtered": filtered, "terminal": terminal}[k][delivered], control[k][delivered])
            if name == "hardcb|base" and not smoke:
                with np.load(B1REF / f"{order}_S.npz", allow_pickle=False) as z:
                    for k, v in (("raw", o["raw"]), ("filtered", filtered), ("terminal", terminal)):
                        np.testing.assert_array_equal(v[delivered].astype(np.float64), z[f"hardcb|{k}"][delivered].astype(np.float64))
                info[name]["hardcb_equals_b1"] = True
        proof = dict(passed=True, scope="smoke" if smoke else "complete", job=job, order=order, source=source, cb_pupil=[sha(f) for f in seeds],
                     delivery=stamp(ctx["delivery"]), rows=len(delivered), arms=info, specs={n: {k: str(v) for k, v in o["spec"].items() if k != "mix_fn"} for n, o in out.items()},
                     held_labels_opened=False, labels_used_only_for_synthetic_injection=True, recognition_replayed=False, seconds=perf_counter() - begin)
        if not smoke:
            proof["output"] = write_npz(WORK / f"{order}_{job}.npz", arrays)
        path = WORK / (f"smoke_{job}.json" if smoke else f"{order}_{job}.json")
        write_receipt(path, **proof)
    print("PASS lane Z", order, job, proof["scope"], sha(path), flush=True)


def released_control(order):
    """released stock raw/filtered/terminal (recording layout) from the pinned matched banks, before any arm runs"""
    from complete_zero import read  # noqa: F401
    if order == "full0":
        from inputs import FULL, FULL_BANK
        path, digest = FULL / "matched_full0.npz", FULL_BANK
    else:
        from upstream import PARENT, PARENT_ARRAYS
        path, digest = PARENT / "matched_within0.npz", PARENT_ARRAYS
    pin(path, digest)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[f"corrected|cap250_{k}"].astype(np.float64) for k in ("raw", "filtered", "terminal")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("order", choices=("full0", "within0"))
    p.add_argument("mode", choices=("smoke", "complete"))
    p.add_argument("cb_pupil", type=Path)
    a = p.parse_args()
    run(a.order, a.mode == "smoke", a.cb_pupil, os.environ["Z_JOB"])
