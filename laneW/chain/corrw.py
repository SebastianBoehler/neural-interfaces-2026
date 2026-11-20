"""Lane W: widened post-mixer correction family (evidence/wave_20nov/laneW_corrections.md). Pure numpy, causal, history resets at every session.
Same density/correction arithmetic as laneP/chain/segcorr.py (bitwise equal at default arguments, asserted in run()); adds feature sources, strength, gate and fit mask."""
import time
from pathlib import Path

import numpy as np

import segcorr

HERE = Path(__file__).resolve().parent
W, N0, MIN_EFF, CLIP, PRIOR = segcorr.W, segcorr.N0, segcorr.MIN_EFF, segcorr.CLIP, segcorr.PRIOR
PEOPLE = [f"P{i:02}" for i in range(1, 11)]
GROUPS = ("raw", "hard", "cbm", "eafb", "sbn", "teacher", "sinc", "fbl")


def dump_members(ctx, archive, outer_mod, proto, stack):
    """Mix member posteriors in dataset-row space (members() is in mixer order: X[perm] = member). Asserts the slot member equals the chain's own raw slot."""
    perm = np.asarray(ctx.Pperm)
    out = {}
    for k, v in ctx.A.members(stack, proto).items():
        v = np.asarray(v, float)
        r = np.empty_like(v)
        r[perm] = v
        out[str(k)] = r
    fam = outer_mod.FAMILY
    base = [ctx.FR17[fam, s] for s in range(6)]
    slot = archive[ctx.SLOT[fam]] + np.mean([b - ctx.FR6[fam, i] for i, b in enumerate(base)], axis=0) + (ctx.CU17[fam, 0] - ctx.FT_BASE[fam, 0])
    np.testing.assert_array_equal(out[outer_mod.SLOT], ctx.norm(slot))
    return out


def correct(raw, labels, x, w=W, gate=None, fit=None):
    """As segcorr.correct (x_hist = x). gate[t] False: window t is not corrected; fit[t] False: window t does not enter the class densities (both default all True)."""
    n, f = x.shape
    out, active = raw.copy(), np.zeros(n, bool)
    cnt, sm, sq = np.zeros(3), np.zeros((3, f)), np.zeros((3, f))
    for t in range(n):
        if (cnt >= MIN_EFF).all() and (gate is None or gate[t]):
            tot = cnt.sum()
            mu0 = sm.sum(0) / tot
            mu = (sm + PRIOR * mu0) / (cnt[:, None] + PRIOR)
            within = (sq - 2 * mu * sm + cnt[:, None] * mu ** 2).sum(0)
            var0 = sq.sum(0) / tot - mu0 ** 2
            v = np.maximum((within + N0 * var0) / (tot + N0), 1e-6)
            ll = -0.5 * (((x[t] - mu) ** 2) / v).sum(1)
            ll = np.clip(ll - ll.mean(), -CLIP, CLIP)
            p = raw[t] * np.exp(w * ll)
            out[t] = p / p.sum()
            active[t] = True
        if fit is None or fit[t]:
            k = labels[t]
            cnt[k] += 1; sm[k] += x[t]; sq[k] += x[t] ** 2
    return out, active


def logc(p):
    z = np.log(np.maximum(p, 1e-6))
    return z - z.mean(1, keepdims=True)


def seg_rows(d, order, person, seed):
    out = np.zeros((len(person), 4, 3))
    for p in PEOPLE:
        z = np.load(d / f"seg_{order}_{seed}_{p}.npz")
        out[z["rows"]] = z["seg"]
    return out


def groups(mem, pupils):
    m = lambda k: mem[k]
    return {"raw": m("sc_fbraw"), "hard": pupils["hard"], "cbm": pupils["cb"], "eafb": m("sc_fb"), "sbn": np.mean([m(k) for k in ("5", "6", "9")], axis=0),
            "teacher": m("10"), "sinc": m("sc_sinc"), "fbl": m("sc_fbl")}


def comp(name, seg, G):
    """One feature block (n x f). seg: dict seed-set -> n x 4 x 3; G: member groups (posteriors) of the base's own member state."""
    kind, _, arg = name.partition(":")
    if kind == "seg":  # seg:SEEDS:VIEW, SEEDS in s0 | s012 (mean of the three stand-in seeds), VIEW whole | disp | dispo | two | half
        s, view = arg.split(":")
        c = seg[s]
        z = c.sum(1); z = z - z.mean(1, keepdims=True)
        if view == "whole": return z
        if view == "disp": return np.concatenate([z, c.std(1)], 1)
        if view == "dispo": return c.std(1)
        if view == "two": return np.concatenate([z, np.abs(c[:, :2].sum(1) - c[:, 2:].sum(1)) / 2], 1)  # two views of half length: SD of 2 views = |a-b|/2
        if view == "half": return np.concatenate([z, (c[:, :2].sum(1) - c[:, 2:].sum(1))], 1)  # signed first-half minus second-half drift
    if kind == "m":  # m:GROUP | m:all (all groups, concatenated)
        return np.concatenate([logc(G[g]) for g in (GROUPS if arg == "all" else (arg,))], 1)
    if kind == "md":  # dispersion across the member groups (class-wise SD of the centred log-posteriors)
        return np.stack([logc(G[g]) for g in GROUPS]).std(0)
    raise KeyError(name)


def spec(sel):
    """name -> (feature blocks, w, gate margin or None, fit-on-gated)"""
    a = {"s0disp": (("seg:s0:disp",), W), "s0whole": (("seg:s0:whole",), W), "s0dispo": (("seg:s0:dispo",), W), "s3disp": (("seg:s012:disp",), W),
         "s3whole": (("seg:s012:whole",), W)}
    a.update({f"m_{g}": ((f"m:{g}",), W) for g in ("raw", "hard", "eafb", "sbn", "teacher", "sinc")})
    b = {"m_all": (("m:all",), W), "m_all_s3": (("m:all", "seg:s012:disp"), W), "md": (("md",), W), "m_all_md": (("m:all", "md"), W),
         "s0two": (("seg:s0:two",), W), "s0half": (("seg:s0:half",), W), "s3two": (("seg:s012:two",), W)}
    for w in (0.05, 0.2, 0.3):
        b[f"s3disp_w{int(w * 100):02}"] = (("seg:s012:disp",), w)
        b[f"m_all_w{int(w * 100):02}"] = (("m:all",), w)
    return {k: v + (None, False) for k, v in (a if sel == "a" else b).items() if sel in "ab"} | (
        {f"s3disp_{m}{int(t * 100):02}": (("seg:s012:disp",), 0.2, t, m == "f") for m in "gf" for t in (0.3, 0.6)} if sel == "b" else {})


BASES = {"hc": "hardcb", "da": "hcF_l5"}


def arms(sel):
    return {f"{b}_{k}": None for k in (spec(sel) if sel in "ab" else {}) for b in BASES}


def run(order, ctx, rows, person, pupils, sel, seg0, seg1):
    mem = ctx["mem"]
    if sel == "dump":
        np.savez(HERE / f"members_{order}.npz", **{f"{s}|{k}": v for s in mem for k, v in mem[s].items()}, hard=pupils["hard"], cb=pupils["cb"])
        print({k: v.shape for k, v in mem["stock"].items()}, flush=True)
        return {}
    raw = ctx["raw"]
    labels = raw["stock"].argmax(1)
    deliv = ctx["delivery"]
    sess = list(rows.values())
    seg = {"s0": seg_rows(seg0, order, person, 0)}
    seg["s012"] = (seg["s0"] + seg_rows(seg1, order, person, 1) + seg_rows(seg1, order, person, 2)) / 3
    for p in PEOPLE:
        np.testing.assert_array_equal(np.load(seg0 / f"seg_{order}_0_{p}.npz")["delivery"], deliv[person[deliv] == p])
    info = {}
    for b, base in BASES.items():
        G = groups(mem["l5" if b == "da" else "stock"], pupils)
        marg = np.sort(raw[base], 1)
        marg = marg[:, -1] - marg[:, -2]
        for k, (blocks, w, thr, fitg) in spec(sel).items():
            x = np.concatenate([comp(c, seg, G) for c in blocks], 1)
            gate = None if thr is None else marg < thr
            out, act, t0 = raw[base].copy(), 0, time.perf_counter()
            for ix in sess:
                o, a = correct(raw[base][ix], labels[ix], x[ix], w, None if gate is None else gate[ix], gate[ix] if fitg else None)
                if k == "s0disp":  # zero-change guard: the lane P form is reproduced bitwise
                    o2, a2 = segcorr.correct(raw[base][ix], labels[ix], x[ix])
                    np.testing.assert_array_equal(o, o2); np.testing.assert_array_equal(a, a2)
                out[ix] = o; act += int(a.sum())
            name = f"{b}_{k}"
            raw[name] = out
            info[name] = dict(active=act, changed=int((out.argmax(1) != raw[base].argmax(1)).sum()), dim=x.shape[1], us_per_window=1e6 * (time.perf_counter() - t0) / len(out))
            assert info[name]["changed"] > 0 and act > 0, name
    return info
