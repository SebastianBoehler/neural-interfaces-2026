"""Revisit (Opus, 21 Nov): clean-slate posterior sources as the mixer input of cap250's real chain (lane Z machinery, stock system only).

usage: REV_SRC=src_ORDER.npz final_r.py ORDER (smoke|complete)
Arm spec src=(name, w): after cap250's mixer, m = (1 - w) * mix + w * source (w = 1: the mix is replaced, the source is the sole
posterior input of rebal -> SessionEM -> TSEM -> count filter -> CSP/LDA). Everything else is lane Z's corrected chain: the GEB
gate, FT banks and the SessionEM/TSEM shadow stay the frozen incumbent ones (replay design), so with w = 1 cap250 still reaches
the output through the gate decision and the shadow, never through the mix. SessionEM/TSEM features and CSP are the stages'
own (signal) features. Sources are recording-layout (2655, 3) posteriors streamed causally under the SAME delivery order
(asserted: REV_SRC 'delivery' equals the chain delivery). Zero-change guards: stock|base equals the released raw/filtered/
terminal bitwise; an arm with w = 0 equals stock|base bitwise (smoke)."""
import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from bridge_orders import write_npz, write_receipt
from bridge_owner import pin, sha, start_module, stamp
from complete_zero import FEAT, HASHES
from matrix22 import parents
from parents22 import C20

import final_z as Z  # lane Z module (session, released_control, tail); its run_chain is not used

WORK = Path(__file__).resolve().parent
WEIGHTS = {"rep": 1.0, "h50": 0.5, "h25": 0.25}


def arm_table(names, smoke):
    out = {"stock|base": dict(system="stock")}
    for n in names:
        for tag, w in WEIGHTS.items():
            out[f"{n}|{tag}"] = dict(system="stock", src=(n, w))
    if smoke:
        n = names[0]
        out = {"stock|base": out["stock|base"], f"{n}|h0": dict(system="stock", src=(n, 0.0)), f"{n}|rep": out[f"{n}|rep"], f"{n}|h50": out[f"{n}|h50"]}
    return out


def run_chain(ctx, archive, patches, incumbent, spec, sources, y):
    """one pass of lane Z's corrected chain (stock members, FT on, corrected slots applied) with cap250's mixer output
    blended by spec['src']; returns (raw tsem output, taps) in recording layout. The blend wraps ctx.A.mix, which
    Z.run_chain then calls as the real mixer."""
    perm, inv = ctx.Pperm, np.argsort(ctx.Pperm)
    orig, cap = ctx.A.mix, {}

    def blended(bn, proto, families, extra):
        out = orig(bn, proto, families, extra)
        m = out[0]
        cap["cap_mix"] = m[inv]
        if "src" in spec:
            name, w = spec["src"]
            s = sources[name][perm]
            m = s.copy() if w == 1.0 else (m if w == 0.0 else (1 - w) * m + w * s)
        return m, out[1]

    ctx.A.mix = blended
    try:
        raw, taps = Z.run_chain(ctx, archive, patches, incumbent, dict(system="stock"), None, y, None, None)
    finally:
        ctx.A.mix = orig
    taps["cap_mix"] = cap["cap_mix"]
    return raw, taps


def run(order, smoke):
    begin = perf_counter()
    source = sha(WORK / "source.sha256")
    pin(FEAT, HASHES[FEAT])
    with np.load(FEAT, allow_pickle=False) as z:
        person = np.asarray([p.rsplit("/", 1)[-1] for p in z["subject"]])
        visits, y = z["session"], z["y"].astype(int)
    assert len(set(person)) == 10 and len(person) == 2655
    parents(order)
    tm = start_module("c22_parent_tail", C20 / "worker.py", (C20 / "worker.py").read_text())
    src_path = Path(os.environ["REV_SRC"])
    with np.load(src_path, allow_pickle=False) as z:
        assert str(z["order"]) == order
        src_delivery = z["delivery"]
        names = [str(n) for n in z["names"]]
        sources = {n: z[n].astype(np.float64) for n in names}
    for n, s in sources.items():
        assert s.shape == (2655, 3) and np.isfinite(s).all(), n
        np.testing.assert_allclose(s.sum(1), 1.0, atol=1e-4, rtol=0)
    arms = arm_table(names, smoke)
    out, times = {}, {}

    def runner(ctx, archive, patches, incumbent, stock):
        np.testing.assert_array_equal(ctx.Pperm, src_delivery)  # sources were streamed in the chain's delivery order
        for name, spec in arms.items():
            t0 = perf_counter()
            raw, taps = run_chain(ctx, archive, patches, incumbent, spec, sources, y)
            out[name] = dict(raw=raw, taps=taps, spec=spec)
            times[name] = perf_counter() - t0
            print("chain", order, name, round(times[name], 1), flush=True)
        np.testing.assert_array_equal(out["stock|base"]["raw"], stock)
        for name in out:
            if name.endswith("|h0"):
                np.testing.assert_array_equal(out[name]["raw"], stock)

    control0 = Z.released_control(order)
    with Z.session(order, runner) as (ctx, control):
        for k in ("raw", "filtered", "terminal"):
            np.testing.assert_array_equal(control[k], control0[k])
        np.testing.assert_array_equal(ctx["delivery"], src_delivery)
        count = ctx["count"]
        sessions = tuple(count.SESS)
        rows = {s: count.C.order_rows(order, s, count.PROTO) for s in sessions}
        p01 = next(s for s, r in rows.items() if set(person[r]) == {"P01"} and set(visits[r]) == {"1"})
        selected = sessions if not smoke else (p01,)
        delivered = np.concatenate([rows[s] for s in selected])
        arrays = dict(index=np.arange(len(delivered)), delivery=delivered, y=y)
        info = {}
        for name, o in out.items():
            t0 = perf_counter()
            filtered, terminal = Z.tail(ctx, order, o["raw"], selected, set(), tm)
            info[name] = dict(chain_s=times[name], tail_s=perf_counter() - t0)
            for k in ("cap_mix", "mix", "rebal", "em", "tsem"):
                arrays[f"{name}|{k}"] = np.asarray(o["taps"][k], dtype=np.float32)
            arrays[f"{name}|filtered"], arrays[f"{name}|terminal"] = filtered.astype(np.float32), terminal.astype(np.float32)
            if name == "stock|base" and not smoke:
                np.testing.assert_array_equal(filtered[delivered], control["filtered"][delivered])
                np.testing.assert_array_equal(terminal[delivered], control["terminal"][delivered])
            if name.endswith("|h0"):  # arm_table puts stock|base first
                np.testing.assert_array_equal(terminal.astype(np.float32)[delivered], arrays["stock|base|terminal"][delivered])
            print("tail", order, name, round(info[name]["tail_s"], 1), flush=True)
        proof = dict(passed=True, scope="smoke" if smoke else "complete", order=order, source=source, rev_src=sha(src_path),
                     delivery=stamp(ctx["delivery"]), rows=len(delivered), arms=info, specs={n: str(o["spec"]) for n, o in out.items()},
                     held_labels_opened=False, recognition_replayed=False, seconds=perf_counter() - begin)
        if not smoke:
            proof["output"] = write_npz(WORK / f"{order}_rev.npz", arrays)
        path = WORK / (f"smoke_rev_{order}.json" if smoke else f"{order}_rev.json")
        write_receipt(path, **proof)
    print("PASS revisit", order, proof["scope"], sha(path), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("order", choices=("full0", "within0"))
    p.add_argument("mode", choices=("smoke", "complete"))
    a = p.parse_args()
    run(a.order, a.mode == "smoke")
