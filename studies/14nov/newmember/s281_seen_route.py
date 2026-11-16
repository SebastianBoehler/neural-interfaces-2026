"""Trace v5.1b recognition for §281's seen-session fine-tuning inputs.

Usage: python -I studies/14nov/newmember/s281_seen_route.py ORDER
The result is a fold-specific dev trace, not a packaged-worker verification.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASES = ("B2_pool", "B2_s345", "B2_s012", "B2_s0", "B2_s1", "B2_s2")
ORDERS = ("within0", "within1", "within2", "within3", "full0", "full1", "runs0", "runs1")


def latch_traces(traces, rows):
    flat = np.concatenate([np.asarray(t["rows"]) for t in traces])
    assert np.array_equal(np.sort(flat), np.sort(rows)), "routing sessions do not partition the rows"
    lookup = {int(g): i for i, g in enumerate(rows)}
    out = np.zeros(len(rows), dtype=bool)
    for t in traces:
        ix, contrast, bio = (np.asarray(t[k]) for k in ("rows", "contrast", "bio"))
        assert len(ix) and contrast.shape == bio.shape == ix.shape
        assert np.isfinite(contrast).all() and np.isfinite(bio).all(), "nonfinite routing ramp"
        assert ((0 <= contrast) & (contrast <= 1) & (0 <= bio) & (bio <= 1)).all()
        out[[lookup[int(g)] for g in ix]] = np.maximum.accumulate((contrast > 0) | (bio > 0))
    return out


def load_routing():
    for key in ("RT_PROTO", "GATE_V51B", "RT_MEMBERS", "FT_SFX", "TAU", "PSC", "G3", "KAPPA", "HALF"):
        assert key not in os.environ, f"unexpected routing override: {key}"
    os.environ.update(RT_PROTO="seen_session", GATE_V51B="1", RT_MEMBERS="1", FT_SFX="QF")
    path = ROOT / "studies/13nov/retv6.py"
    source = path.read_text().split("\nOOF = {bn: oof_rec(bn)")[0]
    assert "def member_deltas(order):" in source, "unexpected retv6 source"
    scope = {"__name__": "retv6_lib", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), scope)
    return scope


def run_order(order):
    assert order in ORDERS, f"unsupported order: {order}"
    module = load_routing()
    K, S, A, ns, proto = (module[k] for k in ("K", "S", "A", "ns", "p"))
    assert proto == "seen_session" and module["GATE_V51B"] and module["RT_MEMBERS"]
    rows = module["IDX"]
    ref_path = ROOT / f"runs/chainout/ret_{order}.npz"
    ref = np.load(ref_path)
    oof = {bn: module["oof_rec"](bn) for bn in BASES}
    P = module["perm"](order); Pp = P[proto]
    for cache in (ns["_BIO"], A._H, K.STREAMS, K.REC):
        cache.clear()
    S.apply(P); K.RANK["cur"] = {q: K._rank(K.IDX[q][P[q]]) for q in K.PR}
    module["apply_members"](module["member_deltas"](order), Pp)
    raw_rmap, raw_bio = A.R3.rmap, ns["bio_stream"]
    pending, traces = [], []

    def rmap(*args):
        result = raw_rmap(*args); pending.append(np.array(result[0], copy=True)); return result

    def bio(p, fold, session, ix, names, t0):
        result = raw_bio(p, fold, session, ix, names, t0)
        assert p == proto and pending, "contrast trace missing"
        traces.append(dict(name=session, rows=np.asarray(ix, int).copy(), contrast=pending.pop(0),
                           bio=np.array(result[0], copy=True)))
        return result

    A.R3.rmap, ns["bio_stream"] = rmap, bio
    old_folds = A.RR.FOLDS
    values, summaries = {"index": rows}, {}
    try:
        A.RR.FOLDS = {proto: old_folds[proto]}
        for bn in BASES:
            traces.clear(); pending.clear(); K.REC.clear()
            mix, a = A.mix(bn, proto, list(A.FAMS)); _, shadow = A.shadow(mix, proto)
            out0, _ = ns["route4"]({proto: mix}, {proto: shadow}, a, oof[bn],
                                   {proto: A.members(bn, proto)["c1"]}, "max", 10)
            assert not pending and len(traces) == 4, "expected four complete returning sessions"
            recognised = latch_traces(traces, rows)
            routed_names = {(proto, t["name"]) for t in traces if (t["contrast"] > 0).any() or (t["bio"] > 0).any()}
            assert set(K.REC) == routed_names, "routed sessions disagree with gate traces"
            routed = K.staged(out0)[0][proto][np.argsort(Pp)]
            plain = shadow[np.argsort(Pp)]
            assert np.allclose(routed[~recognised], plain[~recognised], atol=1e-9, rtol=0), "pre-recognition tail differs from shadow"
            key = f"{order}|{bn}|base"
            assert key in ref.files and routed.shape == ref[key].shape, f"missing baseline {key}"
            diff = float(np.max(np.abs(routed - ref[key])))
            changed = int(np.sum(routed.argmax(1) != ref[key].argmax(1)))
            assert diff < 1e-5 and changed == 0, f"{key}: reference mismatch {diff:.3g}, {changed} labels"
            values[f"{bn}|recognised"] = recognised
            values[f"{bn}|shadow"] = plain
            values[f"{bn}|routed_v51b"] = routed
            summaries[bn] = dict(sessions=len(traces), recognised_windows=int(recognised.sum()),
                                 max_reference_difference=diff, changed_labels=changed)
            print(order, bn, json.dumps(summaries[bn]), flush=True)
    finally:
        A.RR.FOLDS = old_folds; A.R3.rmap = raw_rmap; ns["bio_stream"] = raw_bio
    values["reference_sha256"] = np.array(hashlib.sha256(ref_path.read_bytes()).hexdigest())
    values["order"] = np.array(order)
    return values, summaries


def main():
    assert len(sys.argv) == 2, "supply one delivery order"
    order = sys.argv[1]
    out = ROOT / f"runs/s281_seen_route/{order}.npz"
    assert not out.exists(), f"refusing to overwrite {out}"
    values, _ = run_order(order)
    out.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(out, **values)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
