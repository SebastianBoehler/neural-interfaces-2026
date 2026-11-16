"""§281 diagnostic: trace the v5.1b identity/contrast routing on new people.

Usage: python -I studies/14nov/newmember/s281_routing.py ORDER
Reports decisions only, not treatment accuracy. A nonzero routing ramp refutes
the unrecognised-session assumption used by the §281 personalisation study.
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


def verify_archive_file(path, manifest):
    found = [line.split(maxsplit=1)[0] for line in manifest.read_text().splitlines()
             if len(line.split(maxsplit=1)) == 2 and line.split(maxsplit=1)[1].strip() == path.name]
    assert len(found) == 1, f"archive missing or duplicated in manifest: {path.name}"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == found[0], f"archive SHA mismatch: {path.name}"
    return digest


def check_trace(sessions, expected_rows, session_ids):
    rows = np.concatenate([s["rows"] for s in sessions])
    assert np.array_equal(np.sort(rows), np.sort(expected_rows)), "routing sessions do not partition the test windows"
    expected = set(session_ids[expected_rows])
    observed, routed, ncontrast, nbio = set(), [], 0, 0
    for session in sessions:
        ix, contrast, bio = (np.asarray(session[k]) for k in ("rows", "contrast", "bio"))
        assert len(ix) and contrast.shape == bio.shape == ix.shape
        assert np.isfinite(contrast).all() and np.isfinite(bio).all()
        assert ((0 <= contrast) & (contrast <= 1) & (0 <= bio) & (bio <= 1)).all()
        s = np.unique(session_ids[ix]); assert len(s) == 1 and s[0] not in observed, "invalid or duplicate session"
        observed.add(s[0]); positive = (contrast > 0) | (bio > 0)
        ncontrast += int((contrast > 0).sum()); nbio += int((bio > 0).sum())
        if positive.any():
            routed.append(dict(session=int(s[0]), first=int(np.flatnonzero(positive)[0])))
    assert observed == expected, "missing routing session"
    return dict(sessions=len(observed), routed_sessions=len(routed), contrast_windows=ncontrast,
                bio_windows=nbio, routed=routed)


def load_routing():
    for key in ("RT_PROTO", "GATE_V51B", "RT_MEMBERS", "FT_SFX", "TAU", "PSC", "G3", "KAPPA", "HALF"):
        assert key not in os.environ, f"unexpected environment override: {key}"
    os.environ.update(RT_PROTO="all_loso", GATE_V51B="1", RT_MEMBERS="1", FT_SFX="LPC")
    path = ROOT / "studies/13nov/retv6.py"
    source = path.read_text().split("\nOOF = {bn: oof_rec(bn)")[0]
    assert "def member_deltas(order):" in source
    old = 'f"{REPO}/runs/ttt/probs18/ttt18{order}{FT_SFX}"'
    new = 'f"{REPO}/runs/diag/ttt/probs18/ttt18{order}{FT_SFX}"'
    assert source.count(old) == 2, "unexpected retv6 input paths"
    source = source.replace(old, new)
    scope = {"__name__": "retv6_lib", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), scope)
    return scope


def run_order(order):
    assert order in ORDERS, f"unsupported order {order}"
    module = load_routing()
    K, S, A, ns, proto = (module[k] for k in ("K", "S", "A", "ns", "p"))
    assert proto == "all_loso" and module["GATE_V51B"] and module["RT_MEMBERS"]
    expected = module["IDX"]
    assert np.array_equal(expected, np.arange(len(expected))), "unexpected global row layout"
    original_sess = S.ORIG[[i for i, d in enumerate(S.DICTS) if d is A.pl.B][0]][f"sess|{proto}"].copy()
    archive = ROOT / f"runs/s281_v66_reference/s242c250c_{order}.npz"
    assert archive.is_file(), f"missing §257 control: {archive}"
    archive_sha = verify_archive_file(archive, archive.parent / "sha256.remote.txt")
    ref = np.load(archive)
    oof = {bn: module["oof_rec"](bn) for bn in BASES}
    P = module["perm"](order); Pp = P[proto]
    for cache in (ns["_BIO"], A._H, K.STREAMS, K.REC):
        cache.clear()
    S.apply(P)
    K.RANK["cur"] = {q: K._rank(K.IDX[q][P[q]]) for q in K.PR}
    module["apply_members"](module["member_deltas"](order), Pp)
    raw_rmap, raw_bio = A.R3.rmap, ns["bio_stream"]
    pending, traces = [], []

    def rmap(*args):
        result = raw_rmap(*args); pending.append(np.array(result[0], copy=True)); return result

    def bio(p, fold, session, rows, names, t0):
        result = raw_bio(p, fold, session, rows, names, t0)
        assert p == proto and pending, "contrast trace missing for bio session"
        traces.append(dict(name=session, rows=np.asarray(rows, int).copy(), contrast=pending.pop(0),
                           bio=np.array(result[0], copy=True)))
        return result

    A.R3.rmap, ns["bio_stream"] = rmap, bio
    old_folds = A.RR.FOLDS
    result = {}
    try:
        A.RR.FOLDS = {proto: old_folds[proto]}
        for bn in BASES:
            traces.clear(); pending.clear(); K.REC.clear()
            mix, a = A.mix(bn, proto, list(A.FAMS)); _, shadow = A.shadow(mix, proto)
            key = f"{bn}|P0r"
            assert key in ref.files and np.array_equal(shadow[np.argsort(Pp)].astype(np.float32), ref[key]), f"{key}: shadow differs from archived control"
            ns["route4"]({proto: mix}, {proto: shadow}, a, oof[bn],
                         {proto: A.members(bn, proto)["c1"]}, "max", 10)
            assert not pending, "unmatched contrast trace"
            summary = check_trace(traces, expected, original_sess)
            routed_names = {(proto, t["name"]) for t in traces if (t["contrast"] > 0).any() or (t["bio"] > 0).any()}
            assert set(K.REC) == routed_names, "actual routed sessions disagree with gate traces"
            result[bn] = summary
            print(order, bn, json.dumps(summary), flush=True)
    finally:
        A.RR.FOLDS = old_folds; A.R3.rmap = raw_rmap; ns["bio_stream"] = raw_bio
    return dict(order=order, archive_sha256=archive_sha, bases=result)


def main():
    assert len(sys.argv) == 2, "supply one delivery order"
    order = sys.argv[1]
    out = ROOT / f"runs/s281_v66_route/{order}.json"
    assert not out.exists(), f"refusing to overwrite {out}"
    result = run_order(order)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    assert all(v["routed_sessions"] == 0 for v in result["bases"].values()), f"new-person routing found; see {out}"
    print(f"routing trace OK: {out}", flush=True)


if __name__ == "__main__":
    main()
