"""Package-form replay of the personal nets (read-only): dev s78 form vs (a) without the extra 'sh' member,
(b) personal nets only in the routed tail on routed windows (shadow from the frozen twins)."""
import sys, os, time
sys.dont_write_bytecode = True
import numpy as np
REPO = "<repo>"
F = REPO + "/studies/12nov/pnet.py"; src = open(F).read().split("\nR = {}")[0].replace("os.path.abspath(__file__ + \"/../../..\")", repr(REPO))
__file__ = F
exec(compile(src, F, "exec"))
BASES = os.environ.get("BASES", "B2_pool,B2_s345,B2_s012,B2_s0,B2_s1,B2_s2").split(",")
MIXP = {}

def slot_mix_pkg(mix, c1, H, r, j):
    key = next(k for k, v in A._H.items() if v is H); REC[key] = (np.asarray(r, float).copy(), np.asarray(j).copy())
    proto, s = key; names, gi, f = NM[key]; idx = A.pl.B[f"idx|{proto}"]; pos = {int(g): k for k, g in enumerate(idx)}; rows = np.array([pos[int(g)] for g in gi])
    m2 = mix.copy(); rt = np.asarray(r) > 0; m2[rt] = MIXP[proto][rows][rt]
    return _sm(m2, c1, H, r, j)

def run_pkg(bn, v, archs):
    """shadow and unrouted windows from the frozen twins; routed windows of the routed tail from the personal nets"""
    global ARCHS
    keep = ARCHS; ARCHS = archs
    try:
        fb, eb = swap("base"); fv, ev = swap(v)
    finally:
        ARCHS = keep
    mixes, sh, a = {}, {}, 0.0; REC.clear()
    for p in A.PROTOS:
        mixes[p], a = A.mix(bn, p, fb, eb); _, sh[p] = A.shadow(mixes[p], p); MIXP[p], _ = A.mix(bn, p, fv, ev)
    R3.slot_mix = slot_mix_pkg
    try:
        out0 = ns["route4"](mixes, sh, a, sh["all_loso"], {p: A.members(bn, p)["c1"] for p in A.PROTOS}, "max", 10)[0]
    finally:
        R3.slot_mix = slot_mix
    rec = {k: (r.copy(), j.copy()) for k, (r, j) in REC.items()}
    return out0, staged(out0)[0], rec

def run_dev(bn, v, archs):
    global ARCHS
    keep = ARCHS; ARCHS = archs
    try:
        return run(bn, v)
    finally:
        ARCHS = keep

ALL = list(ARCHS); NOSH = [a for a in ARCHS if a != "sh"]
res = {}
for bn in BASES:
    t0 = time.time()
    for tag, archs in (("5fam", ALL), ("4fam(no sh)", NOSH)):
        _, ts = run_dev(bn, "base", archs); ref = SC.person(ts["seen_session"], "seen_session")
        _, ps = run_dev(bn, "pfh60", archs); d = SC.person(ps["seen_session"], "seen_session") - ref; res.setdefault(("dev", tag), []).append(d)
        _, pk, rec = run_pkg(bn, "pfh60", archs); dk = SC.person(pk["seen_session"], "seen_session") - ref; res.setdefault(("pkg", tag), []).append(dk)
        print(f"{bn} {tag}: dev-form pfh60 seen {d.mean():+.2f} (P04 {d[0]:+.2f} P10 {d[1]:+.2f}) | package-form {dk.mean():+.2f} (P04 {dk[0]:+.2f} P10 {dk[1]:+.2f})", flush=True)
    if bn == BASES[0]:
        for (proto, s), (r, j) in rec.items():
            print(f"     routed {proto} {s.split('/')[-1]}: first routed window {int(np.flatnonzero(r > 0)[0])}, routed share {(r > 0).mean():.2f}, share with 0<r<1 {((r > 0) & (r < 1)).mean():.2f}, persons {sorted(set(j[r > 0].tolist()))}")
    print(f"   {time.time() - t0:.0f}s", flush=True)
for k, v in res.items():
    D_ = np.array(v); print("MEAN", k, f"seen {D_.mean():+.2f} (P04 {D_[:, 0].mean():+.2f}, P10 {D_[:, 1].mean():+.2f}) per base", " ".join(f"{x.mean():+.2f}" for x in D_))
