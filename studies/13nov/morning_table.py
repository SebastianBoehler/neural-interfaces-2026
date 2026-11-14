"""MORNING TABLE (ledger s202): v6f, v6.1, v6.2-safe and v6.2 on the same orders per shuffle scheme, from scored files only.
New people (all ten, package gate): V6Fr - P0r per person, mean over the six bases.
- v6f: the gC5 run with v5.1b labels (L).
- v6.1: s15 with the stack labels (KF; equal to R within GPU noise, s202), or with the Q labels (QF, s203; v6r on new
  people as well).
- Both come from the same machine (runs/leo_npz for KF, else runs/ for QF), so rest-net seeds are paired.
- v6.2 and v6.2-safe equal v6.1 on new people: no new person is routed (s201(2)).
Returning people (P04, P10; seen_session): retv6 gate-unchanged s15 files (runs/retv6/g1_s15_QF_seen_session_<order>.npz).
- v6.2 = ERF - base, v6.2-safe = ER - base.
- v6f and v6.1 equal v5.1b from the first routed window (window 10, check_v6). Windows 0-9 are not modelled here: 0.
Per scheme: mean, 80 % bootstrap interval over orders, without P10 and P10 alone, orders > 0, people > 0.
Usage: python studies/13nov/morning_table.py"""
import glob, os
import numpy as np
REPO = os.path.abspath(__file__ + "/../../.."); NPZ = REPO + "/runs/leo_npz"; RET = REPO + "/runs/retv6"
rng = np.random.default_rng(0)


def mean_bases(f, arm):
    z = np.load(f); b = sorted({k.split("|")[0] for k in z.files}); assert len(b) == 6, (f, b)
    return np.mean([z[f"{bn}|{arm}"] for bn in b], 0)


def one(pat):
    fs = glob.glob(pat); assert len(fs) <= 1, fs; return fs[0] if fs else None


def ci(x):
    bs = [np.mean(rng.choice(x, len(x))) for _ in range(4000)]; return np.percentile(bs, 10), np.percentile(bs, 90)


new, ret = {}, {}
PAIRS = {}  # order -> (v6f file, v6.1 file) from ONE machine (rest-net seed sets differ between machines, s198)
for d, tag in ((REPO + "/runs", "QF"), (NPZ, "RF"), (NPZ, "KF")):  # s203/s204: local Q; Leonardo R (full 4-7), K; K wins
    for f in sorted(glob.glob(f"{d}/shufcombo_*_ocu_gC5_all_ge_hb_ts_gp0_ttt18*L.npz")):
        o = os.path.basename(f).split("_")[1]; g = one(f"{d}/shufcombo_{o}_ocu_s15_all_ge_hb_ts_gp0_ttt18{o}{tag}.npz")
        if g is not None:
            PAIRS[o] = (f, g)
for o, (f, g) in sorted(PAIRS.items()):
    # P0r takes the no-FT re-stream from each FT run's own base outputs (FR17), so GPU float noise can move it between
    # runs (runs 0: up to 0.09 on one person). Each variant is taken against its own P0r (paired); the gap is reported.
    p0 = mean_bases(f, "P0r"); p0k = mean_bases(g, "P0r"); gap = np.abs(p0 - p0k).max()
    if gap > 1e-9:
        print(f"note {o}: v5.1b differs between the v6f and v6.1 files by up to {gap:.2f} (GPU re-stream of the base)")
    new[o] = {"v6f": mean_bases(f, "V6Fr") - p0, "v6.1": mean_bases(g, "V6Fr") - p0k}
for f in sorted(glob.glob(f"{RET}/g1_s15_QF_seen_session_*.npz")):
    o = os.path.basename(f)[:-4].split("_")[-1]; z = np.load(f); b = sorted({k.split("|")[1] for k in z.files})
    assert len(b) == 6, (f, b)
    d = {k: np.mean([z[f"{o}|{bn}|{k}"] - z[f"{o}|{bn}|base"] for bn in b], 0) for k in ("ER", "ERF")}
    ret[o] = {"v6f": np.zeros(2), "v6.1": np.zeros(2), "v6.2-safe": d["ER"], "v6.2": d["ERF"]}
LF = {}  # s209: v6.2hL's new-people path = s15 on v5.1b labels (tag LF), each file against its own P0r
for o in list(new):
    f = one(f"{REPO}/runs/shufcombo_{o}_ocu_s15_all_ge_hb_ts_gp0_ttt18{o}LF.npz") or one(f"{REPO}/runs/leo_npz_lf/shufcombo_{o}_ocu_s15_all_ge_hb_ts_gp0_ttt18{o}LF.npz")
    if f:
        LF[o] = mean_bases(f, "V6Fr") - mean_bases(f, "P0r")
for o in new:
    new[o]["v6.2-safe"] = new[o]["v6.2"] = new[o]["v6.1"]
    new[o]["v6.2hL"] = LF.get(o, np.full(10, np.nan))
    new[o]["v6f+ER"] = new[o]["v6f+ERF"] = new[o]["v6f"]  # s206: v6f's new-people path (L labels, gC5) + the routed additions
for o in ret:  # returning people: the routed additions do not depend on the new people's label source (FT on routed v5.1b)
    ret[o]["v6f+ER"], ret[o]["v6f+ERF"] = ret[o]["v6.2-safe"], ret[o]["v6.2"]; ret[o]["v6.2hL"] = ret[o]["v6.2"]

VARS = ("v6f", "v6.1", "v6.2-safe", "v6.2", "v6f+ER", "v6f+ERF", "v6.2hL")
print("MORNING TABLE: change over v5.1b in balanced-accuracy points, same orders per row")
for grp, D, np_ in (("new people (10)", new, 10), ("returning (P04, P10)", ret, 2)):
    for sch in ("within", "full", "runs"):
        os_ = sorted((o for o in D if o.startswith(sch)), key=lambda o: int(o[len(sch):]))
        if not os_:
            continue
        print(f"\n{grp}, {sch} {', '.join(o[len(sch):] for o in os_)} ({len(os_)} orders)")
        for v in VARS:
            M = np.array([D[o][v] for o in os_]); m = M.mean(1); lo, hi = ci(m) if len(m) > 1 else (m[0], m[0])
            noP10 = np.delete(M, -1, 1).mean(); P10 = M[:, -1]; lab = "P04 alone" if np_ == 2 else "without P10"
            print(f"  {v:9s} {m.mean():+.2f} [{lo:+.2f}, {hi:+.2f}] | {lab} {noP10:+.2f} | P10 {P10.mean():+.2f} "
                  f"(min {P10.min():+.2f}) | orders > 0 {int((m > 0).sum())}/{len(m)} | people > 0 {int((M.mean(0) > 0).sum())}/{np_}")
# s205 (coordinator): no preferred scheme; scheme-averaged gains under equal weights and under within only
MEAN = {(g, sch, v): np.mean([D[o][v].mean() for o in D if o.startswith(sch)]) for g, D in (("new", new), ("ret", ret))
        for sch in ("within", "full", "runs") for v in VARS if any(o.startswith(sch) for o in D)}
print("\nScheme-averaged (mean of the per-scheme means; J = 0.5 new + 0.5 returning, the dev metric):")
for v in VARS:
    eq = {g: np.mean([MEAN[g, sch, v] for sch in ("within", "full", "runs") if (g, sch, v) in MEAN]) for g in ("new", "ret")}
    wi = {g: MEAN.get((g, "within", v), np.nan) for g in ("new", "ret")}
    print(f"  {v:9s} equal weights: new {eq['new']:+.2f} returning {eq['ret']:+.2f} J {0.5 * (eq['new'] + eq['ret']):+.2f} | "
          f"within only: new {wi['new']:+.2f} returning {wi['ret']:+.2f} J {0.5 * (wi['new'] + wi['ret']):+.2f}")
# s212 (dev-validity review): J weighted like the sealed set, 0.566 new + 0.434 (0.2 P04 + 0.8 P10); 80 % intervals that
# resample people (new people) and orders together; returning people keep their fixed weights (two people cannot be resampled)
WN, WR = 0.566, np.array([0.2, 0.8]); rng2 = np.random.default_rng(1)
print("\nSealed-weighted J (0.566 new + 0.434 (0.2 P04 + 0.8 P10)), 80 % interval over people and orders, per scheme:")
for v in VARS:
    line = f"  {v:9s}"
    for sch in ("within", "full", "runs"):
        on, orr = [o for o in new if o.startswith(sch)], [o for o in ret if o.startswith(sch)]
        if not on or not orr:
            continue
        N, Rr = np.array([new[o][v] for o in on]), np.array([ret[o][v] for o in orr])
        j = WN * N.mean() + (1 - WN) * (Rr.mean(0) @ WR); bs = []
        for _ in range(3000):
            a, b, pp = rng2.integers(0, len(on), len(on)), rng2.integers(0, len(orr), len(orr)), rng2.integers(0, 10, 10)
            bs.append(WN * N[a][:, pp].mean() + (1 - WN) * (Rr[b].mean(0) @ WR))
        line += f" | {sch} {j:+.2f} [{np.percentile(bs, 10):+.2f}, {np.percentile(bs, 90):+.2f}]"
    print(line)
LOSS = {}  # v5.1b's own order loss per person (P0r - P0rec) from the same files as v6f, for "gain against order loss"
for o, (f, g) in sorted(PAIRS.items()):
    LOSS[o] = mean_bases(f, "P0r") - mean_bases(f, "P0rec")
print("\nNew-people gain against v5.1b's order loss (per person, pooled over orders): corr and slope")
for sch in ("within", "full", "runs"):
    os_ = [o for o in new if o.startswith(sch) and o in LOSS]
    if os_:
        L = np.concatenate([LOSS[o] for o in os_])
        for v in ("v6f", "v6.1"):
            G = np.concatenate([new[o][v] for o in os_]); print(f"  {sch:6s} {v:5s}: corr {np.corrcoef(L, G)[0, 1]:+.2f}, slope {np.polyfit(L, G, 1)[0]:+.2f} per point of loss (mean loss {L.mean():+.2f})")
print("\nNot in this table: windows 0-9 of returning sessions (package emits v6.1 there, dev v5.1b's shadow); run time and "
      "worker-image checks (V6_BUILD.md); FT run-to-run spread on the GPU (up to 0.034 in posteriors, 64421057).")
