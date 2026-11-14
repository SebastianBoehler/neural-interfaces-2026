"""s266 part (a) scorer: recognition per person (gate266.npz), real-gate routed path minus tail0 (s266_a npz).
Usage: python score266a.py runs/pview/gate266.npz runs/pview/s266_a_<orders>.npz"""
import sys
import numpy as np
g = np.load(sys.argv[1]); z = np.load(sys.argv[2]); PEOPLE = [f"P{i:02d}" for i in range(1, 11)]; OTH = [i for i, q in enumerate(PEOPLE) if q not in ("P04", "P10")]
ORDERS = sorted({k.split("|")[0] for k in z.files}); BASES = sorted({k.split("|")[1] for k in z.files}); ARMS = sorted({k.split("|")[2] for k in z.files})
own = sorted({tuple(k.split("|")[1:3]) for k in g.files if k.startswith("own|") and k.endswith("|j")})
SESS = sorted({s for _, s in own}, key=lambda x: int(x)); PER_OF = {}
for o, s in own:  # a session's person: the last gallery entry
    PER_OF[s] = str(g[f"own|{o}|{s}|names"][-1])
print(f"s266 (a): the package's bio gate (tau 0.26, ramp 0.05, t0 10) on the within-session proxy; {len(ORDERS)} orders x {len(BASES)} bases (scores), {len(own) // len(SESS)} orders (gate)")
print("\nrecognition, own sessions (gallery = nine others + the session's acquisition rows; online rows delivered in order):")
rec = {}
for q in PEOPLE:
    ss = [s for s in SESS if PER_OF[s] == q]; first, share, mr, wrong, miss = [], [], [], 0, 0
    for s in ss:
        for o in sorted({o for o, _ in own}):
            j, r, nm = g[f"own|{o}|{s}|j"], g[f"own|{o}|{s}|r"], g[f"own|{o}|{s}|names"]
            ok = (j == len(nm) - 1); rr = np.where(ok, r, 0.0); wrong += int(((j >= 0) & ~ok & (r > 0)).sum())
            k = np.flatnonzero(rr > 0); first.append(k[0] if len(k) else np.nan); share.append((rr > 0).mean()); mr.append(rr.mean()); miss += int(len(k) == 0)
    rec[q] = (miss == 0, np.nanmedian(first) if miss < len(first) else np.nan)
    print(f"  {q} ({len(ss)} session(s) x orders): recognised {'yes' if miss == 0 else f'NO ({miss}/{len(first)} streams never routed)'}; first routed online window median {np.nanmedian(first):.0f} "
          f"(max {np.nanmax(first):.0f}); routed share of online rows {np.mean(share):.2f}; mean ramp {np.mean(mr):.2f}; windows routed to a wrong person {wrong}")
print(f"  recognised in all streams: {sum(v[0] for v in rec.values())}/10; among the eight (non-P04/P10) {sum(rec[PEOPLE[i]][0] for i in OTH)}/8; "
      f"median time over the eight {np.nanmedian([rec[PEOPLE[i]][1] for i in OTH]):.0f} online windows")
sc = np.concatenate([g[f"own|{o}|{s}|s"] for o, s in own]); print(f"  own-session top score after t0: min {np.nanmin(sc):.2f}, median {np.nanmedian(sc):.2f} (tau 0.26 + ramp 0.05 = 0.31 is r = 1)")
print("\nrecognition, cross-session (P04, P10: gallery = the nine others + all rows of the person's other sessions, as the package holds them):")
for q in ("P04", "P10"):
    first, share, mr, wrong = [], [], [], 0
    for o in sorted({o for o, _ in own}):
        for s in [s for s in SESS if PER_OF[s] == q]:
            j, r, nm = g[f"cross|{o}|{s}|j"], g[f"cross|{o}|{s}|r"], g[f"own|{o}|{s}|names"]; ok = (j == len(nm) - 1); rr = np.where(ok, r, 0.0); wrong += int(((j >= 0) & ~ok & (r > 0)).sum())
            k = np.flatnonzero(rr > 0); first.append(k[0] if len(k) else np.nan); share.append((rr > 0).mean()); mr.append(rr.mean())
    print(f"  {q}: first routed window median {np.nanmedian(first):.0f}, never routed {int(np.isnan(first).sum())}/{len(first)}; share {np.mean(share):.2f}; mean ramp {np.mean(mr):.2f}; wrong-person windows {wrong}")
print("\nfalse routes (every session of another person, all rows delivered, against a gallery holding q's acquisition rows of one session; shipped tau / ramp):")
fk = sorted({k for k in g.files if k.startswith("false|") and k.endswith("|r")}); tot = {q: [0, 0, 0, 0.0] for q in PEOPLE}; ex = []
for k in fk:
    _, o, sq, sp, _ = k.split("|"); q = PER_OF[sq]; r, j, nm = g[k], g[k[:-1] + "j"], g[k[:-1] + "names"]; hit = (j == len(nm) - 1) & (r > 0)
    t = tot[q]; t[0] += 1; t[1] += int(hit.any()); t[2] += int(hit.sum()); t[3] += len(r)
    if hit.any():
        ex.append((o, sq, q, sp, PER_OF[sp], int(hit.sum()), float(r[hit].mean())))
for q in PEOPLE:
    t = tot[q]; print(f"  to {q}: {t[1]}/{t[0]} streams routed at all, {t[2]} windows of {int(t[3])} ({100 * t[2] / t[3]:.2f} %)")
print(f"  all: {sum(t[1] for t in tot.values())}/{sum(t[0] for t in tot.values())} streams, {sum(t[2] for t in tot.values())} windows of {int(sum(t[3] for t in tot.values()))}")
byp = {}
for o, sq, q, sp, p, n, mr_ in ex:
    byp.setdefault((q, p), []).append((o, n, mr_))
for (q, p), v in sorted(byp.items()):
    print(f"    {p}-session routed to {q}: {len(v)} stream(s) (orders {', '.join(x[0] for x in v)}), windows {sum(x[1] for x in v)}, mean ramp {np.mean([x[2] for x in v]):.2f}")
M = {a: np.array([[z[f"{o}|{b}|{a}"] for b in BASES] for o in ORDERS]).mean(1) for a in ARMS}; rng = np.random.default_rng(0)
print(f"\nrouted minus tail0, six-base mean, mean over orders [80 % over orders]; tail0 mean {M['tail0'].mean():.2f}")
for a in [a for a in ARMS if a != "tail0"]:
    d = M[a] - M["tail0"]; m = d.mean(0)
    oo = np.array([np.median(d[rng.integers(0, len(ORDERS), len(ORDERS))][:, OTH].mean(0)) for _ in range(4000)]); pp = np.array([np.median(d[:, rng.choice(OTH, 8)].mean(0)) for _ in range(4000)])
    print(f"{a:10s} mean ten {m.mean():+.2f}; median of the eight {np.median(m[OTH]):+.2f} [{np.percentile(oo, 10):+.2f}, {np.percentile(oo, 90):+.2f} orders; {np.percentile(pp, 10):+.2f}, {np.percentile(pp, 90):+.2f} people]; "
          f">= +2: {(m[OTH] >= 2).sum()}/8 | " + " ".join(f"{q} {v:+.1f}" for q, v in zip(PEOPLE, m)))
for n in ("ER", "ERT"):
    if f"{n}_gate" in ARMS:
        print(f"gate minus oracle (r = 1 from window 0), {n}: median of the eight {np.median((M[n + '_gate'] - M[n + '_ora0']).mean(0)[OTH]):+.2f}; the t0 = 10 part {np.median((M[n + '_ora10'] - M[n + '_ora0']).mean(0)[OTH]):+.2f}, "
              f"recognition/ramp part {np.median((M[n + '_gate'] - M[n + '_ora10']).mean(0)[OTH]):+.2f}")
