"""s235: nested selection over people for shufcombo knob sweeps (v6.2hL-safe, new people, V6Fr).
Each argument is one group of arms: 'K=v,v,...' or a Cartesian product 'K=v,v/K2=v,v'; the arms are the union over the
groups (the shipped setting is always an arm, gain 0). Gain of an arm per order and person = V6Fr (mean over the file's
bases) minus V6Fr of the same order's file without knobs. For each of the 126 splits of the ten people into halves of
five and both directions: choose the arm with the best mean gain over all orders on the selection half, score it on the
other half (per scheme and overall); a split's value is the mean of its two directions.
Usage: python studies/14nov/sweeps/nested.py DIR GROUP [GROUP ...]   (DIR holds the shufcombo_*.npz files)
       ORDERS=within0,within1,full0,full1,runs0,runs1 (default) | TAG=LF (FT tag suffix, default LF)"""
import glob, itertools, os, sys
import numpy as np

DEF = {"EYEW": 1.0, "REST_W": 1.0, "EM_LAM": 0.5, "TS_LAM": 0.15, "BTS": 4.0}
ORDERS = os.environ.get("ORDERS", "within0,within1,full0,full1,runs0,runs1").split(",")
TAG = os.environ.get("TAG", "LF")


def suffix(arm):
    """the knob part of shufcombo's output name; knobs at their shipped value are left unset (s235 check: same output)."""
    s = "".join(f"_{n}{float(arm[k]):g}" for k, n in (("EYEW", "eyew"), ("REST_W", "restw")) if k in arm and float(arm[k]) != DEF[k])
    return s + "".join(f"_{k.lower()}{arm[k]}" for k in ("EM_LAM", "TS_LAM", "BTS") if k in arm and float(arm[k]) != DEF[k])


def load(d, o, arm):
    hit = [f for f in glob.glob(f"{d}/shufcombo_{o}_*ttt18{o}{TAG}{suffix(arm)}.npz") if f.endswith(f"ttt18{o}{TAG}{suffix(arm)}.npz")]
    assert len(hit) == 1, (o, arm, hit)
    z = np.load(hit[0]); bases = sorted({k.split("|")[0] for k in z.files}); assert len(bases) == 6, (hit[0], bases)
    return np.mean([z[f"{b}|V6Fr"] for b in bases], 0)


def arms_of(groups):
    out = [{}]
    for g in groups:
        parts = [(k, v.split(",")) for k, v in (p.split("=") for p in g.split("/"))]
        assert all(k in DEF for k, _ in parts), g
        for combo in itertools.product(*[v for _, v in parts]):
            a = {k: v for (k, _), v in zip(parts, combo) if float(v) != DEF[k]}
            if a not in out:
                out.append(a)
    return out


def main():
    d, groups = sys.argv[1], sys.argv[2:]; arms = arms_of(groups)
    G = np.array([[load(d, o, a) - load(d, o, {}) for o in ORDERS] for a in arms])  # arms x orders x people
    sch = {s: [i for i, o in enumerate(ORDERS) if o.rstrip("0123456789") == s] for s in dict.fromkeys(o.rstrip("0123456789") for o in ORDERS)}
    name = lambda a: ",".join(f"{k}={v}" for k, v in a.items()) or "shipped"
    print(f"groups {' | '.join(groups)}; {len(arms)} arms; orders {','.join(ORDERS)}; gain = V6Fr(arm) - V6Fr(no knobs), new people")
    print("full data (optimistic): arm | all | " + " | ".join(sch) + " | without P10")
    for a, g in zip(arms, G):
        print(f"  {name(a):28s} {g.mean():+.2f} | " + " | ".join(f"{g[ix].mean():+.2f}" for ix in sch.values()) + f" | {np.delete(g, 9, 1).mean():+.2f}")
    sel = G.mean(1)  # arms x people, mean over all orders
    held, picks = {s: [] for s in ["all", *sch]}, []
    for S in itertools.combinations(range(10), 5):
        if 0 not in S:
            continue  # each unordered split once (126)
        H = tuple(q for q in range(10) if q not in S); vals = {s: [] for s in held}
        for a_, b_ in ((S, H), (H, S)):
            k = int(np.argmax(sel[:, list(a_)].mean(1))); picks.append(k)
            vals["all"].append(G[k][:, list(b_)].mean())
            for s, ix in sch.items():
                vals[s].append(G[k][ix][:, list(b_)].mean())
        for s in held:
            held[s].append(np.mean(vals[s]))
    print(f"nested held-out over {len(held['all'])} splits (mean [10 %, 90 %] across splits):")
    for s, v in held.items():
        print(f"  {s:8s} {np.mean(v):+.3f} [{np.percentile(v, 10):+.2f}, {np.percentile(v, 90):+.2f}]")
    cnt = np.bincount(picks, minlength=len(arms))
    print("chosen: " + ", ".join(f"{name(arms[k])} {cnt[k]}" for k in np.argsort(-cnt) if cnt[k]))


if __name__ == "__main__":
    main()
