"""Lane T2 sweep design. Config ids are stable hashes of the cfg dict; stage 1 = reference arms + centre + one-factor-at-a-time
+ a balanced (Latin-style) random design: every level of every factor appears equally often."""
import hashlib
import json
import numpy as np

LEVELS = dict(steps=[4, 8, 16, 32], lr=[3e-4, 1e-3, 3e-3], every=[5, 10, 20], target=["sink", "plain", "hard"],
              aug=["weak", "strong"], lam_im=[0.0, 0.5, 1.0], params=["bn", "last", "all"], anchor=[1e-2, 0.0])
CENTRE = dict(steps=8, lr=3e-4, every=10, target="sink", aug="strong", lam_im=0.5, params="all", anchor=1e-2)
FIXED = dict(bn="prefix", dense_until=16, n0=16, balance=0)   # dense_until = n0: stride is the only schedule knob
REFS = {  # team C's own variants verbatim (their schedule: every 4, dense to 48), same code path
    "A0": dict(bn="source", steps=0, balance=0), "A1": dict(bn="prefix", steps=0, balance=0),
    "A2": dict(bn="prefix", steps=4, balance=0), "A2h": dict(bn="prefix", steps=8, lr=3e-4, balance=0)}


def cid(cfg):
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def full(factors):
    return {**FIXED, **factors}


def stage1(n_random=288, seed=20261120):
    out = {}
    for k, v in REFS.items():
        out[k] = dict(v)
    out["T0_centre"] = full(CENTRE)
    for f, lv in LEVELS.items():
        for l in lv:
            if l != CENTRE[f]:
                out[f"T0_{f}={l}"] = full({**CENTRE, f: l})
    rng = np.random.default_rng(seed)
    cols = {f: np.array([lv[i % len(lv)] for i in range(n_random)], dtype=object) for f, lv in LEVELS.items()}
    for f in cols:
        cols[f] = cols[f][rng.permutation(n_random)]
    for i in range(n_random):
        cfg = full({f: (cols[f][i].item() if hasattr(cols[f][i], "item") else cols[f][i]) for f in LEVELS})
        out.setdefault(f"R{i:03d}", cfg)
    seen, res = {}, {}
    for name, cfg in out.items():  # drop exact duplicates, keep the first name
        h = cid(cfg)
        if h not in seen:
            seen[h] = name
            res[name] = cfg
    return res


if __name__ == "__main__":
    s = stage1()
    print(len(s), "configs")
    for f in LEVELS:
        print(f, {l: sum(1 for c in s.values() if c.get(f) == l) for l in LEVELS[f]})
