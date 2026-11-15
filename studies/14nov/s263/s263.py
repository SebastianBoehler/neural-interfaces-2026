"""s263 core feature scaling, core alone (ledger s263). Variants A-D and Z (zero-change: core1's settings through this code).
Core1's own recipe: ts_core2.Feats (CAR, trace shrinkage 0.1, log-Euclidean reference k 2, tangent space, slow block) and
ts_core2.Core with heads lr1 / lr3 / lr10 (scaler + LR), LOSO over the ten people (all_loso). Only the covariance blocks change:
ts_core2.blocks is replaced by blocks_for(settings); every setting is in the cache key, a key without one fails loudly.
Usage: s263.py prep VARIANT | fold VARIANT I | list. Outputs runs/s263/<variant>/ (cache pkl, fold<I>.json/.npz)."""
import hashlib, json, os, pickle, sys, time
import numpy as np
REPO = os.environ.get("NI26_ROOT") or os.path.abspath(__file__ + "/../../../..")
os.environ["NI26_ROOT"] = REPO
sys.path[:0] = [REPO + "/experiments"]
import protocol  # noqa: E402
import ts_core2 as T  # noqa: E402
from features import bandpass  # noqa: E402

HEADS = ["lr1", "lr3", "lr10"]
KEYFIELDS = ("data", "ref", "est", "k", "bands", "segs", "regional", "slow")
ov17 = T.BANDSETS["ov17"]
GEO = [(round(np.sqrt(a[0] * b[0]), 2), round(np.sqrt(a[1] * b[1]), 2)) for a, b in zip(ov17[:-1], ov17[1:])]
BANDS34 = sorted(ov17 + GEO + [(1, 2)])  # ov17 + 16 geometric midpoints + (1, 2): 34 half-step-spaced bands
SEG3 = [(0, 480), (0, 240), (240, 480)]  # full, 0-2 s, 2-4 s (core1 fs2, same order as array_split)
SEG5 = SEG3 + [(120, 360), (60, 420)]  # + 1-3 s, 0.5-3.5 s
BASE = dict(data="proteus_windows.npz", ref="car", est="t0.1", k=2, bands=[list(b) for b in ov17], segs=SEG3, regional=False, slow="1Hz")
VARIANTS = {
    "Z": BASE,
    "A": dict(BASE, bands=[list(b) for b in BANDS34]),
    "B": dict(BASE, segs=SEG5),
    "C": dict(BASE, bands=[list(b) for b in BANDS34], segs=SEG5),
    "D": dict(BASE, regional=True),
}
REGIONS = {"frontal": lambda n: n[0] in "FA", "central": lambda n: n[0] in "C" or n.startswith("TP"),
           "parieto-occipital": lambda n: n[0] == "P" or n.startswith("PO")}
CALLS = [0]


def key(s):
    missing = [f for f in KEYFIELDS if f not in s]
    if missing:
        raise KeyError(f"s263 cache key lacks settings {missing}")
    extra = [f for f in s if f not in KEYFIELDS]
    if extra:
        raise KeyError(f"s263 settings {extra} are not in the cache key")
    return hashlib.sha1(json.dumps({f: s[f] for f in KEYFIELDS}, sort_keys=True).encode()).hexdigest()[:12]


def region_idx(names):
    out = {}
    for r, f in REGIONS.items():
        out[r] = [i for i, n in enumerate(names) if f(str(n))]
    allc = sorted(sum(out.values(), []))
    assert allc == list(range(len(names))), "regions do not partition the channels"
    return out


def seg_blocks(X, s):
    out = []
    for lo, hi in s["bands"]:
        Xb = np.concatenate([bandpass(X[i:i + 256], lo, hi) for i in range(0, len(X), 256)])
        out += [T.estimate(Xb[..., a:b], s["est"]) for a, b in s["segs"]]
    return out


def blocks_for(X, s, names):
    CALLS[0] += 1
    out = seg_blocks(X, s)
    if s["regional"]:  # X is already CAR; subtracting a region's mean = that region's own CAR
        for r, ix in region_idx(names).items():
            Xr = X[:, ix]
            out += seg_blocks((Xr - Xr.mean(1, keepdims=True)).astype(X.dtype), s)
    return out


def build(d, s):
    key(s)
    assert s["ref"] == "car" and s["k"] == 2 and s["est"] == "t0.1" and s["slow"] == "1Hz"
    orig, names = T.blocks, d["ch_names"]
    T.blocks = lambda X, cfg: blocks_for(X, cfg["s263"], names)
    try:
        cfg = T.parse(f"car_ov17_fs2_t{s['est'][1:]}_le_k{s['k']}")
        cfg["s263"] = s
        fe = T.Feats(d, cfg)
    finally:
        T.blocks = orig
    assert CALLS[0] == 1, "patched blocks were not used exactly once"
    return fe


def cache_path(v):
    return f"{REPO}/runs/s263/{v}/feats_{key(VARIANTS[v])}.pkl"


def load_feats(v):
    s = VARIANTS[v]
    with open(cache_path(v), "rb") as f:
        st = pickle.load(f)
    assert st["key"] == key(s) and st["settings"] == json.loads(json.dumps(s)), "cache does not match the variant settings"
    fe = T.Feats.__new__(T.Feats)
    fe.__dict__.update(st["state"])
    return fe


def main():
    cmd = sys.argv[1]
    if cmd == "list":
        for v, s in VARIANTS.items():
            print(v, key(s), len(s["bands"]), len(s["segs"]), s["regional"])
        return
    v = sys.argv[2]
    s = VARIANTS[v]
    os.makedirs(f"{REPO}/runs/s263/{v}", exist_ok=True)
    d = protocol.load()
    if cmd == "prep":
        t0 = time.time()
        fe = build(d, s)
        with open(cache_path(v) + ".tmp", "wb") as f:
            pickle.dump(dict(key=key(s), settings=json.loads(json.dumps(s)), state=fe.__dict__), f, protocol=4)
        os.replace(cache_path(v) + ".tmp", cache_path(v))
        print(f"prep {v} key {key(s)} dim {fe.F.shape[1]} blocks {len(fe.C)} {time.time() - t0:.0f}s", flush=True)
        return
    i = int(sys.argv[3])
    fe = load_feats(v)
    name, tr, te = list(protocol.folds(d, "all_loso"))[i]
    store = {h: np.full((len(d["y"]), 3), np.nan) for h in HEADS}
    t0 = time.time()
    m = T.Core(fe, HEADS, store)
    m.fit(np.flatnonzero(tr))
    fit = time.time() - t0
    protocol.predict_streams(m, d, te, proba=True)
    res = {h: protocol.score(d["y"][te], store[h][te].argmax(1), d, te)["balanced_accuracy_cell_mean"] for h in HEADS}
    json.dump(dict(variant=v, key=key(s), person=name, ba=res, dim=int(fe.F.shape[1]), fit_s=fit, total_s=time.time() - t0),
              open(f"{REPO}/runs/s263/{v}/fold{i}.json", "w"))
    np.savez_compressed(f"{REPO}/runs/s263/{v}/fold{i}.npz", idx=np.flatnonzero(te), **{h: store[h][te] for h in HEADS})
    print(v, name, {h: round(100 * b, 2) for h, b in res.items()}, f"fit {fit:.0f}s", flush=True)


if __name__ == "__main__":
    main()
