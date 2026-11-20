"""Score and compare TTT variants from stream_valid.py outputs (numpy only + official metric from the submodule).

python analyse.py DIR_WITH_EVAL_NPZ MODE OUT.json [--base A1]
Person value = mean of the person's official cells (stream x interface); delta = variant - base paired per (person,order);
bootstrap resamples people AND orders (10k draws), 80 % / 95 % intervals. Trajectory/collapse statistics per stream-order.
"""
import argparse, glob, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "2026-competition/tracks/bci_decoding"))
from benchmark_utils.metrics import balanced_accuracy  # noqa: E402


def cells(z, k, v):
    rows, lab = z[f"rows|{k}"], z[f"{v}|{k}|label"]
    key = np.char.add(np.char.add(z["stream"][rows], "|"), z["iface"][rows])
    res = {}
    for c in dict.fromkeys(key):
        m = key == c
        res[c] = 100 * balanced_accuracy(z["y"][rows][m], lab[m])
    return res


def boot(M, n=10000, seed=0):
    """M (people, orders) paired deltas -> mean, 80 % and 95 % people+orders bootstrap intervals."""
    rng = np.random.default_rng(seed)
    P, O = M.shape
    ip, io = rng.integers(0, P, (n, P)), rng.integers(0, O, (n, O))
    s = M[ip[:, :, None], io[:, None, :]].mean((1, 2))
    q = lambda l, h: [round(float(np.percentile(s, l)), 2), round(float(np.percentile(s, h)), 2)]  # noqa: E731
    return dict(mean=round(float(M.mean()), 2), i80=q(10, 90), i95=q(2.5, 97.5))


def traj(z, k, v, base):
    """Per stream: accuracy by delivered position quarter, max class share over rolling 40-window blocks, collapse flags."""
    rows = z[f"rows|{k}"]
    out = []
    for s in dict.fromkeys(z["stream"][rows]):
        m = z["stream"][rows] == s
        y = z["y"][rows][m]
        L, B = z[f"{v}|{k}|label"][m], z[f"{base}|{k}|label"][m]
        n = len(y)
        share = lambda l: max(np.bincount(l[i:i + 40], minlength=3).max() / len(l[i:i + 40]) for i in range(16, n - 39))  # noqa: E731
        q = np.array_split(np.arange(n), 4)
        out.append(dict(stream=str(s), n=n, acc_q=[round(float((L[i] == y[i]).mean()), 3) for i in q],
                        base_acc_q=[round(float((B[i] == y[i]).mean()), 3) for i in q],
                        max_share=round(float(share(L)), 3), base_max_share=round(float(share(B)), 3),
                        ttt_collapse=bool(share(L) >= 0.85 and share(B) <= 0.70), onecls=bool(share(L) >= 0.95)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir"), ap.add_argument("mode"), ap.add_argument("out")
    ap.add_argument("--base", default="A1")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.dir}/eval_*.npz"))
    assert files, "no eval files"
    Z = {Path(f).stem.split("_", 1)[1]: np.load(f) for f in files}
    persons = sorted(Z)
    z0 = Z[persons[0]]
    variants = sorted({m.group(1) for f in z0.files if (m := re.fullmatch(r"([^|]+)\|\d+\|label", f))})
    orders = sorted({int(m.group(1)) for f in z0.files if (m := re.fullmatch(r"rows\|(\d+)", f))})
    val = {v: np.array([[np.mean(list(cells(Z[p], k, v).values())) for k in orders] for p in persons]) for v in variants}
    res = dict(mode=a.mode, persons=persons, orders=orders, base=a.base, level={}, delta={}, per_person={}, traj={}, ms={})
    for v in variants:
        res["level"][v] = dict(mean=round(float(val[v].mean()), 2), by_person={p: round(float(val[v][i].mean()), 2) for i, p in enumerate(persons)})
        res["ms"][v] = round(1e3 * float(np.mean(np.concatenate([Z[p][f"{v}|{k}|sec"] for p in persons for k in orders]))), 1)
        if v == a.base:
            continue
        res["delta"][v] = boot(val[v] - val[a.base])
        res["per_person"][v] = {p: round(float((val[v] - val[a.base])[i].mean()), 2) for i, p in enumerate(persons)}
        res["per_person"][v]["worst_person"] = min(res["per_person"][v].items(), key=lambda t: t[1])
        T = [dict(person=p, order=k, **t) for p in persons for k in orders for t in traj(Z[p], k, v, a.base)]
        res["traj"][v] = dict(streams=len(T), ttt_collapse=sum(t["ttt_collapse"] for t in T), onecls=sum(t["onecls"] for t in T),
                              worst_stream_order=min(T, key=lambda t: sum(t["acc_q"]) - sum(t["base_acc_q"])),
                              mean_acc_q=np.round(np.mean([t["acc_q"] for t in T], 0), 3).tolist(),
                              mean_base_acc_q=np.round(np.mean([t["base_acc_q"] for t in T], 0), 3).tolist())
    json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({k: res[k] for k in ("level", "delta", "ms")}, indent=1, default=str)[:3000])
    for v, t in res["traj"].items():
        print(v, {k: t[k] for k in ("streams", "ttt_collapse", "onecls", "mean_acc_q", "mean_base_acc_q")}, res["per_person"][v]["worst_person"])


if __name__ == "__main__":
    main()
