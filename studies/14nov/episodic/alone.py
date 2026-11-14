"""EPISODIC, family alone (ledger s220): the fbraw family's frozen output, stock (ttt7 / ttt18 base) against the paired
control E0 and the episodic arm E, raw and after label-free whole-session balancing (per session the per-class bias b
with mean softmax(log p + b) uniform; non-causal, the panel's "post-balance"). Person-equal mean of cell balanced
accuracy (cell = person x session x interface); new = all_loso, seen = seen_session (P04, P10).
Family = mean of the seeds' posteriors (all six, 0-2, 3-5). Orders: rec and the six delivery orders.
Usage: python -I studies/14nov/episodic/alone.py [EPI_DIR=runs/episodic/probs]"""
import os, sys
from pathlib import Path
WS = Path(__file__).resolve().parents[3]; sys.path[:0] = [str(WS / "experiments")]
import numpy as np  # noqa: E402
import protocol  # noqa: E402

EPI = WS / os.environ.get("EPI_DIR", "runs/episodic/probs"); D7, D18 = WS / "runs/ttt/probs7", WS / "runs/ttt/probs18"
ORD = ["rec0", "within0", "within1", "full0", "full1", "runs0", "runs1"]; SETS = {"s0-5": range(6), "s0-2": range(3), "s3-5": range(3, 6)}
d = protocol.load(); Y = d["y"].astype(int)


def path(arm, o, proto, s):
    if arm == "stock":
        return D7 / f"ttt7_fbraw_base-{proto}-s{s}.npz" if o == "rec0" else D18 / f"ttt18{o}_fbraw_base-{proto}-s{s}.npz"
    return EPI / f"ttt18{o}{arm}_fbraw_base-{proto}-s{s}.npz"


def load(arm, o, proto, seeds):
    zs = [np.load(path(arm, o, proto, s)) for s in seeds]; idx = zs[0]["index"]
    assert all((z["index"] == idx).all() for z in zs); return np.mean([z["proba"] for z in zs], 0), idx


def balance(P, idx, iters=50):
    L = np.log(np.clip(P, 1e-9, 1)); out = L.copy(); st = d["stream"][idx]
    for s in np.unique(st):
        m = st == s; b = np.zeros(3)
        for _ in range(iters):
            q = np.exp(L[m] + b); q /= q.sum(1, keepdims=True); b -= np.log(q.mean(0)); b -= b.mean()
        out[m] = L[m] + b
    return out


def score(P, idx):
    pred, y = P.argmax(1), Y[idx]; cell = np.char.add(np.char.add(d["subject"][idx], "|" + d["session"][idx].astype(str)), "|" + d["interface"][idx])
    per = {}
    for c in np.unique(cell):
        m = cell == c; per.setdefault(d["subject"][idx][m][0], []).append(np.mean([np.mean(pred[m][y[m] == k] == k) for k in np.unique(y[m])]))
    return {p: 100 * np.mean(v) for p, v in per.items()}


def main():
    arms = [a for a in ("stock", "E0", "E") if all(path(a, o, pr, s).exists() for o in ORD for pr in ("all_loso", "seen_session") for s in range(6))]
    print("arms with all files:", arms, flush=True)
    if "E0" in arms:  # no-op check: E0 is the stock recipe through the episodic runner
        dif = max(np.abs(np.load(path("E0", "rec0", pr, s))["proba"] - np.load(path("stock", "rec0", pr, s))["proba"]).max() for pr in ("all_loso", "seen_session") for s in range(6))
        print(f"E0 vs ttt7 stock base, recording order: max |dp| {dif:.2e}", flush=True)
    res = {}
    for proto in ("all_loso", "seen_session"):
        for a in arms:
            for o in ORD:
                for sn, ss in SETS.items():
                    P, idx = load(a, o, proto, ss)
                    res[proto, a, o, sn, "raw"], res[proto, a, o, sn, "bal"] = score(P, idx), score(balance(P, idx), idx)
    for proto in ("all_loso", "seen_session"):
        print(f"\n== {proto}: person-equal cell BA (family = seed-mean); delta vs stock, and without P10 (new)", flush=True)
        for sn in SETS:
            for kind in ("raw", "bal"):
                for o in ("rec0", "mean6"):
                    os_ = ["rec0"] if o == "rec0" else ORD[1:]
                    v = {a: np.mean([np.mean(list(res[proto, a, oo, sn, kind].values())) for oo in os_]) for a in arms}
                    line = f"{sn} {kind} {o:6s} " + " ".join(f"{a} {v[a]:.2f}" for a in arms)
                    for a in [x for x in arms if x != "stock"]:
                        dp = np.mean([[res[proto, a, oo, sn, kind][p] - res[proto, "stock", oo, sn, kind][p] for p in sorted(res[proto, a, oo, sn, kind])] for oo in os_], 0)
                        nop10 = np.mean([x for p, x in zip(sorted(res[proto, a, os_[0], sn, kind]), dp) if not p.endswith("P10")])
                        line += f" | {a}-stock {dp.mean():+.2f} (w/o P10 {nop10:+.2f}; up {int((dp > 0).sum())}/{len(dp)})"
                    print(line, flush=True)
    np.save(WS / "runs/episodic/alone.npy", {"|".join(k): v for k, v in res.items()}, allow_pickle=True)


if __name__ == "__main__":
    main()
