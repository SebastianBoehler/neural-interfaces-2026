"""Review 13 Nov, item 6: what do v5.2's personal nets cost on a false accept? On every public session, the routed windows
use ANOTHER person's personal traces (person (j + k) mod 10 for offsets k = 1, 4, 7) while the gate, the head slot and the
own-space stage keep the routed person j; compared with the same session under v5.1b (no personal nets) and v5.2 (the
right person's nets). In-sample for the right person (the nets were fine-tuned on these sessions); the impostor arms are
out-of-person, which is the false-accept situation. Usage: python studies/13nov/impostor_pnets.py PKG_V52 PKG_V51B [DEVICE]"""
import json, os, sys
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[2]))
sys.path[:0] = [str(WS / "experiments"), str(WS / "scripts")]
import numpy as np  # noqa: E402
import protocol  # noqa: E402
from add_rebalance import load_solver  # noqa: E402


def stream(m, d, rows, k=None):
    m.reset_state(); out = []
    if k is not None:
        pm0 = type(m)._pmix
        m._pmix = lambda j, b, t, pm: pm0(m, (j + k) % 10, b, t, pm)
    for i in rows:
        out.append(int(np.asarray(m.predict(d["X"][i:i + 1]))[0]))
    if k is not None:
        del m._pmix
    return np.array(out)


def bacc(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in np.unique(y)]))


def main():
    v52, v51 = Path(sys.argv[1]), Path(sys.argv[2]); dev = sys.argv[3] if len(sys.argv) > 3 else None
    d = protocol.load(); m, b = load_solver(v52, "v52", dev), load_solver(v51, "v51b", dev); res = {}
    for s in dict.fromkeys(d["stream"]):
        rows = np.flatnonzero(d["stream"] == s); y = d["y"][rows]
        r = {"v51b": bacc(y, stream(b, d, rows)), "v52_right": bacc(y, stream(m, d, rows))}
        for k in (1, 4, 7):
            r[f"v52_wrong{k}"] = bacc(y, stream(m, d, rows, k))
        res[s.split("/")[-1]] = r; print(s, json.dumps(r), flush=True)
    keys = list(next(iter(res.values())))
    mean = {k: float(np.mean([r[k] for r in res.values()])) for k in keys}
    wrong = np.array([[r[f"v52_wrong{k}"] - r["v51b"] for k in (1, 4, 7)] for r in res.values()])
    print("MEAN", json.dumps(mean), flush=True)
    print("IMPOSTOR minus v51b, balanced accuracy points: mean", round(100 * wrong.mean(), 2), "worst session", round(100 * wrong.min(), 2),
          "share of session-offsets below -2 points", round(float((wrong < -0.02).mean()), 2), flush=True)


if __name__ == "__main__":
    main()
