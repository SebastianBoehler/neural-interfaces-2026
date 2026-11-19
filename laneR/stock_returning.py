"""Lane R step 1 (a)+(b): absolute returning BA, recognition rate/latency of cap250 from stored tapes.
python laneR/stock_returning.py LANEA_OUT SCHERER_RET_PARENT SCHERER_NEW_DIR OUT.json
(a) Lane A out/<order>_<P>_<visit>/*.npz arm 'stock'; (b) NI-V ret_cap250_k*/p*_c*.npz and cap250/p*_c*.npz (visit 2 only)."""
import glob, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ba = lambda y, p: 100 * float(np.mean([(p[y == c] == c).mean() for c in range(3)]))


def main(la, ret, new, dest):
    y = np.load(ROOT / "data/proteus_windows.npz", allow_pickle=True)["y"]
    out = {"proteus": {}, "scherer": {}}
    for p in ("P04", "P10"):
        for v in ("2", "3"):
            r = []
            for d in sorted(glob.glob(f"{la}/*_{p}_{v}")):
                n = Path(d).name
                z = np.load(f"{d}/{n}.npz"); j = json.load(open(f"{d}/{n}.json"))
                r.append((ba(y[z["rows"]], z["stock|labels"]), j["stats"]["stock"]["recognition"], j["stats"]["stock"]["routed"], j["n"]))
            assert len(r) == 8, (p, v, len(r))
            out["proteus"][f"{p}|{v}"] = dict(ba=np.mean([x[0] for x in r]), rec_age=[x[1] for x in r], routed_frac=float(np.mean([x[2] / x[3] for x in r])),
                                               n_recognised=sum(x[1] >= 0 for x in r))
    s = np.load(ROOT / "data/scherer2015/windows.npz", allow_pickle=True)
    ys = s["y"]
    def collect(pattern):
        res = {}
        for f in glob.glob(pattern):
            z = np.load(f)
            for k in z.files:
                if k.endswith("|labels"):
                    sch, i, who, vis = k.split("|")[:4]
                    if vis != "2": continue
                    rows = z[k[:-6] + "rows"]; meta = z[k[:-6] + "meta"]
                    res[(who, sch, i)] = (ba(ys[rows], z[k].astype(int)), meta[0], meta[1] / len(rows))
        return res
    R, N = collect(f"{ret}/ret_cap250_k*/p*_c*.npz"), collect(f"{new}/p*_c*.npz")
    people = sorted({k[0] for k in R})
    for who in people:
        out["scherer"][who] = {}
        for sch in ("full", "within"):
            rk = [v for k, v in R.items() if k[0] == who and k[1] == sch]; nk = [v for k, v in N.items() if k[0] == who and k[1] == sch]
            out["scherer"][who][sch] = dict(ret=np.mean([x[0] for x in rk]), new=np.mean([x[0] for x in nk]) if nk else None, n_orders=len(rk),
                                            rec_rate=float(np.mean([x[1] >= 0 for x in rk])), rec_age=float(np.mean([x[1] for x in rk if x[1] >= 0] or [-1])),
                                            routed_frac=float(np.mean([x[2] for x in rk])))
    json.dump(out, open(dest, "w"), indent=1, default=float)
    for k, v in out["proteus"].items(): print("PROTEUS", k, round(v["ba"], 2), "rec", v["n_recognised"], "/8 ages", sorted(v["rec_age"]), "routed", round(v["routed_frac"], 2))
    for sch in ("full", "within"):
        print("SCHERER", sch, "ret %.2f new %.2f" % (np.mean([out["scherer"][w][sch]["ret"] for w in people]), np.mean([out["scherer"][w][sch]["new"] for w in people])),
              "rec rate %.2f age %.1f routed %.2f" % tuple(np.mean([out["scherer"][w][sch][m] for w in people]) for m in ("rec_rate", "rec_age", "routed_frac")))


if __name__ == "__main__":
    main(*sys.argv[1:])
