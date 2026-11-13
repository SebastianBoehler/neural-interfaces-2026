"""Does the packaged fine-tuning act? (coordinator request, 13 Nov.) Every public session streamed with both gates forced
off through the final v6 packages with fine-tuning (gc5) and without (noft); v5.1b's output is v6's inner v5.1b chain
(equal to lean_v5 with the same gates, scripts/check_v6.py). Reports per session and pooled: labels changed against
v5.1b for each form, labels differing between gc5 and noft, balanced accuracy of the three, and max |gc5 - noft| on the
windows where the fine-tuned posteriors were swapped in (window >= 30). In-sample: the members saw these sessions.
Usage: python studies/13nov/v6_ft_effect.py DEVICE PKG_GC5 PKG_NOFT [SCHEME:SEED:PERMS.npz]   (delivery order of shuffle.py)"""
import copy, json, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "lean/tools")]
import protocol  # noqa: E402
from record import load_model  # noqa: E402


def run(m, X):
    c = copy.deepcopy(m); c.route.contrast.tau = c.route.identity.tau = np.inf; c.reset_state(); out, inner, ft = [], [], []
    for x in X:
        out.append(c.step(np.nan_to_num(np.asarray(x, np.float32)))); inner.append(c.trace["out_v5"]); ft.append(bool(c.trace["v6"].get("p_ft")))
    assert c.log6["err"] == 0, c.log6
    return np.array(out), np.array(inner), np.array(ft)


def bacc(y, p):
    return float(np.mean([np.mean(p[y == k] == k) for k in np.unique(y)]))


def main():
    dev, pg, pn = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]); d = protocol.load(); torch.set_num_threads(1)
    G, N = load_model(pg, None, "gc5", dev), load_model(pn, None, "noft", dev); res, Y, L5, LG, LN = {}, [], [], [], []
    order = None
    if len(sys.argv) > 4:
        sc, sd, f = sys.argv[4].split(":"); order = np.load(f)[f"{sc}|{sd}|all_loso"]; assert sorted(order.tolist()) == list(range(len(d["y"])))
    import time
    for s in dict.fromkeys(d["stream"]):
        rows = order[d["stream"][order] == s] if order is not None else np.flatnonzero(d["stream"] == s); X, y = d["X"][rows], d["y"][rows]
        t0 = time.perf_counter()
        og, i5, ftw = run(G, X); tg = time.perf_counter() - t0; t1 = time.perf_counter(); on, i5n, ftn = run(N, X); tn = time.perf_counter() - t1
        assert np.array_equal(i5, i5n) and not ftn.any() and ftw.sum() == max(0, len(rows) - 30)
        l5, lg, ln = i5.argmax(1), og.argmax(1), on.argmax(1); Y.append(y); L5.append(l5); LG.append(lg); LN.append(ln)
        res[s.split("/")[-1]] = dict(windows=len(rows), ft_windows=int(ftw.sum()), changed_gc5=int((lg != l5).sum()), changed_noft=int((ln != l5).sum()),
                                     gc5_vs_noft=int((lg != ln).sum()), max_dp_gc5_noft_ft=float(np.abs(og[ftw] - on[ftw]).max()),
                                     bacc_v51b=bacc(y, l5), bacc_noft=bacc(y, ln), bacc_gc5=bacc(y, lg), sec_gc5=round(tg, 1), sec_noft=round(tn, 1))
        print(s, json.dumps(res[s.split("/")[-1]]), flush=True)
    Y, L5, LG, LN = map(np.concatenate, (Y, L5, LG, LN))
    print("POOLED", json.dumps(dict(order=sys.argv[4] if len(sys.argv) > 4 else "recording", windows=len(Y), changed_gc5=int((LG != L5).sum()), changed_noft=int((LN != L5).sum()), gc5_vs_noft=int((LG != LN).sum()),
                                    mean_session_bacc=dict(v51b=float(np.mean([r["bacc_v51b"] for r in res.values()])), noft=float(np.mean([r["bacc_noft"] for r in res.values()])),
                                                           gc5=float(np.mean([r["bacc_gc5"] for r in res.values()]))),
                                    max_dp_gc5_noft_ft=max(r["max_dp_gc5_noft_ft"] for r in res.values()))), flush=True)


if __name__ == "__main__":
    main()
