"""Agent-as-optimizer harness (2 Nov): an LLM agent proposes configurations of everything a sealed
package can still change (teacher map, member weights, rebalancing, R19e stage) and reads the score.

Usage: python scripts/diag/agent_opt.py CONFIG.json            (search: what the agent sees)
       python scripts/diag/agent_opt.py CONFIG.json --confirm  (held out: run by us, never the agent)
Search scores only dev_loso_all on participants A = P01 P02 P03 P05 P06 with teacher seeds 0-5;
nothing else is computed. --confirm adds B = P07 P08 P10, dev_session (P10's later sessions) and
teacher seeds 6-11 and 0-11, so the agent's choice is tested on people and seeds it never saw.
Config keys (all optional; defaults = cand_nxif):
  fams ["eegnex_ea-m7r", "ifnet_ea-m7r"]  heads in runs/diag/screenW5.npz (also sincshallow_ea-m7r,
       fbcnet_ea-m7r, fblight_ea-m7r, shallow_ea-m7r);   C 0.01;   w 5 (teacher weight)
  drop [regex]  zero head columns whose "arch|corpus:class" name matches;  scale {regex: factor}
  edit_py  file with edit(coef, b, names) -> (coef, b) and/or feats(F, names) -> (F, names)
  member_weights [1, .5, .5, 1, 1, 1, 1, 1, 1]  (v2, fbcnet_ea_tf, fbcnet_raw_tf, shallow_ea,
       shallow_raw, fbcnet_raw, fbcnet_ea_rot05, sincshallow_ea_ptm7r_rot05, fblight_ea_rot05)
  rebal {"alpha": 1, "k": 5, "W": 30};   em {"lam": .5, "n0": 60, "m0": 20, "shrink": .7, "iters": 2,
       "nmin": 10, "events": true}  (events false drops the 16 ocular-event rates)
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts/diag")]
from diag_common import cm, d, mixp, rebal  # noqa: E402
from eval_external_teacher import by_participant  # noqa: E402
import session_em as SE  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

A, B_ = ("P01", "P02", "P03", "P05", "P06"), ("P07", "P08", "P10")
WS = [1, 0.5, 0.5, 1, 1, 1, 1, 1, 1]
y, subj, st = d["y"], d["subject"], d["stream"]
DEV = ~np.isin(subj, [f"Dreyer2026Proteus/{p}" for p in ("P04", "P09")])
short = np.array([s.split("/")[-1] for s in subj])


def heads(H, cfg, seeds):
    fams = cfg.get("fams", ["eegnex_ea-m7r", "ifnet_ea-m7r"])
    F = np.concatenate([np.mean([H[f"{f}-s{s}"] for s in seeds], 0) for f in fams], 1)
    names = [f"{f.split('_')[0]}|{n}" for f in fams for n in H["names"]]
    keep = np.array([not any(re.search(r, n) for r in cfg.get("drop", [])) for n in names])
    F = F * keep
    for r, fac in cfg.get("scale", {}).items():
        F = F * np.array([fac if re.search(r, n) else 1.0 for n in names])
    ed = cfg.get("_edit")
    if ed is not None and hasattr(ed, "feats"):
        F, names = ed.feats(F, names)
    return F, names


def teacher(F, names, cfg, proto, idx):
    def fit(tr, te):
        lr = LogisticRegression(C=cfg.get("C", 0.01), max_iter=3000, class_weight="balanced").fit(F[tr], y[tr])
        coef, b = lr.coef_.copy(), lr.intercept_.copy()
        if cfg.get("_edit") is not None and hasattr(cfg["_edit"], "edit"):
            coef, b = cfg["_edit"].edit(coef, b, names)
        z = F[te] @ coef.T + b
        z = np.exp(z - z.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)
    T = np.zeros((len(y), 3))
    if proto == "dev_session":
        T[idx] = fit(DEV & ~np.isin(np.arange(len(y)), idx), idx)
    else:
        for p in sorted(set(subj[idx])):
            T[subj == p] = fit(subj != p, subj == p)
    return T[idx]


def run(Bc, H, cfg, proto, seeds, people=None):
    k = f"s05_{proto}"
    idx, M = Bc[k + "_idx"], list(Bc[k + "_members"])
    rows = np.isin(short[idx], people) if people else np.ones(len(idx), bool)
    F, names = heads(H, cfg, seeds)
    T = teacher(F, names, cfg, proto, idx)
    rb = {"alpha": 1.0, "k": 5.0, "W": 30, **cfg.get("rebal", {})}
    em = {"lam": 0.5, "n0": 60, "m0": 20, "shrink": 0.7, "iters": 2, "nmin": 10, **cfg.get("em", {})}
    view = np.c_[Bc["G"], Bc["EV"]] if em.pop("events", True) else Bc["G"]
    ix = idx[rows]
    b = rebal(mixp([P[rows] for P in M] + [T[rows]], cfg.get("member_weights", WS) + [cfg.get("w", 5)]), ix, **rb)
    e = SE.apply(view[ix], b, st[ix], **em)
    r = Bc[k + "_R19e"][rows].argmax(1)
    return by_participant(ix, e.argmax(1)), by_participant(ix, r), cm(ix, e.argmax(1)) - cm(ix, r)


def main():
    cfg, confirm = json.loads(Path(sys.argv[1]).read_text()), "--confirm" in sys.argv
    if cfg.get("edit_py"):
        s = importlib.util.spec_from_file_location("edit", cfg["edit_py"])
        cfg["_edit"] = importlib.util.module_from_spec(s)
        s.loader.exec_module(cfg["_edit"])
    Bc, H = np.load(ROOT / "runs/diag/base_cache.npz"), np.load(ROOT / "runs/diag/screenW5.npz")
    if not confirm:
        pp, ref, _ = run(Bc, H, cfg, "dev_loso_all", range(6), list(A))
        delta = {p: round(pp[p] - ref[p], 2) for p in A}
        print(json.dumps({"search_mean_delta_vs_R19e": round(float(np.mean(list(delta.values()))), 3),
                          "worst_delta": min(delta.values()), "per_participant_delta": delta}))
        return
    out = {}
    for half, seeds in (("s05", range(6)), ("s611", range(6, 12)), ("s011", range(12))):
        pp, ref, dla_mean = run(Bc, H, cfg, "dev_loso_all", seeds)
        _, _, ds_d = run(Bc, H, cfg, "dev_session", seeds)
        dla = {p: round(pp[p] - ref[p], 2) for p in pp}
        out[half] = {"sel_delta": round(float((dla_mean + ds_d) / 2), 2), "dla_delta": round(float(dla_mean), 2),
                     "ds_delta": round(float(ds_d), 2), "A_mean": round(float(np.mean([dla[p] for p in A])), 2),
                     "B_mean": round(float(np.mean([dla[p] for p in B_])), 2),
                     "no_P10_mean": round(float(np.mean([v for p, v in dla.items() if p != "P10"])), 2),
                     "per_participant": dla}
        print(half, json.dumps(out[half]), flush=True)


if __name__ == "__main__":
    main()
