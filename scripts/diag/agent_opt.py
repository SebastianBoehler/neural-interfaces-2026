"""Agent-as-optimizer harness (2 Nov): an LLM agent proposes configurations of everything a sealed
package can still change (teacher map, member weights, rebalancing, R19e stage) and reads the score.

Usage: python scripts/diag/agent_opt.py CONFIG.json [--group A|G1|G2]  (search: what the agent sees)
       python scripts/diag/agent_opt.py CONFIG.json --confirm  (held out: run by us, never the agent)
Search scores only dev_loso_all on the group's participants with teacher seeds 0-5; nothing else is
computed. Round 1 used A = P01 P02 P03 P05 P06 (held out B = P07 P08 P10). Round 2 cross-fits two
balanced folds, G1 = P01 P02 P05 P07 and G2 = P03 P06 P08 P10: an agent searches one, its pick is
confirmed on the other. Search output also gives per-class recall, the confusion matrix (rows true
MI / SUB / WG), per-interface change and the teacher's own accuracy, all on the group only.
--confirm adds every participant, dev_session (P10's later sessions) and seeds 6-11 and 0-11.
Round 3 (``--nested GRID.json``, a list of 2-8 configs): for each of the 8 dev participants the grid
config (or the package, always candidate 0) with the best mean change over the other seven is chosen
and scores the left-out person, so the printed score already pays for the selection. Prints only the
nested per-person change against the package, the pick per fold, and the full-set pick.
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
from diag_common import cells, cm, d, mixp, rebal  # noqa: E402
from eval_external_teacher import by_participant  # noqa: E402
import session_em as SE  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

A, B_ = ("P01", "P02", "P03", "P05", "P06"), ("P07", "P08", "P10")
GROUPS = {"A": A, "G1": ("P01", "P02", "P05", "P07"), "G2": ("P03", "P06", "P08", "P10")}
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
    run.last = dict(ix=ix, e=e.argmax(1), r=r, t=T[rows].argmax(1))
    return by_participant(ix, e.argmax(1)), by_participant(ix, r), cm(ix, e.argmax(1)) - cm(ix, r)


def diagnostics(group):
    L = run.last
    ix, yt = L["ix"], y[L["ix"]]
    conf = lambda p: (np.bincount(3 * yt + p, minlength=9).reshape(3, 3) / np.bincount(yt, minlength=3)[:, None])
    by_if = lambda p: {i: np.mean([v for (_, j), v in cells(ix, p).items() if j == i]) for i in ("Graz", "BrainHero")}
    ce, cr = by_if(L["e"]), by_if(L["r"])
    return {"recall_MI_SUB_WG": np.round(100 * np.diag(conf(L["e"])), 1).tolist(),
            "recall_R19e": np.round(100 * np.diag(conf(L["r"])), 1).tolist(),
            "confusion_rows_true": np.round(100 * conf(L["e"]), 1).tolist(),
            "interface_delta": {i: round(float(ce[i] - cr[i]), 2) for i in ce},
            "teacher_alone_bacc": {p: round(v, 1) for p, v in by_participant(ix, L["t"]).items()}}


def prep(cfg):
    if cfg.get("edit_py"):
        s = importlib.util.spec_from_file_location("edit", cfg["edit_py"])
        cfg["_edit"] = importlib.util.module_from_spec(s)
        s.loader.exec_module(cfg["_edit"])
    return cfg


def nested(Bc, H, grid):
    P8 = GROUPS["G1"] + GROUPS["G2"]
    D = []
    for cfg in [{}] + grid:
        pp, ref, _ = run(Bc, H, prep(cfg), "dev_loso_all", range(6), list(P8))
        D.append({p: pp[p] - ref[p] for p in P8})
    best = lambda ps: max(range(len(D)), key=lambda j: np.mean([D[j][q] for q in ps]))
    picks = {p: best([q for q in P8 if q != p]) for p in P8}
    gain = {p: round(D[picks[p]][p] - D[0][p], 2) for p in P8}
    print(json.dumps({"nested_gain_vs_package": round(float(np.mean(list(gain.values()))), 3),
                      "nested_gain_without_P10": round(float(np.mean([v for p, v in gain.items() if p != "P10"])), 3),
                      "per_person_gain": gain, "pick_per_fold": picks, "full_set_pick": best(P8),
                      "note": "pick 0 = package; pick j = grid[j-1]"}))


def main():
    Bc, H = np.load(ROOT / "runs/diag/base_cache.npz"), np.load(ROOT / "runs/diag/screenW5.npz")
    if "--nested" in sys.argv:
        grid = json.loads(Path(sys.argv[sys.argv.index("--nested") + 1]).read_text())
        assert 2 <= len(grid) <= 8, "a grid holds 2-8 candidate configs"
        return nested(Bc, H, grid)
    cfg, confirm = prep(json.loads(Path(sys.argv[1]).read_text())), "--confirm" in sys.argv
    if not confirm:
        g = GROUPS[sys.argv[sys.argv.index("--group") + 1]] if "--group" in sys.argv else A
        pp, ref, _ = run(Bc, H, cfg, "dev_loso_all", range(6), list(g))
        delta = {p: round(pp[p] - ref[p], 2) for p in g}
        print(json.dumps({"search_mean_delta_vs_R19e": round(float(np.mean(list(delta.values()))), 3),
                          "worst_delta": min(delta.values()), "per_participant_delta": delta, **diagnostics(g)}))
        return
    out = {}
    for half, seeds in (("s05", range(6)), ("s611", range(6, 12)), ("s011", range(12))):
        pp, ref, dla_mean = run(Bc, H, cfg, "dev_loso_all", seeds)
        _, _, ds_d = run(Bc, H, cfg, "dev_session", seeds)
        dla = {p: round(pp[p] - ref[p], 2) for p in pp}
        out[half] = {"sel_delta": round(float((dla_mean + ds_d) / 2), 2), "dla_delta": round(float(dla_mean), 2),
                     "ds_delta": round(float(ds_d), 2), "A_mean": round(float(np.mean([dla[p] for p in A])), 2),
                     "B_mean": round(float(np.mean([dla[p] for p in B_])), 2),
                     **{f"{k}_mean": round(float(np.mean([dla[p] for p in GROUPS[k]])), 2) for k in ("G1", "G2")},
                     "no_P10_mean": round(float(np.mean([v for p, v in dla.items() if p != "P10"])), 2),
                     "per_participant": dla}
        print(half, json.dumps(out[half]), flush=True)


if __name__ == "__main__":
    main()
