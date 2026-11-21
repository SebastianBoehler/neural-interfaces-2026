"""R1a on the hard-only and hardcb bases, replayed through the real final Solver on real fold packages.

python r1a_bases.py MODE PERSON FIRST LAST OUT [--smoke] [--heldnew]
MODE hard|hardcb. Returning mode: fold ret_PERSON (visit 1 enrolled), visits 2-3, full-session orders FIRST..LAST;
writes OUT/<arm>/pred_ret_PERSON.npz (rows_/labels_ keys as stream_fold.py) for arms MODE and MODE_r1a, plus a json
with recognition age and late-refit ages per stream. Fails loudly unless every stream is recognised and R1a refits
exactly once (age 100) on the R1a arm and never on the parent. --heldnew: fold new_PERSON, visit 1, requires zero
recognition and identical labels. Orders 71000-71003 are checked against the stored hardcb / R1a tapes when given.
"""
import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "submissions/final_stack")]
from check_r1a import load_templates, replay, verify  # noqa: E402
from data import delivery  # noqa: E402

W = Path("/leonardo_work/AIFAC_P02_1636/ni26/cap250/runs")
FOLDS = W / "production-20261120-32a94ab7/hardcb"
STORED = {"hardcb": FOLDS, "hardcb_r1a": W / "redteam-21nov/R1a"}


def main(a):
    verify(a.zips, a.mode)
    with np.load(a.data, allow_pickle=False) as d:
        sessions = np.char.add(np.char.add(d["subject"], "|"), d["session"])
        channels, x = d["ch_names"], torch.from_numpy(d["X"])
    orders = range(a.first, a.last + 1)
    visits = ("1",) if a.heldnew else ("2", "3")
    fold = FOLDS / (f"new_{a.person}" if a.heldnew else f"ret_{a.person}") / "pkg"
    arms = (a.mode, f"{a.mode}_r1a")
    tapes = {arm: {} for arm in arms}
    info, bad = [], []
    with ExitStack() as stack:
        models = load_templates(stack, a.zips, a.mode, channels, "cuda:0", fold)
        for order in orders:
            plan = delivery(sessions, order)
            for visit in visits:
                rows = plan[sessions[plan] == f"Dreyer2026Proteus/{a.person}|{visit}"]
                if a.smoke and (order, visit) != (a.first, visits[0]):
                    continue
                res = [replay(m, x[rows].to("cuda:0")) for m in models]
                for arm, (labels, rec, late, sec) in zip(arms, res):
                    tapes[arm][f"rows_{order}_{visit}"] = rows
                    tapes[arm][f"labels_{order}_{visit}"] = labels
                    if arm == arms[0] and late:
                        bad.append(f"{arm} {order}_{visit}: parent refit after recognition {late}")
                if res[0][1] != res[1][1]:
                    bad.append(f"{order}_{visit}: recognition differs {res[0][1]} vs {res[1][1]}")
                if a.heldnew and (res[1][1] >= 0 or res[1][2]):
                    bad.append(f"{order}_{visit}: held-new recognised or refit {res[1][1:3]}")
                if not a.heldnew and (not 0 <= res[1][1] < 100 or res[1][2] != [100]):
                    bad.append(f"{order}_{visit}: R1a recognition {res[1][1]}, late {res[1][2]}")
                changed = int(np.count_nonzero(res[0][0] != res[1][0]))
                if a.heldnew and changed:
                    bad.append(f"{order}_{visit}: {changed} labels changed on held-new")
                row = dict(order=order, visit=visit, windows=len(rows), recognised=res[1][1], late=res[1][2],
                           changed=changed, seconds=[round(r[3], 1) for r in res])
                for arm in arms:  # stored-tape parity (hardcb only, orders 71000-71003)
                    path = STORED.get(arm, Path()) / f"pred_ret_{a.person}.npz"
                    if not a.heldnew and path.is_file() and order <= 71003:
                        with np.load(path) as t:
                            np.testing.assert_array_equal(t[f"rows_{order}_{visit}"], rows)
                            row[f"stored_mismatch_{arm}"] = int(np.count_nonzero(
                                t[f"labels_{order}_{visit}"] != tapes[arm][f"labels_{order}_{visit}"]))
                info.append(row)
                print(json.dumps(row), flush=True)
    tag = f"{'new' if a.heldnew else 'ret'}_{a.person}"
    for arm in arms:
        (a.out / arm).mkdir(parents=True, exist_ok=True)
        np.savez(a.out / arm / f"pred_{tag}.npz", **tapes[arm])
    (a.out / f"info_{a.mode}_{tag}_{a.first}_{a.last}.json").write_text(json.dumps(info, indent=1))
    if bad:
        raise SystemExit("FAILED: " + "; ".join(bad))
    print("OK", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("hard", "hardcb"))
    p.add_argument("person")
    p.add_argument("first", type=int)
    p.add_argument("last", type=int)
    p.add_argument("out", type=Path)
    p.add_argument("--zips", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--heldnew", action="store_true")
    main(p.parse_args())
