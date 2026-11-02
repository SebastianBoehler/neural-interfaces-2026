"""Member additions and swaps in R19e from saved probabilities (overnight 2 Nov).

Usage: python scripts/diag/eval_members.py BASE_CACHE OUT_JSON NAME=OP[,OP...] ...
OP is ``+SPEC`` (add a member) or ``OLD~SPEC`` (replace the R15 member named OLD), with
SPEC as in mix_probs (``name:lo-hi*weight``). Each variant is mixed with the R15 member
weights, rebalanced and passed through the R19e stage (lam 0.5, n0 60, m0 20; gaze and
event features from BASE_CACHE), exactly as eval_external_teacher.py. The unchanged R15
mix is the reference and must reproduce BASE_CACHE's s05 R19e. Reports sel = mean(ds,
dla) cell means, al, and per-participant dla deltas. Selection on sel only.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts"), str(ROOT / "scripts/diag")]
from diag_common import R15, cm, d, load, mixp, rebal  # noqa: E402
from eval_external_teacher import by_participant  # noqa: E402
from mix_probs import parse  # noqa: E402
import session_em as SE  # noqa: E402

PROTOS = ("dev_loso_all", "dev_session", "all_loso")


def variant(ops):
    specs = list(R15)
    for op in ops:
        if op.startswith("+"):
            specs.append(op[1:])
        else:
            old, new = op.split("~")
            i = [parse(s)[0] for s in specs].index(old)
            w = parse(specs[i])[2]
            specs[i] = new if "*" in new else f"{new}*{w:g}"
    return specs


def r19e(B, proto, specs):
    _, Ps, ws, idx, seeds = load(proto, specs)
    GE = np.c_[B["G"], B["EV"]]
    e = SE.apply(GE[idx], rebal(mixp(Ps, ws), idx), d["stream"][idx], lam=0.5, n0=60, m0=20)
    return idx, e, seeds


def main():
    B, out = np.load(sys.argv[1]), Path(sys.argv[2])
    variants = {"R15": []} | {k: v.split(",") for k, v in (a.split("=", 1) for a in sys.argv[3:])}
    res, ref = {}, {}
    for name, ops in variants.items():
        try:
            specs = variant(ops)
            r = {"specs": specs}
            for proto in PROTOS:
                idx, e, seeds = r19e(B, proto, specs)
                r[proto] = round(cm(idx, e.argmax(1)), 2)
                pp = by_participant(idx, e.argmax(1))
                if name == "R15":
                    ref[proto] = pp
                    k = f"s05_{proto}_R19e"
                    if k in B.files and len(B[k]) == len(e):
                        r[f"{proto}_matches_cache"] = bool(np.allclose(B[k], e, atol=1e-6))
                else:
                    r[f"{proto}_per_participant"] = {p: round(pp[p] - ref[proto][p], 2) for p in pp}
                r[f"{proto}_seeds"] = {n: s for n, s in seeds.items()} if name != "R15" else None
            r["sel"] = round((r["dev_loso_all"] + r["dev_session"]) / 2, 2)
        except (AssertionError, ValueError, KeyError) as ex:  # missing probabilities
            print(name, "skipped:", ex, flush=True)
            continue
        if name != "R15":
            r["sel_delta"] = round(r["sel"] - res["R15"]["sel"], 2)
        res[name] = r
        print(name, "sel", r["sel"], "delta", r.get("sel_delta"), "dla", r["dev_loso_all"],
              "ds", r["dev_session"], "al", r["all_loso"],
              "cache" if name == "R15" else "", {k: v for k, v in r.items() if k.endswith("cache")}, flush=True)
    out.write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
