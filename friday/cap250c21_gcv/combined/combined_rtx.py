"""Prove the combined native selector on RTX and time fresh paired sessions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import replay
from combined_selector import SharedCovRetiringSelector
from gate_rtx import equal_prefix, g_prefix, live
from native21 import stream
from retire_rtx import proof
from scope import external, known
from settings import sha
from worker import authentic_stock


SIGNATURES = ("rows", "recognized", "eligible", "alternate_rows", "alternate_refits", "alternate_fused",
              "alternate_fit_ages", "retired_at", "shared_calls")


def receipt(template, trial, eeg, rows, expected, age, eligible):
    output = proof(template, trial, eeg, rows, expected, age, eligible)
    gate = trial[1]
    shares = len(rows) if age < 0 else age
    counts = output["alternate"]
    if counts["shared_calls"] != shares or counts["verified_covariances"] != shares:
        raise ValueError("exact stock covariance reuse count differs")
    stock, alternate = gate.model.cp, gate.outer.cp
    if len(alternate.C) != shares:
        raise ValueError("alternate covariance retention differs from the live sharing count")
    if age < 0:
        for a, b in zip(stock.C, alternate.C, strict=True):
            np.testing.assert_array_equal(a, b)
            if np.shares_memory(a, b):
                raise ValueError("alternate covariance aliases stock")
    elif stock.C or stock.P or stock.model is not None:
        raise ValueError("the original stock recognizer must clear its native CSP history")
    return output


def smoke(template, eeg, who, orders, channels, anchor):
    rows = orders[0][who+"|1"]
    cold = stream(template, eeg, rows, "stock")
    authentic_stock(cold, anchor)
    g = live(template, eeg, rows, audit=True, selector=SharedCovRetiringSelector)
    unknown = receipt(template, g, eeg, rows, anchor["labels"], -1, 100)
    authentic_stock((dict(unknown, labels=anchor["labels"]),None), anchor)
    info = unknown["alternate"]
    if (info["alternate_refits"], info["alternate_fused"], info["hook"]["warm_rows"]) != (5,30,100):
        raise ValueError("genuine unknown G warm alternate work differs")
    px, p, expected = known(channels)
    recognized = live(template, px, p, audit=True, selector=SharedCovRetiringSelector)
    p10 = receipt(template, recognized, px, p, expected, 10, 0)
    if p10["route"]["routed"] != 21 or p10["selected_changes"] != 0:
        raise ValueError("actual sticky recognition fixture did not protect stock")
    live(template, eeg, orders[0][who+"|2"][:17], selector=SharedCovRetiringSelector)
    repeated = stream(template, eeg, rows[:31], "stock")
    short = live(template, eeg, rows[:31], True, SharedCovRetiringSelector)
    prefix = g_prefix(template, cold, g[1], short[1], repeated)
    equal_prefix(recognized[1], live(template, px, p[:17], True, SharedCovRetiringSelector)[1], 17)
    return dict(unrecognized=unknown, known=p10, unrecognized_prefix=prefix, known_prefix_exact=True)


def timing(template, eeg, who, orders, channels, index):
    proof = json.loads(Path("smoke.json").read_text())
    delayed = json.loads(Path("delayed.json").read_text())
    if (any(p["source"] != sha("source.sha256") or not p["passed"] or p["held_labels_opened"] for p in (proof,delayed))
            or not delayed["exact_dual_emission"] or delayed["eligible_ages"] != list(range(20,28))
            or delayed["alternate"]["verified_covariances"] != 28):
        raise ValueError("the distinct signed covariance and exact retirement scopes must pass before timing")
    if index == 0:
        x, rows = eeg, orders[0][who+"|1"]
        name, order = "unrecognized", ("stock", "combined", "combined", "stock")
    else:
        x, rows, _ = known(channels)
        name, order = "known", ("stock", "full", "combined", "combined", "full", "stock")
    wanted = proof[name]
    warm = stream(template, x, rows, "stock", inspect=False)[0]
    runs = []
    for arm in order:
        if arm == "stock":
            r = stream(template,x,rows,"stock",inspect=False)[0]
        elif arm == "full":
            r = live(template,x,rows)[0]
            if r["alternate"]["alternate_rows"] != len(rows):
                raise ValueError("timed full dual did not advance all alternate rows")
        else:
            r = live(template,x,rows,selector=SharedCovRetiringSelector)[0]
            if (any(r["alternate"][k] != wanted["alternate"][k] for k in SIGNATURES)
                    or r["alternate"]["verified_covariances"] != 0):
                raise ValueError("timed covariance sharing or recognition retirement differs")
        if (r["activity"] != warm["activity"] or r["activity"] != wanted["activity"]
                or r["labels"] != (warm["labels"] if arm == "stock" else wanted["labels"])):
            raise ValueError("timed combined path changed genuine work or selected decisions")
        runs.append(r)
    t = {k: np.asarray([r["seconds"] for r, a in zip(runs,order,strict=True) if a == k]) for k in dict.fromkeys(order)}
    paired = t["combined"] / t["stock"]
    ratio, worst = float(t["combined"].mean()/t["stock"].mean()), float(paired.max())
    out = dict(fixture=name, stock_warm=warm, order=order, runs=runs, ratio=ratio, paired=paired.tolist(),
               worst=worst, meets_timing=ratio <= 1.12 and worst <= 1.15)
    if index == 1:
        out["combined_vs_full"] = float(t["combined"].mean()/t["full"].mean())
    return out


def main(mode, index):
    if torch.__version__ != "2.14.1+cu130" or "RTX 2080 Ti" not in torch.cuda.get_device_name():
        raise ValueError("pinned RTX worker is required")
    eeg, who, channels, orders, anchor, archive = external(5)
    template = replay.load(Path("/package"), channels)
    if mode == "smoke":
        result = smoke(template, eeg, who, orders, channels, anchor)
    else:
        result = timing(template, eeg, who, orders, channels, index)
    path = Path("smoke.json" if mode == "smoke" else f"timing_{index}.json")
    with path.open("x") as f:
        json.dump(dict(result, passed=True, source=sha("source.sha256"), archive=archive,
                       device=torch.cuda.get_device_name(), held_labels_opened=False),
                  f, sort_keys=True, allow_nan=False)
        f.write("\n")
    path.chmod(0o444)
    print("PASS F-C21gct", mode, index, sha(path), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("smoke", "timing"))
    p.add_argument("index", type=int, choices=(0,1))
    opts = p.parse_args()
    main(opts.mode, opts.index)
