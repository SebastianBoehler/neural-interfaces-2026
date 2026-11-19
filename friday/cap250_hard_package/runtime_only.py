"""Measure exact signed decisions and warm time on two frozen package fixtures."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from qualify import ARM_ORDER, LOCK, STOCK, fixtures, load, timed_run
from settings import sha


ARCHIVE = "6a569180d343c038558fe7eccb7903a36e2bdf694af0df3687bff3b17049fbcc"
PROOFS = (
    ("diagnosis_0.json", "ba3cc6f6d9bdf373ea065e76c293b2b7bee46b281ca729d71a12259e24e9a190",
     "012cb9f01cb88baa0cbfdbc311bf3470c4107d5759fc35006cf926d3ae3d8ce2"),
    ("smoke_1.json", "e1b650082d00b0577453a4c4abb7e378b6993309d0c199c324878ba355d1d808",
     "69abae0f51fa89f6dd420dad8b863d70ba530638288575e34f7c7a5fb2706489"),
)


def pinned(cases):
    checked = []
    for index, (name, (file, digest, source)) in enumerate(zip(ARM_ORDER, PROOFS, strict=True)):
        if sha(file) != digest:
            raise ValueError(f"the pinned fixture proof changed: {name}")
        pin = json.loads(Path(file).read_text())
        actual = pin["result"]
        _, rows, _, stock, hard, signed = cases[name]
        native, resets = actual["native"], actual["resets"]
        if (pin["source"] != source or pin["archive"] != ARCHIVE or pin["stock"] != STOCK
                or pin["actors"] != LOCK or pin["held_labels_opened"]
                or actual["fixture"] != name or actual["independent_diagnostic_only"] != (index == 0)
                or not actual["cold_g_exact_archive"] or actual["rows"] != rows.tolist()
                or actual["stock"] != stock.tolist() or actual["hard"] != hard.tolist()
                or actual["stock_work"] != signed["activity"] or native["calls"] != len(rows)
                or not all(native[k] for k in ("sha_pupils_exact", "exact_private_tail",
                                               "exact_stock_and_candidate_native_inputs", "exact_same_input_prefix"))
                or set(resets) != {"active_copy", "reused"}
                or any(not (v["exact_cpu_actor_reset"] and v["labels_exact"] and v["native_rows"] == 17)
                       for v in resets.values())):
            raise ValueError(f"the signed fixture lacks an exact causal prerequisite: {name}")
        checked.append(actual)
    return checked


def paired(stock, trial, case, check, index):
    windows, _, pupils, _, _, _ = case
    templates = {"stock": stock, "hard": trial}

    def play(arm):
        changed = arm == "hard"
        result = timed_run(templates[arm], windows, changed, expected=pupils if changed else None)
        if (result["labels"] != check[arm] or result["work"] != check[arm + "_work"]
                or result["exact_cpu_pupils"] != changed):
            raise ValueError(f"the timed {arm} native Solver changed a signed decision, pupil or counter")
        del result["labels"]
        return result

    warm = {arm: play(arm) for arm in ("stock", "hard")}
    order = ("stock", "hard", "hard", "stock") if index == 0 else ("hard", "stock", "stock", "hard")
    runs = [play(arm) for arm in order]
    st, hd = (np.array([r["seconds"] for a, r in zip(order, runs, strict=True) if a == arm])
              for arm in ("stock", "hard"))
    mean, worst = float(hd.mean() / st.mean()), float((hd / st).max())
    return dict(fixture=ARM_ORDER[index], order=order, warmups=warm, runs=runs, stock_mean=float(st.mean()),
                hard_mean=float(hd.mean()), ratio=mean, paired=(hd / st).tolist(), worst=worst,
                meets_timing=mean <= 1.12 and worst <= 1.15)


def main(args):
    if torch.__version__ != "2.14.1+cu130" or "RTX 2080 Ti" not in torch.cuda.get_device_name():
        raise ValueError("pinned clean RTX image required")
    torch.set_num_threads(1)
    if sha(args.trial) != ARCHIVE:
        raise ValueError("runtime archive differs from the signed diagnostic")
    cases, channels, original = fixtures()
    controls = pinned(cases)
    with TemporaryDirectory() as root:
        stock, trial = load(args.stock, args.trial, Path(root), channels)
        result = paired(stock, trial, cases[ARM_ORDER[args.index]], controls[args.index], args.index)
    path = Path(f"runtime_only_{args.index}.json")
    with path.open("x") as file:
        json.dump(dict(source=sha("source.sha256"), archive=ARCHIVE, stock=STOCK, actors=LOCK,
                       original_external_cold=original, proofs={p[0]: p[1] for p in PROOFS},
                       held_labels_opened=False, result=result), file, sort_keys=True, allow_nan=False)
        file.write("\n")
    path.chmod(0o444)
    print("SEALED F-C25pt", args.index, "MEETS" if result["meets_timing"] else "MISSES", sha(path), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=int, choices=(0, 1))
    parser.add_argument("stock", type=Path)
    parser.add_argument("trial", type=Path)
    main(parser.parse_args())
