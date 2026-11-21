"""Collect paired live predictions with Lane TB's official cell weights and intervals."""
import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "laneTB"))
import verify_collect as V
from confirm_collect import pooled_dir
from dose import DOSE_STARTS

DOSE_PAIRS = (("nine25", "base"), ("nine15", "base"), ("nine25", "nine15"))


def load(root, arms, orders, data):
    V.C.ORDERS = tuple(orders)
    receipts = {}
    with TemporaryDirectory() as tmp:
        merged = Path(tmp)
        for mode, person in V.C.FOLDS:
            fold = f"{mode}_{person}"
            receipts[fold] = {}
            for arm in arms:
                target = merged / fold / arm
                target.mkdir(parents=True)
                streams, seen = {}, set()
                for order in orders[::2]:
                    with np.load(root / fold / arm / f"pred_{fold}_{order}.npz", allow_pickle=False) as z:
                        seen.add(str(z["receipt"]))
                        for key in z.files:
                            if key == "receipt":
                                continue
                            if key in streams:
                                raise ValueError(f"duplicate {fold} {arm} {key}")
                            streams[key] = z[key]
                if len(seen) != 1:
                    raise ValueError(f"changed provenance for {fold} {arm}")
                receipts[fold][arm] = json.loads(seen.pop())
                np.savez(target / f"pred_{fold}.npz", **streams, source_sha="merged", weights_sha="merged")
        return {arm: V.C.load_arm(merged, arm, data)[0] for arm in arms}, receipts


def restrict(cells, orders):
    return {arm: {k: v for k, v in rows.items() if k[1] in orders} for arm, rows in cells.items()}


def contrasts(cells, pairs, orders):
    comparisons = {f"{a} - {b}": V.contrast(cells[a], cells[b], orders) for a, b in pairs}
    for result in comparisons.values():
        del result["proxy_57_43_unverified"]
    return comparisons


def absolute(cells, orders):
    V.C.ORDERS = orders
    out = {}
    for arm, rows in cells.items():
        new, counts = V.C.table(rows, "new", V.D.PEOPLE)
        ret, ret_counts = V.C.table(rows, "ret", V.D.RETURNING)
        out[arm] = dict(new=float(np.average(new, axis=1, weights=counts).mean()),
                        returning=float(np.average(ret, axis=1, weights=ret_counts).mean()),
                        people=dict(zip(V.D.RETURNING, ret.mean(axis=0).tolist(), strict=True)))
    return out


def dose(root, data, orders):
    arms = ("base", "nine15", "nine25")
    cells, receipts = load(root, arms, orders, data)
    for fold, by_arm in receipts.items():
        for field in ("data_sha", "hard_sha", "weights_sha", "train_streams"):
            if any(by_arm[a][field] != by_arm["base"][field] for a in arms[1:]):
                raise ValueError(f"incompatible receipts for {fold}: {field}")
        if by_arm["nine15"]["bank_sha"] != by_arm["nine25"]["bank_sha"]:
            raise ValueError(f"the two doses used different banks for {fold}")
    blocks = {f"{orders[0]}–{orders[-1]}": orders}
    if len(orders) == 32:
        blocks.update({f"{part[0]}–{part[-1]}": part for part in (orders[:16], orders[16:])})
    findings = {}
    for name, part in blocks.items():
        selected = restrict(cells, part)
        findings[name] = dict(absolute=absolute(selected, part), contrasts=contrasts(selected, DOSE_PAIRS, part))
    return dict(blocks=findings, receipts=receipts)


def print_contrasts(results):
    print("| Contrast | Held-new Δ [80%] | Returning Δ [80%] | P04 / P10 | 50/50 Δ [80%] |")
    print("|---|---|---|---|---|")
    for name, result in results.items():
        n, r = result["new"], result["returning"]
        p4, p10 = (r["people"][p]["mean"] for p in ("P04", "P10"))
        def summary(mean, ci):
            return f"{mean:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"
        print(f"| {name} | {summary(n['mean'], n['ci80'])} | {summary(r['mean'], r['ci80'])} | "
              f"{p4:+.2f} / {p10:+.2f} | {summary(result['j'], result['j_ci80'])} |")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--json", type=Path, required=True)
    p.add_argument("--half", type=int, choices=(1, 2))
    p.add_argument("--start", type=int, choices=DOSE_STARTS, default=DOSE_STARTS[0])
    p.add_argument("--prior-root", type=Path, help="original 71400 block for the pooled 64-order report")
    a = p.parse_args()
    orders = tuple(range(a.start, a.start + 32))
    if a.half is not None:
        orders = orders[16 * (a.half - 1):16 * a.half]
    data = V.D.load(a.data)
    report = dose(a.root, data, orders)
    if a.prior_root is not None:
        if a.start != DOSE_STARTS[1] or a.half is not None:
            p.error("--prior-root requires the full follow-up block")
        old = tuple(range(DOSE_STARTS[0], DOSE_STARTS[0] + 32))
        with TemporaryDirectory() as scratch:
            merged = pooled_dir((a.prior_root.resolve(), a.root.resolve()), scratch)
            pooled = dose(merged, data, old + orders)["blocks"].popitem()[1]
        report["blocks"]["71400–71431 and 71600–71631"] = pooled
    a.json.write_text(json.dumps(report, indent=1) + "\n")
    for name, block in report["blocks"].items():
        print(name)
        print_contrasts(block["contrasts"])
        print("Absolute official cell BA:", block["absolute"])


if __name__ == "__main__":
    main()
