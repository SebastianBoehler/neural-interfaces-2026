"""Report the preregistered Lane TB-verify confirmation (71900–71915)."""
import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ORDERS = tuple(range(71900, 71916))
ARMS = ("base", "nine15", "nine25")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.run / "code/scripts/laneTB"))
    import verify_collect as verify

    data = verify.D.load(args.data)
    with TemporaryDirectory() as tmp:
        verify.merge(args.run / "out/hard", Path(tmp), ORDERS)
        cells = verify.load(Path(tmp), ORDERS, data)
    # A zero contrast through the same collector must be identically zero.
    zero = verify.contrast(cells["base"], cells["base"], ORDERS)
    if zero["j"] != 0 or zero["j_ci80"] != [0, 0]:
        raise ValueError(f"nonzero hard versus hard contrast: {zero}")

    verify.C.ORDERS = ORDERS
    rng = np.random.default_rng(20261121)
    order_draws = rng.integers(len(ORDERS), size=(20_000, len(ORDERS)))
    scores = {}
    for arm in ARMS:
        people_rng = np.random.default_rng(71120)
        new, new_draws = verify.C.by_cohort(cells[arm], "new", verify.D.PEOPLE, order_draws, people_rng)
        ret, ret_draws = verify.C.by_cohort(cells[arm], "ret", verify.D.RETURNING, order_draws, people_rng)
        scores[arm] = {
            "new": new, "returning": ret,
            "mean_50_50": (new["mean"] + ret["mean"]) / 2,
            "ci80_50_50": verify.C.interval((new_draws + ret_draws) / 2),
        }
    comparisons = (("nine25", "base"), ("nine15", "base"), ("nine25", "nine15"))
    contrasts = {}
    print("| Paired Δ pp | Held-new [80%] | Returning [80%] | P04 / P10 | 50/50 new/returning [80%] |")
    print("|---|---:|---:|---:|---:|")
    for arm, ref in comparisons:
        r = verify.contrast(cells[arm], cells[ref], ORDERS)
        new, ret = r["new"], r["returning"]
        pp = ret["people"]
        label = f"{arm} − {'hard' if ref == 'base' else ref}"
        print(f"| {label} | {interval(new['mean'], new['ci80'])} | {interval(ret['mean'], ret['ci80'])} | "
              f"{pp['P04']['mean']:+.2f} / {pp['P10']['mean']:+.2f} | {interval(r['j'], r['j_ci80'])} |")
        contrasts[f"{arm}_vs_{ref}"] = {
            "new": new, "returning": ret, "mean_50_50": r["j"], "ci80_50_50": r["j_ci80"],
        }
    result = dict(orders=ORDERS, scores=scores, contrasts=contrasts)
    out = args.run / "out/confirmation.json"
    pending = out.with_suffix(".tmp")
    pending.write_text(json.dumps(result, indent=1) + "\n")
    pending.replace(out)


def interval(mean, bounds):
    return f"{mean:+.2f} [{bounds[0]:+.2f}, {bounds[1]:+.2f}]"


if __name__ == "__main__":
    main()
