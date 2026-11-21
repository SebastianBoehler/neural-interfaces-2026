"""Collect paired PROTEUS and Scherer component scores from checked replays."""
import argparse
import json
from pathlib import Path

import numpy as np

from diet_decomposition_patch import ARMS
from returning_diet_report import ORDERS, digest, load_bank, metadata, report
from returning_em_niv import validate_pair


def proteus(data, baseline, captures):
    cases, files = load_bank(baseline, captures, metadata(data), "held", arms=ARMS)
    result = report(cases, ORDERS["held"], arms=ARMS, nested_orders=True)
    result.update(orders=ORDERS["held"], captures_sha256=files,
                  manifest_sha256=digest(baseline / "em_archive_manifest.json"))
    for arm in ARMS:
        row = result["arms"][arm]
        del row["passes_rule"]
        point, lo, _ = row["delta"]["J"]
        row["passes_proteus"] = bool(point >= 0.10 and lo > 0)
    return result


def nested(delta):
    rng = np.random.default_rng(19)
    norder, npeople = delta.shape
    people = rng.integers(npeople, size=(20_000, npeople))
    orders = rng.integers(norder, size=(20_000, npeople, norder))
    draw = delta[orders, people[:, :, None]].mean((1, 2))
    return np.r_[delta.mean(), np.quantile(draw, (0.1, 0.9))]


def scherer(data, baseline, captures):
    from niv import collect as niv
    from quarter_niv import FIRST

    for file in (baseline / "lead").glob("p*_c*.npz"):
        counts = validate_pair(file, baseline / "trial" / file.name)
        assert all(not (v["recognised"] or v["changed"]) for v in counts.values()), file
    new = niv.cohort_b_new(baseline, "lead", data)
    assert np.isfinite(new["cell"]).sum() == 288 and (new["rec"][np.isfinite(new["rec"])] == -1).all()
    base = niv.cohort_b_ret(baseline, "lead", data)
    assert base is not None and np.isfinite(base["cell"]).sum() == 144
    result = dict(orders=list(range(FIRST, FIRST + 8)), baseline=niv.absolute(base["cell"]), arms={})
    for arm in ARMS:
        arm_dir = captures / arm
        files = sorted(arm_dir.glob("ret_trial_k*/p*_c*.npz"))
        if len(files) != 9:
            continue
        for f in files:
            person = f.parent.name.removeprefix("ret_trial_")
            counts = validate_pair(baseline / f"ret_lead_{person}" / f.name, f)
            assert set(counts) == {"full", "within"} and all(
                v["streams"] == v["recognised"] == 8 for v in counts.values()
            ), f
        out = niv.cohort_b_ret(arm_dir, "trial", data)
        assert np.isfinite(out["cell"]).sum() == 144
        delta = niv.stats.person_table(out["cell"]) - niv.stats.person_table(base["cell"])
        returning = {s: nested(delta[i]) for i, s in enumerate(("full", "within"))}
        result["arms"][arm] = dict(
            returning={s: v.tolist() for s, v in returning.items()},
            joint={s: (v / 2).tolist() for s, v in returning.items()},
            absolute=niv.absolute(out["cell"]),
            passes_scherer=bool(all(v[0] >= -0.20 for v in returning.values())),
            captures_sha256={str(f.relative_to(arm_dir)): digest(f) for f in files},
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instrument", choices=("proteus", "scherer"))
    for name in ("data", "baseline", "captures", "out"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    collect = proteus if args.instrument == "proteus" else scherer
    result = collect(args.data, args.baseline, args.captures)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
