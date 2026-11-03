"""Label agreement of two sealed packages through the full predict path (audit, 3 Nov).

Usage: python scripts/compare_packages.py NEW_DIR REF_DIR OUT_JSON   (CPU, about 10-20 min)
Streams every public window, session by session (state reset at session boundaries, as the
evaluator does), through each package's Solver.predict: members, teacher, rebalancing and the
R19e stage. Reports label agreement and the in-sample scores on the official test split and on
all windows. build_sealed.compare stops at the rebalanced member mix; this covers the rest.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402
from add_rebalance import load_solver, stream  # noqa: E402
from build_sealed import score, sha256  # noqa: E402


def main():
    new, ref, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    d = protocol.load()
    res = {"new": str(new), "reference": str(ref),
           "files_sha256": {k: {f.name: sha256(f) for f in sorted(p.iterdir()) if f.is_file()}
                            for k, p in (("new", new), ("reference", ref))}}
    for name, mask in (("official_test", d["split"] == "test"), ("all_windows", np.ones(len(d["y"]), bool))):
        a = stream(load_solver(new, "cmp_new"), d, mask, lambda m, x: m.predict(x))
        b = stream(load_solver(ref, "cmp_ref"), d, mask, lambda m, x: m.predict(x))
        res[name] = {"n_windows": int(mask.sum()), "label_agreement": float(np.mean(a == b)),
                     "n_disagree": int(np.sum(a != b)), "new": score(d, mask, a), "reference": score(d, mask, b)}
        print(name, json.dumps({k: v for k, v in res[name].items() if k in ("label_agreement", "n_disagree")}),
              flush=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
