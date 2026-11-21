"""Build a signed final-stack derivative by changing one live setting."""
import argparse
from functools import partial
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "submissions/final_stack"))
from build_cb_speed import build  # noqa: E402
from patch_ba_six_fc import LEAD_SHA as NINE_SHA, once  # noqa: E402

BA_SHA = "fc2acf89bc194e7e576dd9fa9b6a6858207d06fe98396a1e4e1f785c04e9a5aa"
PARENTS = {"nine25": NINE_SHA, "ba_six": BA_SHA}
WEIGHTS = (0.25, 0.30, 0.35, 0.40, 0.50)
DOSES = {f"w{round(w * 100)}": w for w in WEIGHTS}
STOCK = (1.15, 5.0, 10)
EARLY = {
    "rb_alpha": (0.575, 5.0, 10),
    "rb_mass": (1.15, 20.0, 10),
    "em_delay": (1.15, 5.0, 40),
}
ROUTE = "        self.tail.em.extra = np.asarray(extra, dtype=np.float64)\n        self.contrast.next, self.identity.next"


def parameters(tail):
    return tail.rb.alpha, tail.rb.k, tail.em.nmin


def patch(source, arm):
    if arm in DOSES:
        text = once(source, "    TEAM_B_MIX = 0.25\n", f"    TEAM_B_MIX = {DOSES[arm]}\n")
    elif arm in EARLY:
        schedule = (
            "        if not self.U:\n"
            f"            self.tail.rb.alpha, self.tail.rb.k, self.tail.em.nmin = {EARLY[arm]!r}\n"
            "        elif len(self.U) == 40:\n"
            f"            self.tail.rb.alpha, self.tail.rb.k, self.tail.em.nmin = {STOCK!r}\n"
        )
        text = once(source, ROUTE, schedule + ROUTE)
    else:
        raise ValueError(f"unregistered arm: {arm}")
    compile(text, "submission.py", "exec")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roster", choices=tuple(PARENTS))
    parser.add_argument("arm", choices=tuple(DOSES) + tuple(EARLY))
    parser.add_argument("parent", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    build(args.parent, args.target, expected_sha=PARENTS[args.roster], patch_source=partial(patch, arm=args.arm))


if __name__ == "__main__":
    main()
