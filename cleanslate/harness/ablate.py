"""Component ablation of a team system through the same harness (stdlib + compare.py).

Teams expose each component as an env switch read by THEIR train.py (and written into the package config; see
cleanslate/reference/train.py: CS_REF_EA, CS_REF_ADABN, CS_REF_PRIOR). Switching one off retrains every fold.
  submit: python ablate.py submit --team reference --off CS_REF_EA=0 --orders 71000,71001 [--base-tag T]
          -> run_lopo.py with --env, tag abl_<NAME>_<VAL>_<orders>  (same orders and rosters as the full-system run)
  report: python ablate.py report RUN_FULL RUN_ABLATED     -> compare.py FULL - ABLATED: the component's paired delta
          (positive = the component helps), people+orders bootstrap 80 % interval, held-new / returning (P04,P10 only) / sealed 0.57-0.43.
Whole-system rule: the delta is measured inside the full system, never alone; the interval is the reading aid, not a gate.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv):
    if len(argv) < 2 or argv[0] not in ("submit", "report"):
        sys.exit(__doc__)
    if argv[0] == "report":
        sys.exit(subprocess.call([sys.executable, str(HERE / "compare.py"), *argv[1:]]))
    rest = argv[1:]
    offs = [rest[i + 1] for i, a in enumerate(rest) if a == "--off"]
    assert len(offs) == 1 and "=" in offs[0], "give exactly one --off NAME=VAL"
    rest = [a for i, a in enumerate(rest) if a != "--off" and (i == 0 or rest[i - 1] != "--off")]
    orders = rest[rest.index("--orders") + 1] if "--orders" in rest else "71000,71001"
    tag = "abl_" + offs[0].replace("=", "") + "_o" + orders.replace(",", "_")
    sys.exit(subprocess.call([sys.executable, str(HERE / "run_lopo.py"), *rest, "--env", offs[0], "--tag", tag]))


if __name__ == "__main__":
    main(sys.argv[1:])
