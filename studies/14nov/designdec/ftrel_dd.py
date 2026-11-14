"""DESIGNDEC (2) fine-tuning labels (ledger s239): studies/14nov/ftrelabel/ftrel.py unchanged (KIND v51b, arms s15 and r15),
with r15's per-refit labels replaced by the run-smoothed ones (labels.py -> runs/labels/relab_dd239_<order>.npz) and the
output tag ttt18<order>LPD, so s229's ttt18<order>LP files are never overwritten. s15 is the in-run control (labels as
emitted, bit-identical to s229's s15). Score: shufcombo FT_TAG=ttt18<order>LPD FT_V=r15 against FT_TAG=ttt18<order>LP FT_V=r15.
Usage (Leonardo array, from the work dir): SCHEME=within PSEED=0 python studies/14nov/designdec/ftrel_dd.py TASK | smoke"""
import importlib.util, os, sys
from pathlib import Path
WS = Path(os.environ.get("NI26_WS", Path(__file__).resolve().parents[3])); ORD = f"{os.environ['SCHEME']}{os.environ['PSEED']}"
os.environ.update(KIND="v51b", ARMS="s15,r15", RELAB_NPZ=str(WS / f"runs/labels/relab_dd239_{ORD}.npz"))
assert os.path.exists(os.environ["RELAB_NPZ"]), os.environ["RELAB_NPZ"]
_s = importlib.util.spec_from_file_location("ftrel", WS / "studies/14nov/ftrelabel/ftrel.py"); FR = importlib.util.module_from_spec(_s); _s.loader.exec_module(FR)
assert FR.T7.TAG == f"ttt18{ORD}LP", FR.T7.TAG
FR.T7.TAG += "D"; FR.T7.SMOKE_OUT += "dd"
if __name__ == "__main__":
    FR.main(sys.argv)
