"""Third form (ledger s256): tttssl2's C10 (joint SimCLR at weight 0.1, adaptation sB40: SessionBN affine only, 5e-3, 40
full-batch steps on the whole held-out session; non-causal ceiling) on a larger trunk, FBCNet on raw input
(ttt5's "fbraw": fbcnet_raw_sbn, k 20, no pretrained trunk). Twin: the same recipe with 3 outputs (na only).
Everything else is tttssl.run / tttssl2.make unchanged; outputs runs/tttssl/probs_fbraw/tttssl_<form>_<arm>-s<seed>.npz,
scored with FORMS=twin,C10 python studies/14nov/tttssl/score.py runs/tttssl/probs_fbraw.
Usage (Galvani): python studies/14nov/tttssl/third.py TASK (form = (twin, C10)[TASK % 2], seed = TASK // 2) | smoke"""
import importlib.util, sys
from pathlib import Path
_s = importlib.util.spec_from_file_location("tttssl2", Path(__file__).resolve().parent / "tttssl2.py")
T2 = importlib.util.module_from_spec(_s); _s.loader.exec_module(T2); T = T2.T; SB = T.SB
T.CFG, T.K, T.OUT = "fbcnet_raw_sbn", 20.0, T.WS / "runs/tttssl/probs_fbraw"; FORMS = ("twin", "C10")
T.ARMS["twin"] = {"na": None}; T.ARMS["C10"] = {"na": None, "sB40": T2.ARMS2["sB40"]}
_make = T2.make


def make(form, d, seed, mir):
    if form == "twin":
        m = SB.make(T.CFG, d, seed); assert type(m) is SB.SessionNet and not m.aug and m.init is None and not m.net_kw; return m
    return _make(form, d, seed, mir)


T.make = make; assert T.run.__globals__["make"] is make and T.CFG == "fbcnet_raw_sbn" and T.run.__globals__["K"] == 20.0


def main():
    d = T.protocol.load(); T.OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "smoke":
        (T.OUT.parent / "smoke3").mkdir(exist_ok=True)
        for form in FORMS:
            T.LAM = T2.LAMS["C10"]; T.run(form, d, 0, T.OUT.parent / "smoke3", smoke=True)
        return
    task = int(sys.argv[1]); form = FORMS[task % 2]; T.LAM = T2.LAMS["C10"]; print(f"form {form} cfg {T.CFG} k {T.K} LAM {T.LAM}", flush=True)
    assert T.JointNet._train.__globals__["LAM"] == 0.1; T.run(form, d, task // 2, T.OUT)


if __name__ == "__main__":
    main()
