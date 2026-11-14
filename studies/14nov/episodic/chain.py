"""EPISODIC, in the chain (ledger s220): shufcombo with the fbraw slot's frozen output replaced by the retrained family.
For each order the FT tag ttt18<order><ARM> is assembled in runs/episodic/t17 (fbraw: this study's files; fb, sinc, fbl:
links to the stock base of the same order, ttt7 for recording order) and scored with FT_V=base, so the arms are
  P0r  v5.1b, members and stages in the delivery order, slot 5 = the arm's family
  V6r  P0r + eye-net view + rest member (the v6 chain without fine-tuning)
against the same arms with the stock tag ttt18<order> (recording order: tag ttt7, i.e. the shipped slot).
Package gate (GATE_P0), eye net re-streamed (runs/ocushuf/<order>), recording-order base ttt7 (T6_DIR probs7).
Usage: python -I studies/14nov/episodic/chain.py [ARMS=E,E0] [ORDERS=within0,...]   (new people; PROTO=seen_session too)"""
import glob, os, subprocess, sys, time
from pathlib import Path
import numpy as np
WS = Path(__file__).resolve().parents[3]; EPI = WS / os.environ.get("EPI_DIR", "runs/episodic/probs"); T17 = WS / "runs/episodic/t17"
ARMS = os.environ.get("ARMS", "E,E0").split(","); ORDERS = os.environ.get("ORDERS", "rec0,within0,within1,full0,full1,runs0,runs1").split(",")
PROTO = os.environ.get("PROTO", "all_loso"); FAMS = ("fbraw", "fb", "sinc", "fbl"); BASES = ("B2_s012", "B2_s345", "B2_pool")
LOG = WS / "runs/episodic/chain_logs"; LOG.mkdir(parents=True, exist_ok=True); T17.mkdir(parents=True, exist_ok=True)


def link(o, arm):
    for proto in ("all_loso", "seen_session"):
        for f in FAMS:
            for s in range(6):
                src = (EPI / f"ttt18{o}{arm}_fbraw_base-{proto}-s{s}.npz" if f == "fbraw" else
                       WS / "runs/ttt/probs7" / f"ttt7_{f}_base-{proto}-s{s}.npz" if o == "rec0" else
                       WS / "runs/ttt/probs18" / f"ttt18{o}_{f}_base-{proto}-s{s}.npz")
                assert src.exists(), src
                dst = T17 / f"ttt18{o}{arm}_{f}_base-{proto}-s{s}.npz"
                if dst.is_symlink() or dst.exists():
                    dst.unlink()
                dst.symlink_to(src)


def run(o, tag, t17):
    sch, ps = o.rstrip("0123456789"), o[len(o.rstrip("0123456789")):]
    env = {**os.environ, "SCHEME": sch, "PSEED": ps, "FT_V": "base", "T6_DIR": str(WS / "runs/ttt/probs7"), "REC_PFX": "ttt7",
           "T17_DIR": str(t17), "FT_TAG": tag, "GATE_P0": "1", "PROTO": PROTO}
    env.pop("OCU_DIR", None)
    if sch != "rec":
        env["OCU_DIR"] = f"runs/ocushuf/{o}"
    pre = "" if PROTO == "all_loso" else PROTO + "_"; t0 = time.time() - 1
    with open(LOG / f"{pre}{o}_{tag}.log", "w") as fh:  # always rerun: an older file of the same tag may carry other options
        subprocess.run([sys.executable, "-I", str(WS / "studies/13nov/shufcombo.py")], env=env, cwd=WS, stdout=fh, stderr=subprocess.STDOUT, check=True)
    out = [f for f in glob.glob(str(WS / f"runs/shufcombo_{pre}{o}_*_{tag}.npz")) if os.path.getmtime(f) >= t0]
    assert len(out) == 1, out
    return np.load(out[0])


def main():
    D = {}
    for o in ORDERS:
        ref = run(o, "ttt7", WS / "runs/ttt/probs7") if o == "rec0" else run(o, f"ttt18{o}", WS / "runs/ttt/probs18")
        if o == "rec0":  # the stock reference in recording order streams the shipped base itself: P0r must equal P0rec
            assert all(np.allclose(ref[f"{bn}|P0r"], ref[f"{bn}|P0rec"]) for bn in BASES), "rec0 stock P0r != P0rec"
        for arm in ARMS:
            link(o, arm); z = run(o, f"ttt18{o}{arm}", T17)
            for bn in BASES:
                for a in ("P0r", "V6r"):
                    D[o, arm, bn, a] = z[f"{bn}|{a}"] - ref[f"{bn}|{a}"]
    print(f"\n== chain, {PROTO}: arm minus stock (rec0: stock = ttt7, the shipped base), person-equal; [w/o P10] (people up)")
    for arm in ARMS:
        for a in ("P0r", "V6r"):
            for bn in BASES:
                row = []
                for o in ORDERS:
                    v = D[o, arm, bn, a]
                    row.append(f"{o} {v.mean():+.2f}" + (f"[{np.delete(v, 9).mean():+.2f}]({int((v > 0).sum())})" if len(v) == 10 else ""))
                sh = [D[o, arm, bn, a] for o in ORDERS if o != "rec0"]
                m = np.mean(sh, 0) if sh else np.zeros(1)
                print(f"{arm} {a} {bn}: " + " ".join(row) + (f" | mean6 {m.mean():+.2f}" + (f" [w/o P10 {np.delete(m, 9).mean():+.2f}]" if len(m) == 10 else "") if sh else ""), flush=True)
    np.save(WS / f"runs/episodic/chain_{PROTO}.npy", {"|".join(k): v for k, v in D.items()}, allow_pickle=True)


if __name__ == "__main__":
    main()
