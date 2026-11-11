"""Build a routing-v4 package: copy a route3geb package, add the identity-gate arrays (bio_*) and the v4 code.

Usage: python scripts/add_bio_gate.py SRC_DIR OUT_DIR MODEL.npz tau=0.25 [ramp=0.05] [t0=10]   (tau is required)
MODEL.npz is the external two-covariance model (experiments/bio_id.py, BIO_EXPORT). The galleries are the public
PROTEUS sessions of the ten people in `route_names` (the order of the personal heads). Every other file and array is
copied unchanged; submission.py becomes submissions/ensemble_v4/submission.py.
"""
import hashlib, shutil, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from bio_gate import Bio, fold_params  # noqa: E402


def main():
    src, out, model = Path(sys.argv[1]), Path(sys.argv[2]), dict(np.load(sys.argv[3]))
    kw = {k: float(v) for k, v in (a.split("=") for a in sys.argv[4:])}
    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name not in ("weights.npz", "submission.py") and f.is_file():
            shutil.copy2(f, out / f.name)
    shutil.copy2(ROOT / "submissions/ensemble_v4/submission.py", out / "submission.py")
    z = dict(np.load(src / "weights.npz", allow_pickle=False)); assert not any(k.startswith("bio_") for k in z)
    d = np.load(ROOT / "data/proteus_windows.npz", allow_pickle=True)
    sub = (ROOT / "submissions/ensemble_v4/submission.py").read_text()
    assert all(f'"{c}"' in sub or f"'{c}'" in sub for c in map(str, d["ch_names"])), "channel names differ from the submission's CH_NAMES"
    assert len(z["route_names"]) == len(z["route_ps1_W"]), "gallery order must follow the personal heads"
    per = np.array([s[-3:] for s in d["subject"]]); stream = np.char.add(np.char.add(per, "|"), d["session"].astype(str))
    b0 = Bio.__new__(Bio); b0.win = np.hanning(120); feats = np.stack([b0.feat(x) for x in d["X"]])
    p = fold_params(model, feats, stream, [str(n) for n in z["route_names"]], tau=kw["tau"], ramp=kw.get("ramp", 0.05), t0=int(kw.get("t0", 10)))
    z.update({k: np.asarray(v) for k, v in p.items()})
    np.savez(out / "weights.npz", **z)
    print("gallery sessions per person", dict(zip(z["route_names"], p["bio_m"])), "| tau", p["bio_tau"], "ramp", p["bio_ramp"], "t0", p["bio_t0"])
    for f in sorted(out.iterdir()):
        if f.name in ("weights.npz", "submission.py"):
            print(f.name, hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_size)


if __name__ == "__main__":
    main()
