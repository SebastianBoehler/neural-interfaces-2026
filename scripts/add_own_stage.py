"""Build a v5 package: copy a routing-v4.1 package, add the own-space stage arrays (own_*) and the v5 code.

Usage: python scripts/add_own_stage.py SRC_DIR OUT_DIR beta=0.10
For each person in `route_names` (the order of the personal heads) the three own-space members of
studies/12nov/ownspace.py (ledger s37, s43, s45) are fitted on all public sessions of that person. Every other file
and array is copied unchanged; submission.py becomes submissions/ensemble_v5/submission.py.
"""
import hashlib, shutil, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "studies/12nov"))
import ownspace as O  # noqa: E402


def person_arrays(name):
    rows = [np.flatnonzero((O.PER == name) & (O.SES == s)) for s in sorted(set(O.SES[O.PER == name]))]
    m = O.Personal(rows); sc, lr = m.T.steps[0][1], m.T.steps[1][1]
    return dict(prior=np.stack(m.prior), csp=np.stack(m.W), t_mu=sc.mean_, t_sd=sc.scale_, t_W=lr.coef_, t_c=lr.intercept_,
                c_W=m.C.coef_, c_c=m.C.intercept_, s_W=m.S.coef_, s_c=m.S.intercept_), len(rows)


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2]); kw = {k: float(v) for k, v in (a.split("=") for a in sys.argv[3:])}
    code = ROOT / "submissions/ensemble_v5/submission.py"; out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name not in ("weights.npz", "submission.py") and f.is_file():
            shutil.copy2(f, out / f.name)
    shutil.copy2(code, out / "submission.py")
    z = dict(np.load(src / "weights.npz", allow_pickle=False)); assert "bio_V" in z and not any(k.startswith("own_") for k in z)
    sub = code.read_text(); d = np.load(ROOT / "data/proteus_windows.npz", allow_pickle=True)
    import ast
    chn = next(ast.literal_eval(n.value) for n in ast.parse(sub).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CH_NAMES")
    assert [str(c) for c in d["ch_names"]] == list(chn), "channel names or their order differ from the submission's CH_NAMES"
    names = [str(n) for n in z["route_names"]]; parts, nses = zip(*[person_arrays(n) for n in names])
    for k in parts[0]:
        z[f"own_{k}"] = np.stack([p[k] for p in parts]).astype(np.float64)
    z["own_bands"] = np.asarray(O.BANDS, dtype=np.float64); z["own_beta"] = np.float64(kw["beta"]); z["own_kref"] = np.float64(O.KREF)
    np.savez(out / "weights.npz", **z)
    print("public sessions per person", dict(zip(names, nses)), "| beta", kw["beta"], "| own arrays", {k: z[k].shape for k in z if k.startswith("own_")})
    for f in sorted(out.iterdir()):
        if f.name in ("weights.npz", "submission.py"):
            print(f.name, hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_size)


if __name__ == "__main__":
    main()
