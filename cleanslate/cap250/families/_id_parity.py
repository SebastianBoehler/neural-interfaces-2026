"""Compare a hilbert/identity fit with the sealed archive, or prove that held-out people never reach it.

  parity PART_WEIGHTS ARCHIVE_WEIGHTS PREFIX...   per-key bitwise / max-abs report for keys starting with the prefixes
  exposure FAMILY PEOPLE FIRST DATA WORKDIR       fit twice, second time with every non-source row replaced by noise
"""

import subprocess
import sys
from pathlib import Path

import numpy as np

FAM = Path(__file__).resolve().parent


def parity(part, archive, prefixes):
    a, b = np.load(part), np.load(archive)
    keys = sorted(k for k in b.files if k.startswith(tuple(prefixes)))
    extra = sorted(set(k for k in a.files) - set(keys))
    worst, bad = 0.0, []
    for k in keys:
        if k not in a.files:
            bad.append((k, "missing"))
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape or x.dtype != y.dtype:
            bad.append((k, f"{x.shape}/{x.dtype} vs {y.shape}/{y.dtype}"))
            continue
        same = np.array_equal(x, y)
        d = 0.0 if same else float(np.max(np.abs(x.astype(np.complex128) - y.astype(np.complex128))))
        scale = float(np.max(np.abs(y))) or 1.0
        print(f"{k:16s} {'BITWISE' if same else f'max|d|={d:.3e} rel={d / scale:.3e}'}")
        worst = max(worst, d / scale)
    print("keys", len(keys), "extra", extra, "bad", bad, "worst_rel", worst)
    return not bad and not extra


def exposure(family, people, first, data, work):
    work = Path(work)
    with np.load(data, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    person = np.array([str(s).rsplit("/", 1)[-1] for s in arrays["subject"]])
    session = arrays["session"].astype(str)
    keep = np.isin(person, people.split(",")) & ~(np.isin(person, first.split(",") if first else []) & (session != "1"))
    rng = np.random.default_rng(0)
    x = arrays["X"].copy()
    x[~keep] = rng.standard_normal(x[~keep].shape).astype(x.dtype) * 50
    arrays["X"] = x
    arrays["y"] = np.where(keep, arrays["y"], rng.integers(0, 3, len(keep)).astype(arrays["y"].dtype))
    np.savez(work / "perturbed.npz", **arrays)
    out = {}
    for tag, path in (("real", data), ("perturbed", work / "perturbed.npz")):
        cmd = [sys.executable, "-B", str(FAM / f"{family}.py"), "train", "--people", people, "--data", str(path),
               "--seed", "0", "--out", str(work / tag)]
        if first:
            cmd += ["--first-session-only", first]
        subprocess.run(cmd, check=True)
        out[tag] = np.load(work / tag / "weights.npz")
    ok = set(out["real"].files) == set(out["perturbed"].files) and all(
        np.array_equal(out["real"][k], out["perturbed"][k]) for k in out["real"].files)
    print(f"EXPOSURE {family} {people} first={first}: dropped rows {int((~keep).sum())}, outputs identical: {ok}")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "parity":
        sys.exit(0 if parity(sys.argv[2], sys.argv[3], sys.argv[4:]) else 1)
    sys.exit(0 if exposure(*sys.argv[2:7]) else 1)
