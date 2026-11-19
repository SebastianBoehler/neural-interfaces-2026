"""Self-check of niv/enrol.py: rebuild the rows of single-session public people from their own session and compare
with the rows stored in the package (stored rows come from the original builders).  python niv/check_enrol.py ZIP [P01 ...]"""
import io
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2026-competition/tracks/bci_decoding"))
import enrol  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
zip_path, people = sys.argv[1], sys.argv[2:] or ["P01"]
d = np.load(ROOT / "data/proteus_windows.npz", allow_pickle=True)
with TemporaryDirectory() as tmp, zipfile.ZipFile(zip_path) as zf:
    zf.extractall(tmp)
    z = dict(np.load(Path(tmp) / "weights.npz", allow_pickle=False))
    names = [str(n) for n in z["route_names"]]
    for p in people:
        rows_idx = np.flatnonzero((np.char.endswith(d["subject"].astype(str), p)) & (d["session"].astype(str) == "1"))
        rows = enrol.person_rows(tmp, z, d["X"][rows_idx], d["y"][rows_idx])
        i = names.index(p)
        for k, v in rows.items():
            if k == "route_names" or k not in z:
                continue
            ref, new = np.asarray(z[k])[i].astype(float), np.asarray(v)[0].astype(float)
            cos = float(ref.ravel() @ new.ravel() / (np.linalg.norm(ref) * np.linalg.norm(new) + 1e-12))
            print(f"{p} {k:12s} max|d| {np.abs(ref - new).max():.3e}  rel {np.linalg.norm(ref - new) / (np.linalg.norm(ref) + 1e-12):.3e}  cos {cos:.6f}", flush=True)
