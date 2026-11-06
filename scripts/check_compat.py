"""Check that the current submissions/ensemble/submission.py reproduces existing packages exactly (5 Nov, SJit).

Usage: python scripts/check_compat.py PKG_DIR [PKG_DIR ...] [--mix] [--out JSON]
Each package is streamed per session over all public windows, as scripts/set_params.py does: once with its own
submission.py and once with the current code (copied with the package's data files into a temporary folder).
Predicted labels must be byte-identical; with --mix the pre-rebalancing member mix too (np.array_equal).
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import protocol  # noqa: E402
from add_rebalance import SRC, load_solver, stream  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkgs", nargs="+", type=Path)
    ap.add_argument("--mix", action="store_true")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    d = protocol.load()
    rows = np.ones(len(d["y"]), bool)
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    res = {"new_code_sha256": sha(SRC), "n_windows": int(rows.sum())}
    for pkg in a.pkgs:
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            new = Path(tmp)
            for f in pkg.iterdir():
                if f.is_file() and f.name != "submission.py":
                    shutil.copyfile(f, new / f.name)
            shutil.copyfile(SRC, new / "submission.py")
            fns = {"labels": lambda m, x: m.predict(x)}
            if a.mix:
                fns["mix"] = lambda m, x: m.mixed_proba(x)
            r = {"package_code_sha256": sha(pkg / "submission.py")}
            for what, fn in fns.items():
                old_out = stream(load_solver(pkg, f"old_{what}"), d, rows, fn)
                new_out = stream(load_solver(new, f"new_{what}"), d, rows, fn)
                r[f"{what}_byte_identical"] = bool(np.array_equal(old_out, new_out))
                diff = old_out != new_out
                r[f"{what}_n_windows_differ"] = int((diff.any(-1) if diff.ndim > 1 else diff).sum())
            r["seconds"] = round(time.time() - t0, 1)
        res[str(pkg)] = r
        print(pkg, json.dumps(r), flush=True)
    if a.out:
        a.out.write_text(json.dumps(res, indent=1))
    ok = all(v for p in a.pkgs for k, v in res[str(p)].items() if k.endswith("_byte_identical"))
    print("ALL BYTE-IDENTICAL" if ok else "MISMATCH", flush=True)
    assert ok


if __name__ == "__main__":
    main()
