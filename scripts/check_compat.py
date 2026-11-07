"""Check that the current submissions/ensemble/submission.py reproduces existing packages exactly (5 Nov, SJit).

Usage: python scripts/check_compat.py PKG_DIR [PKG_DIR ...] [--mix | --post] [--code FILE] [--out JSON]
Each package is streamed per session over all public windows, as scripts/set_params.py does: once with its own
submission.py and once with the current code (or --code; copied with the package's data files into a temporary
folder). Predicted labels must be byte-identical; with --mix the pre-rebalancing member mix too (np.array_equal).
--post (7 Nov) does one predict pass per code and also records the member mix and the per-window output of every
stage present (SessionEM, TSEM, routing; the last one = the final posteriors), all of which must be byte-identical.
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


def recorder(f, out, first=False):
    """f, recording a copy of each output (first: its first row) in out; the output itself is passed on."""
    def g(*args):
        o = f(*args)
        out.append(np.array(o[0] if first else o))
        return o
    return g


def post_pass(m, d, rows):
    """One predict stream recording labels, the member mix (before predict's NaN guard, as --mix) and each stage's
    per-window output (--post). Instance wrappers only: the routed tail's own stage copies stay unwrapped."""
    rec = {"mix": []}
    m.mixed_proba = recorder(m.mixed_proba, rec["mix"], first=True)
    for name in ("em", "tsem", "route"):
        st = getattr(m, name, None)
        if st is not None:
            rec[name] = []
            st.step = recorder(st.step, rec[name])
    labels = stream(m, d, rows, lambda mm, x: mm.predict(x))
    return {"labels": labels, **{k: np.asarray(v) for k, v in rec.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkgs", nargs="+", type=Path)
    ap.add_argument("--mix", action="store_true")
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--code", type=Path, default=SRC)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    d = protocol.load()
    rows = np.ones(len(d["y"]), bool)
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    res = {"new_code_sha256": sha(a.code), "n_windows": int(rows.sum())}
    for pkg in a.pkgs:
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            new = Path(tmp)
            for f in pkg.iterdir():
                if f.is_file() and f.name != "submission.py":
                    shutil.copyfile(f, new / f.name)
            shutil.copyfile(a.code, new / "submission.py")
            r, outs = {"package_code_sha256": sha(pkg / "submission.py")}, {}
            if a.post:
                old, cur = (post_pass(load_solver(p, f"{tag}_post"), d, rows) for p, tag in ((pkg, "old"), (new, "new")))
                assert sorted(old) == sorted(cur) and len(old["mix"]) == rows.sum()
                r["stages"] = [k for k in ("em", "tsem", "route") if k in old]
                outs = {k: (old[k], cur[k]) for k in old}
                if r["stages"]:
                    outs["final_posteriors"] = outs[r["stages"][-1]]
            else:
                fns = {"labels": lambda m, x: m.predict(x)}
                if a.mix:
                    fns["mix"] = lambda m, x: m.mixed_proba(x)
                for what, fn in fns.items():
                    outs[what] = (stream(load_solver(pkg, f"old_{what}"), d, rows, fn),
                                  stream(load_solver(new, f"new_{what}"), d, rows, fn))
            for what, (old_out, new_out) in outs.items():
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
