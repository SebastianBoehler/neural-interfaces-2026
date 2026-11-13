"""Assemble a package directory: the bundled submission.py plus the data files of an existing package.

Usage: python lean/tools/build.py DATA_DIR OUT_DIR

DATA_DIR is a package directory (weights.npz and the TorchScript files it names); its files are copied unchanged.
The bundle must be current (lean/tools/bundle.py --check). The build fails if a file named by a member with
non-zero weight is missing or if the package does not load. OUT_DIR.sha256 lists the digests of the package files.
"""
import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bundle import OUT as BUNDLE, bundle  # noqa: E402
from record import load_model  # noqa: E402


def main():
    data, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    assert BUNDLE.read_text() == bundle(), "submissions/lean_v5/submission.py is stale: run lean/tools/bundle.py"
    with np.load(data / "weights.npz") as z:
        w = z["member_weights"]
        named = {str(z[k]) for i in range(int(z["n_nets"])) if w[1 + i] != 0
                 for k in (f"net{i}_jit_file", f"net{i}_sjit_file") if k in z.files}
    files = sorted(p for p in data.iterdir() if p.suffix in (".npz", ".pt"))
    missing = named - {p.name for p in files}
    assert not missing, f"{data} lacks files named by weights.npz: {sorted(missing)}"
    unused = {p.name for p in files} - named - {"weights.npz"}
    assert not unused, f"{data} holds files no member uses: {sorted(unused)}"
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(BUNDLE, out / "submission.py")
    for p in files:
        shutil.copyfile(p, out / p.name)
    load_model(out, None)  # the package must load (weights checked against the code, every net file opened)
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(out.iterdir())]
    Path(f"{out}.sha256").write_text("\n".join(lines) + "\n")
    same = all(hashlib.sha256((data / p.name).read_bytes()).hexdigest() == hashlib.sha256((out / p.name).read_bytes()).hexdigest()
               for p in files)
    assert same, "a data file changed while copying"
    print(f"built {out}: submission.py + {len(files)} data files, {sum(p.stat().st_size for p in out.iterdir()) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
