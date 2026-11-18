"""Exercise both pinned Python solvers on a public EEG window and fresh copies."""
import hashlib
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "2026-competition"
sys.path.insert(0, str(OFFICIAL))
sys.path.insert(0, str(ROOT / "native/tools"))
from benchmark_utils.streaming import fresh_model
from runtime_reference import load_solver


def main():
    revision = subprocess.check_output(
        ["git", "-C", str(OFFICIAL), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != "a66bc23d1a0e686b895c093a642b45c74970758b":
        raise ValueError(f"unexpected official source {revision}")
    archives = []
    for line in (ROOT / "friday/inputs.sha256").read_text().splitlines():
        digest, filename = line.split()
        path = ROOT / filename
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"SHA-256 mismatch for {filename}")
        if path.suffix == ".zip":
            archives.append(path)

    if [path.name for path in archives] != ["v51b_uploaded.zip", "v66_cp1_cap250.zip"]:
        raise ValueError("expected both frozen archives in the Friday manifest")
    torch.set_num_threads(1)
    with np.load(ROOT / "data/proteus_windows.npz", allow_pickle=False) as data:
        idx = np.flatnonzero((data["subject"] == "Dreyer2026Proteus/P01") &
                             (data["session"] == "1"))[0]
        x = torch.from_numpy(data["X"][idx:idx + 1].copy())
    if x.shape != (1, 41, 480) or x.dtype != torch.float32:
        raise ValueError("incorrect public EEG tensor")
    for path in archives:
        with TemporaryDirectory() as scratch, ZipFile(path) as archive:
            archive.extractall(scratch)
            model = load_solver(Path(scratch), "cpu")
            labels = [fresh_model(model).predict(x).cpu() for _ in range(2)]
            first, repeated = labels
            if (first.shape != (1,) or first.dtype != torch.int64 or
                    first.item() != 2 or not torch.equal(first, repeated)):
                raise RuntimeError(f"invalid fresh-session predictions {labels}")
            print(f"{path.name}: P01/session 1 first window -> {first.item()} twice")
    print("OK")


if __name__ == "__main__":
    main()
