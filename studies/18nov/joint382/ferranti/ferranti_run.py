"""Run the source-only Ferranti replication after its file and device checks."""
from pathlib import Path
import subprocess
import sys

import torch


def main(mode):
    if "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("Expected the registered H100 worker")
    if mode not in ("smoke", "F", "J", "K"):
        raise ValueError(mode)
    bundle = Path("bundle").resolve()
    arms = ("F", "J", "K") if mode == "smoke" else (mode,)
    for arm in arms:
        cmd = [sys.executable, str(bundle / "fit_bundle.py"), str(bundle), arm, "1382"]
        if mode == "smoke":
            cmd.append("smoke")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main(sys.argv[1])
