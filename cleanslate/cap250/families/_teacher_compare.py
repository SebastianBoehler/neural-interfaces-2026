"""Compare a refit teacher part against the archived cand_v51b_own teacher (same ten-person scope)."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source import ASSETS, load_source, PEOPLE
from _teacher_lib import batched, ea_stream


def main(part, archive=ASSETS / "runs/sealed/cand_v51b_own"):
    part, archive = Path(part), Path(archive)
    data = load_source(PEOPLE)
    with np.load(part / "weights.npz") as new, np.load(archive / "weights.npz") as old:
        p = {k: new[f"net34_{k}"] for k in ("align", "k", "G", "scale", "shrink")}
        po = {k: old[f"net34_{k}"] for k in p}
    res = {k: float(np.abs(np.asarray(p[k], float) - np.asarray(po[k], float)).max()) for k in p}
    H = ea_stream(data["X"], data["stream"], po)  # archive EA stream: isolates the teacher itself
    a = batched(torch.jit.load(str(part / "teacher.pt")), H)
    b = batched(torch.jit.load(str(archive / "teacher.pt")), H)
    res.update(max_abs_logit=float(np.abs(a - b).max()), mean_abs_logit=float(np.abs(a - b).mean()),
               argmax_agree=float(np.mean(a.argmax(1) == b.argmax(1))), n=len(H),
               argmax_disagree=int(np.sum(a.argmax(1) != b.argmax(1))))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
