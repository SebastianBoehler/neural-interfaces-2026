"""Compare one fresh sbn_* part with the archived cap250 members (weights keys and trace outputs).

python families/_sbn_parity.py --family sbn_fbl --seed 0 --part DIR [--archive runs/sealed/cand_v51b_own] [--data D]
Reports G/scale/align/k/shrink differences and, per trace, the max |p_fresh - p_archive| and the argmax
disagreement over the first sessions of the all-ten source data (streamed window batches, state reset).
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from components import SBN
from source import ASSETS, default_data, load_source, PEOPLE, ROOT

sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "scripts")]
import deep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=tuple(SBN), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--part", type=Path, required=True)
    ap.add_argument("--archive", type=Path, default=ASSETS / "runs/sealed/cand_v51b_own")
    ap.add_argument("--data", type=Path, default=default_data())
    ap.add_argument("--sessions", type=int, default=4)
    args = ap.parse_args()
    new, old = np.load(args.part / "weights.npz"), np.load(args.archive / "weights.npz")
    data = load_source(PEOPLE, (), args.data)
    sessions = list(dict.fromkeys(data["stream"]))[:args.sessions]
    report = {}
    for base, prefix in SBN[args.family]:
        i, name = base + args.seed, f"{prefix}_s{args.seed}.pt"
        r = {key: float(np.abs(new[f"net{i}_{key}"].astype(float) - old[f"net{i}_{key}"].astype(float)).max())
             for key in ("align", "k", "G", "scale", "shrink")}
        r["sjit_file_equal"] = bool(new[f"net{i}_sjit_file"] == old[f"net{i}_sjit_file"])
        G = old[f"net{i}_G"]
        r["G_rel"] = r.pop("G") / float(np.abs(G).max())
        a, b = (torch.jit.load(str(p / name)).eval() for p in (args.part, args.archive))
        diff, flips, n = 0.0, 0, 0
        # the alignment stream is the archive's own G/k: apply it to windows exactly as the package does
        d = data
        net = deep.AlignedNet(d, arch="shallow", align=bool(old[f"net{i}_align"]), k=float(old[f"net{i}_k"]))
        net.G, net.scale = old[f"net{i}_G"], float(old[f"net{i}_scale"])
        with torch.no_grad():
            for s in sessions:
                rows = np.flatnonzero(d["stream"] == s)
                H = torch.from_numpy(net._aligned(rows, causal=True))
                for m in (a, b):
                    for nm, buf in m.named_buffers():
                        if nm.rsplit(".", 1)[-1] in ("s1", "s2", "n"):
                            buf.zero_()
                pa, pb = (torch.softmax(m(H).double(), 1).numpy() for m in (a, b))
                diff = max(diff, float(np.abs(pa - pb).max()))
                flips += int((pa.argmax(1) != pb.argmax(1)).sum())
                n += len(rows)
        r.update(trace_max_abs_prob_diff=diff, argmax_flips=flips, windows=n)
        report[name] = r
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
