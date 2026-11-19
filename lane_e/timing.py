"""Lane E runtime: warmed, synchronised, paired timing of exact package ZIPs on forced-unrecognised Scherer fixtures.

python lane_e/timing.py OUT.json DEVICE ZIP[:TAG] ...   (first ZIP = reference, normally cap250)
Fixtures F120 / F190 / F378: the first n Scherer windows (visits concatenated, recording order) streamed as ONE session
(external people never recognise; refits from window 20, cap 250 binds on F378). Every arm is loaded once; one untimed
warm-up run each; then 2 rounds per fixture with the arm order alternating (forward, reverse), fresh copy per run.
Reports seconds per run, refits, ft_s (the guard's counter), over, capped, errors, and label agreement with the reference.
"""
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H  # noqa: E402

out, dev, *specs = sys.argv[1:]
arms = [(s.split(":")[1] if ":" in s else Path(s).stem, s.split(":")[0]) for s in specs]
X, subject, session, ch = H.scherer(H.ROOT / "data/scherer2015/windows.npz") if not Path("/external.npz").exists() else H.scherer("/external.npz")
FIX = {"F120": 120, "F190": 190, "F378": 378}
res = {"device": dev, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "runs": []}
with ExitStack() as stack:
    models = {tag: H.load(zp, ch, dev, stack.enter_context(TemporaryDirectory())) for tag, zp in arms}
    for tag in models:  # untimed warm-up (CUDA kernels, TorchScript specialisation, first refit)
        H.run(models[tag], X, H.stream_rows(subject, session, 60), dev)
    ref_labels = {}
    for fx, n in FIX.items():
        rows = H.stream_rows(subject, session, n)
        for rnd in range(2):
            order = list(models) if rnd == 0 else list(models)[::-1]
            for tag in order:
                labels, outs, sec, m = H.run(models[tag], X, rows, dev)
                ref = ref_labels.setdefault(fx, labels) if tag == arms[0][0] and rnd == 0 else ref_labels.get(fx)
                rec = dict(fixture=fx, round=rnd, arm=tag, sec=round(sec, 2), refits=m.ft.refits, steps=m.ft.steps, ft_s=round(m.ft.ft_sec, 2),
                           over=m.ft.over, capped=m.ft.capped, err=int(m.log6["err"]), ft_nets=len(m.ft.member),
                           agree_ref=None if ref is None else round(float((labels == ref).mean()), 4))
                res["runs"].append(rec)
                print(json.dumps(rec), flush=True)
                del m
Path(out).write_text(json.dumps(res, indent=1))
