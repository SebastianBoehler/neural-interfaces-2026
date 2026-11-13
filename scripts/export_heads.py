"""Per-corpus head log-probabilities of pretrained trunks on the PROTEUS windows: the teacher's inputs
(rebuild of the 11 Nov export, whose script was untracked; ledger s148). Each checkpoint of experiments/pretrain.py
is rebuilt from its saved meta (architecture, corpus blocks), run on the 2,655 PROTEUS windows aligned per session as
pretraining aligned its corpora (ALIGN "ea": session mean-covariance whitening), and every corpus block is turned into
log-probabilities over its own classes. Output: one (2655, n_outputs) array per checkpoint stem plus "names"
(corpus:class), the format of runs/tsch/heads.npz read by studies/13nov/teach2.py.
Usage: python -I scripts/export_heads.py OUT.npz CKPT [CKPT ...]"""
import json
import sys
from pathlib import Path

import braindecode.models as bd
import inspect
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from deep import ARCHS, device  # noqa: E402
from pretrain import align_sessions  # noqa: E402


def main():
    out, ckpts = Path(sys.argv[1]), [Path(c) for c in sys.argv[2:]]
    assert ckpts, "no checkpoints given"
    with np.load(ROOT / "data/proteus_windows.npz") as z:
        X0, s = z["X"].astype(np.float32), np.char.add(np.char.add(z["subject"].astype(str), "|"), z["session"].astype(str))
    res, names, dev = {}, None, device()
    for c in ckpts:
        ck = torch.load(c, map_location="cpu"); meta = json.loads(ck["meta"]); corp = meta["corpora"]
        for v in corp.values():  # checkpoints before the unified block (R17) store no "n"
            v.setdefault("n", len(v["classes"]))
        off = max(v["offset"] + v["n"] for v in corp.values()); cls = getattr(bd, ARCHS[meta["arch"]])
        if ck["state"]["final_layer.bias"].shape[0] == 3 and off != 3:  # --unified (R17): only the shared block's 3-class head is saved
            assert meta.get("unified"), f"{c}: a 3-output head without --unified"
            corp = {"unified": {"offset": 0, "n": 3, "classes": ["MI", "SUB", "WORD"]}}; off = 3
        kw = dict(n_chans=41, n_outputs=off, n_times=480)
        if "sfreq" in inspect.signature(cls).parameters:
            kw["sfreq"] = 120.0
        net = cls(**kw); net.load_state_dict(ck["state"]); net.to(dev).eval()
        X = align_sessions(X0.copy(), s, meta["align"])
        with torch.no_grad():
            Z = torch.cat([net(torch.from_numpy(X[i:i + 256]).to(dev)).cpu() for i in range(0, len(X), 256)]).double()
        L = np.zeros(Z.shape); nm = [None] * off
        for k, v in corp.items():
            o, n = v["offset"], v["n"]; L[:, o:o + n] = torch.log_softmax(Z[:, o:o + n], 1).numpy()
            for j in range(n):
                nm[o + j] = f"{k}:{v['classes'][j]}"
        assert np.isfinite(L).all() and L.max() <= 1e-9, c
        assert names is None or nm == names[:len(nm)] or names == nm[:len(names)], f"{c}: corpus blocks differ from the other checkpoints"
        names = nm if names is None or len(nm) > len(names) else names; res[c.stem] = L
        print(f"{c.stem}: {off} outputs, mean max log-prob {L.max(1).mean():.3f}", flush=True)
    np.savez(out, names=np.array(names), **res); print("written", out, len(res), "checkpoints")


if __name__ == "__main__":
    main()
