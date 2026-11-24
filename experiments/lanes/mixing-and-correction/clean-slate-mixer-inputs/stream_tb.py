"""Revisit (Opus, 20-21 Nov): stream one held person through a Team B R1 package under a named shuffle_perms key, keep the enc posterior.

Inside the worker image (binds as cleanslate/teamB/dev.sbatch plus /perms/shuffle_perms.npz):
python revisit/stream_tb.py --pkg /work/new_P01/pkg --held P01 --keys "full|0|all_loso;within|0|all_loso" --out /out/tb_P01.npz
enc posterior = softmax of the encoder-ensemble log-probability for the newest window (causal EA, prefix re-embedding every K
windows, exactly the package's 'enc' arm); asserted equal in argmax to the package's own enc arm. R1 packages hold no experts
and lp_beta = 0, both asserted, so P[-1] is the enc posterior. Output keys: rows|KEY, post|KEY (n,3) in delivery order."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/cs/harness")
import data as D  # noqa: E402
import stream as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    for k in ("pkg", "held", "keys", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--perms", default="/perms/shuffle_perms.npz")
    a = ap.parse_args()
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    d = D.load()
    x = torch.from_numpy(d["X"])
    query = np.flatnonzero(d["person"] == a.held)
    model = S.load_solver(Path(a.pkg), d["ch_names"], dev)
    assert not getattr(model, "experts", None) and model.cfg.get("lp_beta", 0) <= 0, "enc posterior is not P[-1] for this package"
    _, streaming = D.official()
    out = {}
    for key in a.keys.split(";"):
        with np.load(a.perms) as p:
            rows = p[key]
        rows = rows[np.isin(rows, query)]
        assert np.array_equal(np.sort(rows), query)
        post = np.zeros((len(rows), 3), np.float32)
        for st in dict.fromkeys(d["stream"][rows]):
            idx = np.flatnonzero(d["stream"][rows] == st)
            m = streaming.fresh_model(model)
            assert not m.X and m.S is None, "state crossed the session boundary"
            for i in idx:
                r = int(rows[i])
                m.predict(x[r:r + 1].to(dev))
                post[i] = m.last[0][-1].float().cpu().numpy()
            assert list(post[idx].argmax(1)) == m.arms["enc"], "posterior is not the enc arm"
        out[f"rows|{key}"], out[f"post|{key}"] = rows, post
        print("PASS", a.held, key, round(float((post.argmax(1) == d["y"][rows]).mean()), 4), flush=True)
    np.savez(a.out, **out)


if __name__ == "__main__":
    main()
