"""Same returning fold, with both native recognizers disabled before the first window."""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(os.environ["CAP250_RELEASE"]) / "cleanslate/harness"))
import data as harness
from stream import load_solver


def main(args):
    d = harness.load(args.data)
    _, streaming = harness.official()
    for who in (["P04"] if args.smoke else ["P04", "P10"]):
        model = load_solver(args.stock / f"ret_{who}/pkg", d["ch_names"], "cuda:0")
        base = np.load(args.stock / f"pred_ret_{who}.npz")
        output = {}
        for seed in ([71000] if args.smoke else range(71000, 71004)):
            for key, rows in harness.plans(d, seed, [who + "|2", who + "|3"]).items():
                tag = f"{seed}_{key[-1]}"
                np.testing.assert_array_equal(rows, base[f"rows_{tag}"])
                m = streaming.fresh_model(model)
                m.route.contrast.t_start = m.route.identity.t0 = len(rows) + 1
                pred = []
                for row in rows:
                    lab = m.predict(torch.from_numpy(d["X"][row:row + 1]).to("cuda:0"))
                    assert lab.shape == (1,) and lab.dtype == torch.int64
                    pred.append(int(lab.item()))
                assert m.route.log["routed"] == 0 and m.log6["rec"] == -1 and m.log6["err"] == 0
                np.testing.assert_array_equal(pred[:10], base[f"labels_{tag}"][:10])
                if args.smoke:
                    m = streaming.fresh_model(model)
                    same = [int(m.predict(torch.from_numpy(d["X"][r:r+1]).to("cuda:0")).item()) for r in rows]
                    np.testing.assert_array_equal(same, base[f"labels_{tag}"])
                    assert m.log6["err"] == 0 and m.log6["rec"] >= 0
                    print("SMOKE OK: stock label parity and route disabled", who, tag, flush=True)
                    return
                output[f"rows_{tag}"] = rows
                output[f"labels_{tag}"] = np.array(pred, np.int8)
                print("population", who, tag, flush=True)
        np.savez(args.out / f"pred_pop_{who}.npz", **output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("stock", "data", "out"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args())
