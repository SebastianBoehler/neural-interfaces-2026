"""Lane O: F-C25 hard actor (F32c pair, fold checkpoints) streamed over the eight full orders full0..full7 for one held person and seed.

usage (pupil.sbatch environment, cwd = release dir): hard_orders.py HELD SEED [smoke]
Same code path as friday/f32c_v2 produce32.py (states = held_joint.paired, frozen_member, pupils.do_order); only the delivery differs
(runs/shuffle_perms.npz 'full|k|all_loso', k = 0..7; seeds 71000+k). Zero-change: full0 must equal the archived hard|full0 fold bitwise."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from anchors import sha
from held import frozen_member
from held_joint import external_metadata, paired
from locked32 import C20_PUPILS, F32, ACTOR
from pupils import do_order
from replay_f13 import ROOT, raw_inputs

ORDERS = [f"full{k}" for k in range(8)]


def main(held, seed, smoke):
    torch.set_num_threads(2)
    assert sha(F32 / "actor.sha256") == ACTOR
    x, subject, people, stream, _ = raw_inputs()
    visit = np.array([v.rsplit("|", 1)[-1] for v in stream])
    with np.load(ROOT / "runs/shuffle_perms.npz", allow_pickle=False) as z:
        perms = {o: z[f"full|{o[-1]}|all_loso"] for o in ORDERS}
    for p in perms.values():
        assert np.array_equal(np.sort(p), np.arange(len(x)))
    rows = np.flatnonzero(people == held)
    checkpoint, fit = paired(held, seed, False, subject, visit, external_metadata())["hard"]
    assert set(fit["source_people"]) == set(subject[people != held]) and fit["source_labels_only"]
    model, layer = frozen_member(checkpoint, fit)
    model.skip = False
    out, info = {}, {}
    for o in ORDERS[:1] if smoke else ORDERS:
        p, delta, clips = do_order(model, layer, x, stream, rows, perms[o], fit["scale"])
        assert clips >= 0 and 0 <= delta <= 2e-5
        out[o], out[o + "_delivery"] = p, perms[o][np.isin(perms[o], rows)]
        info[o] = dict(prefix=delta, active=clips)
    with np.load(C20_PUPILS / "pupils" / f"{held}_s{seed}.npz", allow_pickle=False) as old:
        np.testing.assert_array_equal(old["index"], rows)
        np.testing.assert_array_equal(out["full0"], old["hard|full0"])  # zero-change
    name = f"hard_{held}_s{seed}{'_smoke' if smoke else ''}.npz"
    np.savez(name, index=rows, **out)
    Path(name.replace(".npz", ".json")).write_text(json.dumps(dict(held=held, seed=seed, actor=ACTOR, orders=info, zero_change="full0 == archived")))
    print("PASS laneO hard", held, seed, sha(Path(name)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), len(sys.argv) > 3 and sys.argv[3] == "smoke")
