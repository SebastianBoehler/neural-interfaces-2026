"""Lane H stage B: production raw FBCNet SessionBN member, LOPO on PROTEUS, with and without a pooled-corpus trunk.

python laneH/lopo.py ARM SEED [FOLD ...]   ARM A = fbcnet_raw_sbn (shipped recipe, public PROTEUS only, no trunk)
  B = fbcnet_raw_pthp7_sbn, C = fbcnet_raw_pthp9_sbn (same net/optimiser/epochs, trunk from pretrain_h.py, same seed)
FOLD P01..P10 (all_loso: the person's rows are never trained on) or "all" (deployment, no query).
Bank (laneC format): index, full0..full7 (posteriors in index order, computed with the session streamed in delivery
order), full*_delivery, train. Checks: every external-trunk arm really loaded the trunk; batch == one-window prefix."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "2026-competition/tracks/bci_decoding")]
import protocol  # noqa: E402
import run_cv  # noqa: E402
import session_bn  # noqa: E402

CFG = {"A": "fbcnet_raw_sbn", "B": "fbcnet_raw_pthp7_sbn", "C": "fbcnet_raw_pthp9_sbn", "S": "fbcnet_raw_pths9_sbn"}  # S: smoke trunk (1 epoch)
ORDERS = [f"full{i}" for i in range(8)]
ORDER_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
K_TEST = 20.0  # shipped sbn_fbcnet_raw_k20


def run(arm, seed, held, work):
    import hashlib
    d = protocol.load(())
    perms = ROOT / "runs/shuffle_perms.npz"
    assert hashlib.sha256(perms.read_bytes()).hexdigest() == ORDER_SHA, "order archive changed"
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    assert set(person) == {f"P{i:02}" for i in range(1, 11)} and (d["split"] != "external").all()
    query = np.flatnonzero(person == held) if held != "all" else np.array([], int)
    train = np.flatnonzero(person != held)
    run_cv.SEED[0] = seed
    net = session_bn.make(CFG[arm], d, seed)
    assert (arm == "A") == (net.init is None), "trunk wiring differs from the arm"
    if arm == "S" or os.environ.get("LANEH_QUICK"):
        net.epochs = 1
    t0 = time.time()
    net.fit(train)
    if arm != "A":  # the trunk must actually have been loaded (deep.py filters silently by shape)
        sd = torch.load(net.init.format(seed=seed), map_location="cpu")["state"]
        own = net.net.state_dict()
        moved = [k for k in sd if k in own and own[k].shape == sd[k].shape and own[k].dtype.is_floating_point]
        assert len(moved) >= len(own) - 4, "trunk keys missing"
    net.set_k(K_TEST)
    net.net.eval()
    out = dict(index=query, train=train)
    with np.load(perms) as p:
        for o in ORDERS if len(query) else []:
            rows = p[f"full|{o[-1]}|all_loso"]
            rows = rows[np.isin(rows, query)]
            np.testing.assert_array_equal(np.sort(rows), query)
            post = np.zeros((len(query), 3))
            for st in dict.fromkeys(d["stream"][rows]):  # session by session, delivery order
                r = rows[d["stream"][rows] == st]
                pr = net.predict_proba_stream(r)
                assert np.isfinite(pr).all() and (pr >= 0).all()
                np.testing.assert_allclose(pr[:3], net.predict_proba_stream(r[:3]), atol=1e-4, rtol=0)  # causal prefix
                post[np.searchsorted(query, r)] = pr
            out[o], out[o + "_delivery"] = post, rows
    stem = work / f"h_{arm}_{seed}_{held}"
    np.savez(f"{stem}.npz", **out)
    json.dump(dict(arm=arm, seed=seed, held=held, cfg=CFG[arm], train_rows=len(train), query_rows=len(query),
                   init=net.init, k_test=K_TEST, epochs=net.epochs, sec=round(time.time() - t0),
                   gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                   job=os.environ.get("SLURM_JOB_ID")), open(f"{stem}.json", "w"))
    print("PASS", stem.name, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    arm, seed = sys.argv[1], int(sys.argv[2])
    work = Path(os.environ["LANEH_WORK"])
    for held in sys.argv[3:] or [f"P{i:02}" for i in range(1, 11)] + ["all"]:
        run(arm, seed, held, work)
