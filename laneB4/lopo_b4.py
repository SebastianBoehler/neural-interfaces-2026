"""Lane B4: one SessionBN family, one seed, LOPO on PROTEUS (laneH/lopo.py bank format, any shipped family).

python laneB4/lopo_b4.py FAMILY SEED [FOLD ...]   FAMILY raw | eafbc | fblight | sinc (shipped recipes, public PROTEUS only);
FOLD P01..P10 (person never trained on) or "all" (deployment fit, no query). Default: P01..P10 and all.
Writes WORK/b4_FAMILY_SEED_FOLD.{npz,json}: index, train, full0..7 (+_delivery), sec per fit and per predict."""
import hashlib
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

# family: (session_bn.make cfg, k at test, needs an external trunk); shipped: raw k20, EA FBC k2/k5 (k5 here), FBLight k5, Sinc m7r k20
FAM = {"raw": ("fbcnet_raw_sbn", 20.0, False), "eafbc": ("fb_sbn_rot05", 5.0, False),
       "fblight": ("fblight_ea_rot05_sbn", 5.0, False), "sinc": ("sincshallow_ea_ptm7r_rot05_sbn", 20.0, True)}
ORDERS = [f"full{i}" for i in range(8)]
ORDER_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"


def run(fam, seed, held, work):
    cfg, k_test, trunk = FAM[fam]
    d = protocol.load(())
    perms = ROOT / "runs/shuffle_perms.npz"
    assert hashlib.sha256(perms.read_bytes()).hexdigest() == ORDER_SHA, "order archive changed"
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    assert set(person) == {f"P{i:02}" for i in range(1, 11)} and (d["split"] != "external").all()
    query = np.flatnonzero(person == held) if held != "all" else np.array([], int)
    train = np.flatnonzero(person != held)
    run_cv.SEED[0] = seed
    net = session_bn.make(cfg, d, seed)
    assert isinstance(net, session_bn.SessionNet), type(net)
    assert trunk == (net.init is not None), "trunk wiring differs from the family"
    epochs_shipped = net.epochs
    if os.environ.get("LANEH_QUICK"):
        net.epochs = 1
    t0 = time.time()
    net.fit(train)
    fit_sec = time.time() - t0
    if trunk:
        assert Path(net.init.format(seed=seed)).exists(), "trunk missing"
    net.set_k(k_test)
    net.net.eval()
    out = dict(index=query, train=train)
    t1 = time.time()
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
                np.testing.assert_allclose(pr[:3], net.predict_proba_stream(r[:3]), atol=5e-4 if fam == "fblight" else 1e-4, rtol=0)  # causal prefix (FBLight float noise 1.2e-4 batch vs one-window; a leak would be >1e-2)
                post[np.searchsorted(query, r)] = pr
            out[o], out[o + "_delivery"] = post, rows
    stem = work / f"b4_{fam}_{seed}_{held}"
    np.savez(f"{stem}.npz", **out)
    json.dump(dict(fam=fam, seed=seed, held=held, cfg=cfg, init=net.init, k_test=k_test, epochs=net.epochs,
                   epochs_shipped=epochs_shipped, train_rows=len(train), query_rows=len(query), fit_sec=round(fit_sec, 1),
                   predict_sec=round(time.time() - t1, 1), gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                   job=os.environ.get("SLURM_JOB_ID")), open(f"{stem}.json", "w"))
    print("PASS", stem.name, round(fit_sec), flush=True)


if __name__ == "__main__":
    fam, seed = sys.argv[1], int(sys.argv[2])
    work = Path(os.environ["LANEH_WORK"])
    for held in sys.argv[3:] or [f"P{i:02}" for i in range(1, 11)] + ["all"]:
        run(fam, seed, held, work)
