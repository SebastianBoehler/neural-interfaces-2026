"""Lane I: raw FBCNet SessionBN LOPO fits (lane H arm A recipe), eight pinned orders streamed under each moment rule.

python laneI/shrink_lopo.py SEED [FOLD ...]     FOLD P01..P10 (default all ten). One fit per (seed, held person); the arms only
change the eval-time moment rule. Output WORK/i_{seed}_{held}.npz: index, train, `{arm}|full{o}` posteriors, `full{o}_delivery`."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "2026-competition/tracks/bci_decoding"), str(ROOT / "laneI")]
import protocol  # noqa: E402
import run_cv  # noqa: E402
import session_bn  # noqa: E402
import sbn_shrink  # noqa: E402

ORDERS = [f"full{i}" for i in range(8)]
ORDER_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
TAUS = (5, 10, 20, 40, 80, 160)
ARMS = [f"k{t}" for t in TAUS] + ["s2"]


def stream_all(net, d, perms, query, arm):
    out = {}
    for o in ORDERS:
        rows = perms[f"full|{o[-1]}|all_loso"]
        rows = rows[np.isin(rows, query)]
        np.testing.assert_array_equal(np.sort(rows), query)
        post = np.zeros((len(query), 3))
        for st in dict.fromkeys(d["stream"][rows]):
            r = rows[d["stream"][rows] == st]
            pr = net.predict_proba_stream(r)
            assert np.isfinite(pr).all() and (pr >= 0).all()
            np.testing.assert_allclose(pr[:3], net.predict_proba_stream(r[:3]), atol=1e-4, rtol=0)  # causal prefix
            post[np.searchsorted(query, r)] = pr
        out[f"{arm}|{o}"], out[o + "_delivery"] = post, rows
    return out


def formula_check(mod, x):
    """S2 weight at a = n/(n+20) must reproduce the stock k=20 eval (silent no-op guard for the new arithmetic)."""
    import copy
    mod, x = copy.deepcopy(mod).double().eval(), x.double()  # float64: the identity is algebraic, float32 cancellation is not the point
    mod.mode, mod.k = "stock", 20.0
    ref = mod(x)
    mod.mode = "s2"
    real = mod.weight_s2
    mod.weight_s2 = lambda n, r1, m1: n / (n + 20.0)
    try:
        alt = mod(x)
    finally:
        mod.weight_s2 = real
    err = float((ref - alt).abs().max())
    assert err < 1e-9, err
    return err


def run(seed, held, work):
    d = protocol.load(())
    perms_path = ROOT / "runs/shuffle_perms.npz"
    import hashlib
    assert hashlib.sha256(perms_path.read_bytes()).hexdigest() == ORDER_SHA, "order archive changed"
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    assert set(person) == {f"P{i:02}" for i in range(1, 11)} and (d["split"] != "external").all()
    query = np.flatnonzero(person == held)
    train = np.flatnonzero(person != held)
    run_cv.SEED[0] = seed
    net = session_bn.make("fbcnet_raw_sbn", d, seed)
    assert net.init is None
    t0 = time.time()
    net.fit(train)
    net.net.eval()
    mods = sbn_shrink.install(net)
    assert len(mods) == 1, len(mods)
    info = sbn_shrink.calibrate(net, d, train, mods)
    cap = []
    h = mods[0].register_forward_hook(lambda mod, inp, out: cap.append(inp[0].detach().clone()))
    net.predict_proba_stream(query[:24])
    h.remove()
    info["formula_err"] = formula_check(mods[0], cap[0])
    out = dict(index=query, train=train)
    with np.load(perms_path) as perms:
        for arm in ARMS:
            for m in mods:
                m.mode = "s2" if arm == "s2" else "stock"
                m.k = 20.0 if arm == "s2" else float(arm[1:])
            out.update(stream_all(net, d, perms, query, arm))
    for m in mods:
        m.mode, m.k = "stock", 20.0
    stem = work / f"i_{seed}_{held}"
    np.savez(f"{stem}.npz", **out)
    json.dump(dict(seed=seed, held=held, train_rows=len(train), query_rows=len(query), arms=ARMS, calib=info,
                   T_over_W=[(m.T / m.Wsrc).tolist() for m in mods], sec=round(time.time() - t0),
                   gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                   job=os.environ.get("SLURM_JOB_ID")), open(f"{stem}.json", "w"))
    print("PASS", stem.name, round(time.time() - t0), info, flush=True)


if __name__ == "__main__":
    work = Path(os.environ["LANEI_WORK"])
    work.mkdir(parents=True, exist_ok=True)
    for held in sys.argv[2:] or [f"P{i:02}" for i in range(1, 11)]:
        run(int(sys.argv[1]), held, work)
