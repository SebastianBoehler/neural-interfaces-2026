"""Lane B5: learning curve of the raw FBCNet SessionBN member over the number of training people N (full0 delivery).

python laneB5/curve.py SEED HELD   one task: for held person HELD and net seed SEED, fit the shipped recipe
(fbcnet_raw_sbn, 40 epochs, k 20, public PROTEUS, no trunk) on N of the other 9 people for N=5..9, S nested random
subsets per N<9 (subset s = first N of one fixed permutation per (HELD, s); N=9 once), score HELD streamed in the pinned
full0 order. Writes WORK/b5_{seed}_{held}.npz: for each config 'N{n}s{s}' the posteriors (query rows, delivery order)."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "2026-competition/tracks/bci_decoding")]
import protocol  # noqa: E402
import run_cv  # noqa: E402
import session_bn  # noqa: E402

NS, S = (5, 6, 7, 8), 5
ORDER_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"
PEOPLE = [f"P{i:02}" for i in range(1, 11)]


def configs(held):
    """(name, training people) in a fixed order; the permutation depends on HELD and s only (nested across N, shared by seeds)."""
    others = [p for p in PEOPLE if p != held]
    out = [("N9s0", others)]
    for s in range(S):
        perm = np.random.default_rng([20261120, PEOPLE.index(held), s]).permutation(others).tolist()
        out += [(f"N{n}s{s}", sorted(perm[:n])) for n in NS]
    return out


def main(seed, held, work):
    import hashlib
    d = protocol.load(())
    perms = ROOT / "runs/shuffle_perms.npz"
    assert hashlib.sha256(perms.read_bytes()).hexdigest() == ORDER_SHA, "order archive changed"
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    assert set(person) == set(PEOPLE) and (d["split"] != "external").all()
    query = np.flatnonzero(person == held)
    with np.load(perms) as p:
        rows = p["full|0|all_loso"]
    rows = rows[np.isin(rows, query)]
    np.testing.assert_array_equal(np.sort(rows), query)
    out, meta = dict(index=query, delivery=rows), []
    quick = bool(os.environ.get("B5_QUICK"))  # smoke: 1 epoch, first 3 configs
    for name, people in configs(held)[:3 if quick else None]:
        assert held not in people and len(people) == int(name[1:].split("s")[0])
        train = np.flatnonzero(np.isin(person, people))
        t0 = time.time()
        run_cv.SEED[0] = seed
        net = session_bn.make("fbcnet_raw_sbn", d, seed)
        assert net.init is None and net.epochs == 40, (net.init, net.epochs)  # shipped recipe, no trunk
        net.epochs = 1 if quick else net.epochs
        net.fit(train)
        net.set_k(20.0)
        net.net.eval()
        post = np.zeros((len(query), 3))
        for st in dict.fromkeys(d["stream"][rows]):
            r = rows[d["stream"][rows] == st]
            pr = net.predict_proba_stream(r)
            assert np.isfinite(pr).all() and (pr >= 0).all()
            post[np.searchsorted(query, r)] = pr
        out[name] = post
        meta.append(dict(cfg=name, people=people, train_rows=len(train), sec=round(time.time() - t0)))
        print("PASS", name, held, seed, round(time.time() - t0), flush=True)
    stem = work / f"b5{'q' if quick else ''}_{seed}_{held}"
    np.savez(f"{stem}.npz", **out)
    json.dump(dict(seed=seed, held=held, cfgs=meta, job=os.environ.get("SLURM_JOB_ID")), open(f"{stem}.json", "w"))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], Path(os.environ["B5_WORK"]))
