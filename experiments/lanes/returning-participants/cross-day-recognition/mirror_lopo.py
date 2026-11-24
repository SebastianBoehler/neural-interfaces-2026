"""Lane Q test 2 (handedness diagnostic): LOPO member fit on PROTEUS (unchanged recipe), posteriors on the held person
for the ORIGINAL input and for the left-right MIRRORED input (homologous channel swap, midline fixed), full0..full3 orders.
python laneQ/mirror_lopo.py CFG SEED [FOLD ...]   CFG: fbcnet_raw_sbn (laneH arm A recipe) or fblight_ea_rot05_sbn.
Mirror applies to the whole session stream (incl. the EA/SessionBN statistics), nothing else changes. Labels are not read here."""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "experiments"), str(ROOT / "2026-competition/tracks/bci_decoding")]
import protocol  # noqa: E402
import run_cv  # noqa: E402
import session_bn  # noqa: E402

ORDERS = [f"full{i}" for i in range(4)]
ORDER_SHA = "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f"


def mirror_perm(names):
    names = [str(n) for n in names]
    def mate(n):
        d = n[-1]
        if not d.isdigit() or n in ("Pz", "Cz", "Fz", "CPz", "FCz"):
            return n
        m = n[:-1] + str(int(d) + 1 if int(d) % 2 else int(d) - 1)
        assert m in names, (n, m)
        return m
    p = np.array([names.index(mate(n)) for n in names])
    assert (p[p] == np.arange(len(p))).all() and (p != np.arange(len(p))).sum() == 36
    return p


def run(cfg, seed, held, work):
    d = protocol.load(())
    perms = ROOT / "runs/shuffle_perms.npz"
    assert hashlib.sha256(perms.read_bytes()).hexdigest() == ORDER_SHA
    person = np.array([s.rsplit("/", 1)[-1] for s in d["subject"]])
    query = np.flatnonzero(person == held)
    train = np.flatnonzero(person != held)
    run_cv.SEED[0] = seed
    net = session_bn.make(cfg, d, seed)
    if os.environ.get('LANEQ_QUICK'):
        net.epochs = 1  # smoke only
    t0 = time.time()
    net.fit(train)
    net.set_k(20.0)
    net.net.eval()
    perm = mirror_perm(d["ch_names"])
    X0 = d["X"]
    out = dict(index=query, perm=perm)
    with np.load(perms) as p:
        for view in ("orig", "mirror"):
            d["X"] = X0 if view == "orig" else X0[:, perm, :]
            for o in ORDERS:
                rows = p[f"full|{o[-1]}|all_loso"]
                rows = rows[np.isin(rows, query)]
                post = np.zeros((len(query), 3))
                for st in dict.fromkeys(d["stream"][rows]):
                    r = rows[d["stream"][rows] == st]
                    post[np.searchsorted(query, r)] = net.predict_proba_stream(r)
                assert np.isfinite(post).all()
                out[f"{view}|{o}"], out[f"{view}|{o}|delivery"] = post, rows
    d["X"] = X0
    stem = work / f"q_mirror_{cfg}_{seed}_{held}"
    np.savez(f"{stem}.npz", **out)
    json.dump(dict(cfg=cfg, seed=seed, held=held, sec=round(time.time() - t0), job=os.environ.get("SLURM_JOB_ID"),
                   gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"), open(f"{stem}.json", "w"))
    print("PASS", stem.name, round(time.time() - t0), flush=True)


if __name__ == "__main__":
    cfg, seed = sys.argv[1], int(sys.argv[2])
    work = Path(os.environ["LANEQ_WORK"])
    for held in sys.argv[3:] or [f"P{i:02}" for i in range(1, 11)]:
        run(cfg, seed, held, work)
