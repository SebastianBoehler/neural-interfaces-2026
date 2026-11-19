"""Measure hard-only transfer across six same-actor native external deliveries."""
import argparse
from contextlib import chdir
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path("/mnt/lustre/work/gehler/gfh527")
OLD = "ni26_friday_f32x_external_cpu_v2_20261119"
NEW = "ni26_friday_c25_external_v1_20261119"
BALANCE = "ni26_friday_f35x_external_v1_20261119"
SCOPES = {
    "old": (OLD, "a68c914b4ebfd56fc8ca5b222e12a1b1be473773e40790f7df467884ed80e7d5",
            (14411,15411), (0.05952380952380961,0.13888888888888903), "hard"),
    "new": (NEW, "b30b58762ab5edf7694d8238de72f1f023aa137e5ac66fdbae0adbad087623f3",
            (16411,17411), (0.15873015873015908,0.4563492063492063), "hard"),
    "balance": (BALANCE, "30c722ab666899d9efc02ca254be03f339b6e544c2952ca46ec5f691f136972f",
                (26411,27411), (0.28439153439153486,0.22486772486772544), "hard_h100"),
}
ACTORS = "0292022abd39067cefebd0e3a0bd8f1834a04bfc67d7d41081cfe6abaae8ebd2"
BALANCE_ACTORS = "090a5cd4ea2ab7f136021ae51a36a74e832b1302b07f56fb0d3ee317edb0f20e"
DATA = BASE/"neural-interfaces-2026/data/scherer2015/windows.npz"
DATA_SHA = "a6188799753276451235678d08e92fd79162a95ec8099dbbb85e2ed9e2701e91"


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def seal(path, payload):
    with path.open("x") as handle:
        json.dump(payload,handle,sort_keys=True,allow_nan=False)
        handle.write("\n")
    path.chmod(0o444)
    return sha(path)


def dispatch(scope, path):
    import score_f32x as original

    folder, digest, orders, _, hard = SCOPES[scope]
    release = BASE/folder
    signed = release/"before_labels.json"
    actor_path = release/"actors.json"
    old_actor_path = BASE/OLD/"actors.json"
    assert sha(signed) == digest and sha(old_actor_path) == ACTORS
    actor_lock = sha(actor_path)
    assert actor_lock == (BALANCE_ACTORS if scope == "balance" else ACTORS)
    reference = json.loads(old_actor_path.read_text())
    current = json.loads(actor_path.read_text())
    receipt_key = "h100_receipts" if scope == "balance" else "export_receipts"
    assert current["device"] == reference["device"] == "cpu"
    assert current["actors"][hard] == reference["actors"]["hard"]
    assert current[receipt_key] == reference["export_receipts"]
    assert tuple(original.SEEDS) == orders
    columns = [original.ARMS.index(arm) for arm in ("stock",hard)]

    def reauthenticate(name, data):
        assert name == Path("before_labels.json")
        assert json.dumps(data,sort_keys=True) == json.dumps(json.loads(signed.read_text()),sort_keys=True)
        return digest

    original.seal = reauthenticate
    with chdir(release):
        bank, _, _, _, _, _, proof, _ = original.qualify()
    assert proof == digest and bank[columns].shape == (2,2,2130)
    with path.open("xb") as handle:
        np.savez_compressed(handle, choices=bank[columns], orders=orders, parent=digest, actors=actor_lock)
    path.chmod(0o444)
    print("PASS verified parent", scope, sha(path), flush=True)


def score(out):
    from protocol import score as official

    assert sha(DATA) == DATA_SHA
    locked, choices = {}, []
    for scope, (_, digest, orders, _, _) in SCOPES.items():
        path = out/f"{scope}.npz"
        assert not path.stat().st_mode&0o222
        with np.load(path,allow_pickle=False) as z:
            assert str(z["parent"]) == digest
            assert str(z["actors"]) == (BALANCE_ACTORS if scope == "balance" else ACTORS)
            assert z["orders"].tolist() == list(orders) and z["choices"].shape == (2,2,2130)
            choices.append(z["choices"])
        locked[scope] = sha(path)
    bank = np.concatenate(choices,axis=1)
    order_ids = [seed for _,_,orders,_,_ in SCOPES.values() for seed in orders]
    n_orders = len(order_ids)
    before = seal(out/"before_labels.json",dict(held_labels_opened=False, shards=locked, orders=order_ids,
                  arrays=hashlib.sha256(bank.tobytes()).hexdigest(), actors=[ACTORS,BALANCE_ACTORS]))
    with np.load(DATA,allow_pickle=False) as z:
        y, who, visit, interface = [z[k] for k in ("y","subject","session","interface")]
    people = sorted(set(who))
    assert len(people) == 9
    streams = np.char.add(np.char.add(who,"|"),visit)
    cells = np.empty((2,9,2,n_orders,3),dtype=float)
    for i, person in enumerate(people):
        for v, day in enumerate(("1","2")):
            for c in range(3):
                rows = (who==person)&(visit==day)&(y==c)
                assert rows.any()
                cells[:,i,v,:,c] = 100*np.mean(bank[:,:,rows]==c,axis=2)
    for arm in range(2):
        for order in range(n_orders):
            q = official(y,bank[arm,order],dict(stream=streams,interface=interface),np.ones(len(y),dtype=bool))
            assert q["n_cells"]==18
            np.testing.assert_allclose(np.sort(cells[arm,:,:,order].mean(-1).ravel()),
                                       np.sort(100*np.asarray(q["cell_scores"])),rtol=0,atol=1e-10)
    delta = cells[1]-cells[0]
    orders = delta.mean((0,1,3))
    expected = [pp for _,_,_,reference,_ in SCOPES.values() for pp in reference]
    np.testing.assert_allclose(orders,expected,rtol=0,atol=1e-10)
    rng = np.random.default_rng(468)
    i = rng.integers(9,size=(15000,9,1,1))
    v = rng.integers(2,size=(15000,9,2,1))
    o = rng.integers(n_orders,size=(15000,9,2,n_orders))
    boot_class = delta[i,v,o].mean((1,2,3))
    people_pp = delta.mean((1,2,3))
    loo = np.array([np.delete(people_pp,i).mean() for i in range(9)])
    payload = dict(before_labels=before, pooled_pp=float(delta.mean()),
                   paired80=np.percentile(boot_class.mean(1),[10,90]).tolist(),
                   orders=dict(zip(order_ids,orders.tolist(),strict=True)),
                   people_pp=dict(zip(people,people_pp.tolist(),strict=True)),
                   class_pp=delta.mean((0,1,2)).tolist(),
                   class_paired80=np.percentile(boot_class,[10,90],axis=0).T.tolist(),
                   leave_one_person_pp=[float(loo.min()),float(loo.max())],
                   scope="retrospective full-system external DEV, same nine people and six deliveries")
    print("PASS pooled hard-only", seal(out/"result.json",payload),payload,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=(*SCOPES,"score"))
    mode = p.parse_args().mode
    if mode == "score":
        score(Path.cwd())
    else:
        dispatch(mode, Path.cwd()/f"{mode}.npz")
