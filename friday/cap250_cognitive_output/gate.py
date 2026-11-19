"""Evaluate a fixed class-group selector on independent complete terminal banks."""
import argparse
from contextlib import chdir
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path("/mnt/lustre/work/gehler/gfh527")
NAMES = ("stock", "hard", "joint", "protected_hard", "protected_joint", "restricted_oracle")
CONTRASTS = ((4, 0), (4, 3), (4, 2), (3, 0), (2, 0), (5, 0))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signed(path, digest):
    assert sha(path) == digest and not path.stat().st_mode & 0o222
    return json.loads(path.read_text())


def write(path, payload):
    with path.open("x") as handle:
        json.dump(payload, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
    path.chmod(0o444)
    return sha(path)


def gated(bank):
    stock, hard, joint = bank
    treated = [np.where((stock != 0) & (alt != 0), alt, stock) for alt in (hard, joint)]
    for result in treated:
        np.testing.assert_array_equal(result == 0, stock == 0)
        assert np.any(result != stock)
    return np.stack([*bank, *treated])


def oracle(bank, gold):
    return np.concatenate([bank, np.where(bank[0] == gold, bank[0], bank[4])[None]])


def measures(point, boot):
    values = dict(zip(NAMES, point.tolist(), strict=True))
    contrasts = {}
    for i, j in CONTRASTS:
        pp = float(point[i] - point[j])
        lo, hi = np.percentile(boot[:, i] - boot[:, j], (10, 90)).tolist()
        contrasts[f"{NAMES[i]}_minus_{NAMES[j]}"] = dict(
            pp=pp, paired80=[lo, hi], verdict="supported" if lo > 0 else "refuted" if hi <= 0 else "undecided")
    return dict(absolute_pct=values, contrasts=contrasts)


def changes(bank, gold):
    a, b = bank[0], bank[4]
    result = {}
    for c in range(3):
        scope = gold == c
        result[str(c)] = dict(rescues=int(np.count_nonzero(scope & (a != gold) & (b == gold))),
                              regressions=int(np.count_nonzero(scope & (a == gold) & (b != gold))))
    assert result["0"] == dict(rescues=0, regressions=0)
    return result


def held(out):
    from score32 import WORK, authenticate, METRIC, METRIC_HASH, FEAT, HASHES, pin, start_module

    lock = "aaf7e95734d83cae19fb05d0305107138884a1600d7f2de3cf8ffb3b6e13ce6e"
    prior = signed(WORK/"before_labels.json", lock)
    smoke = json.loads((WORK/"outer_smoke.json").read_text())
    bank = {}
    for order, bound in prior["orders"].items():
        raw, receipt, proof = authenticate(order, smoke)
        assert bound["proof"] == proof and bound["bank"] == receipt["output"]
        assert bound["predictions"] == hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
        bank[order] = gated(raw[:, [0, 1, 3]].T)
    seal = write(out/"before_labels.json", dict(parent=lock, held_labels_opened=False,
                 outputs={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in bank.items()}))
    pin(METRIC, METRIC_HASH)
    pin(FEAT, HASHES[FEAT])
    metric = start_module("c27_official", METRIC, METRIC.read_text())
    with np.load(FEAT, allow_pickle=False) as z:
        gold, subject, visit, interface = (z[k] for k in ("y", "subject", "session", "interface"))
    groups = np.stack((subject, visit, interface), axis=1)
    rng = np.random.default_rng(460)
    draw = rng.integers(10, size=(4000, 10))
    reports = {}
    for order, decisions in bank.items():
        matrix = oracle(decisions, gold)
        cells = [np.stack([metric.group_scores(metric.balanced_accuracy,gold[m],q[m],groups[m])
                           for q in matrix]) for who in sorted(set(subject)) if (m := subject==who).any()]
        mass = np.array([c.shape[1] for c in cells])
        assert len(mass) == 10 and mass.sum() == 28
        sums = 100*np.stack([c.sum(axis=1) for c in cells])
        reports[order] = dict(**measures(sums.sum(0)/28, sums[draw].sum(1)/mass[draw].sum(1)[:,None]),
                              changes=changes(decisions,gold))
    return dict(scope="FULL-SYSTEM HELD-NEW DEV", orders=reports, before_labels=seal)


def external(out):
    import score_f32x as parent
    release = BASE/"ni26_friday_f32x_external_cpu_v2_20261119"
    lock = "a68c914b4ebfd56fc8ca5b222e12a1b1be473773e40790f7df467884ed80e7d5"
    prior = signed(release/"before_labels.json", lock)

    def authenticate_only(path, payload):
        assert path == Path("before_labels.json")
        assert json.dumps(payload, sort_keys=True) == json.dumps(prior, sort_keys=True)
        return lock

    parent.seal = authenticate_only
    with chdir(release):
        original, subject, visit, interface, streams, people, verified, _ = parent.qualify()
    assert verified == lock
    band = gated(original)
    seal = write(out/"before_labels.json", dict(parent=lock, held_labels_opened=False,
                 outputs=hashlib.sha256(band.tobytes()).hexdigest()))
    with np.load(parent.DATA, allow_pickle=False) as z:
        gold = z["y"]
    matrix = oracle(band,gold)
    meta = dict(stream=streams, interface=interface)
    cells = np.concatenate([parent.official_cells(a,gold,subject,visit,meta,people)[0]
                            for a in np.split(matrix,2)])
    rng = np.random.default_rng(468)
    i = rng.integers(9,size=(10000,9,1,1))
    v = rng.integers(2,size=(10000,9,2,1))
    o = rng.integers(2,size=(10000,9,2,2))
    result = measures(cells.mean((1,2,3)), cells[:,i,v,o].mean((2,3,4)).T)
    return dict(scope="FULL-SYSTEM EXTERNAL DEV", **result, changes=changes(band,gold),
                order_primary_pp=(cells[4]-cells[0]).mean((0,1)).tolist(), before_labels=seal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=("held", "external"))
    args = parser.parse_args()
    result = {"held":held, "external":external}[args.scope](Path.cwd())
    print("PASS F-C27", args.scope, write(Path("result.json"),result), result, flush=True)
