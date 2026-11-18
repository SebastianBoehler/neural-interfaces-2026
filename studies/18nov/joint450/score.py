"""Linked-person stress of the corrected §430/431 paired full-system predictions."""
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("lineage", ROOT / "studies/18nov/lineage419/score.py")
lineage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lineage)

EVIDENCE = ROOT / "evidence/next_wave_16nov"
ORDERS = lineage.metric.ORDERS
PEOPLE = lineage.PEOPLE["all_loso"]
GROUPS = {s: [i for i, o in enumerate(ORDERS) if o.startswith(s)] for s in ("within", "full", "runs")}
SEEN = [PEOPLE.index(p) for p in lineage.PEOPLE["seen_session"]]
RETURN = np.array([0.2, 0.8])
NEW = 0.566


def verified_pairs():
    assert (ROOT / "runs/joint450/replay.txt").read_text().splitlines()[-1] == "OK"
    owner = {p: json.loads((EVIDENCE / f"aux{n}_result.json").read_text())
             for p, n in (("all_loso", 430), ("seen_session", 431))}
    acquired = {p: [] for p in owner}
    perms = {}
    shared = {}

    def observe(p, order, rows, perm, labels, cells, people, old, historical):
        if p == "all_loso":
            perms[order] = perm
            path = ROOT / f"runs/aux430/{order}.npz"
            lineage.pin(path, owner[p]["three_sha256"][order])
            new = lineage.absolute(path, "scores")
        else:
            path = ROOT / f"runs/aux431/{order}.npz"
            lineage.pin(path, owner[p]["output_sha256"][order])
            detector = lineage.detector
            new = np.full((detector.C.N, 3), 1 / 3)
            assert np.array_equal(rows, lineage.absolute(path, "rows"))
            new[rows] = lineage.absolute(path, "three")
            hc = detector.J.hyp_cache(order, "wprior", detector.X, detector.LAM,
                                      sessions=detector.SESS, proto=p)
            urn = detector.J.filt_joint(new, hc, "urn")
            lat = detector.latch(order)
            lineage.metric.verify_zero(new, detector.gate(new, urn, order, dict.fromkeys(lat)))
            new = detector.gate(new, urn, order, lat)[rows]
            full = perms[order]
            shared[order] = {str(s): bool(np.array_equal(full[detector.C.SES[full] == s],
                                                          perm[detector.C.SES[perm] == s]))
                             for s in detector.SESS}
        acquired[p].append(lineage.metric.person_accuracy(new, labels, cells, people, lineage.PEOPLE[p]))

    endpoints, delta, cell_counts = {}, {}, {}
    for p, old, _, cells, _ in lineage.guarded_matrices(observe):
        new = np.array(acquired[p])
        delta[p] = new - old
        cell_counts[p] = cells
        assert np.array_equal(old - old, np.zeros_like(old))
        endpoints[p] = {}
        for name, weights in ([("official", cells), ("assumed_returning", RETURN)]
                              if p == "seen_session" else [("official", cells)]):
            expected = owner[p]["three_minus_old"][name]
            for label, matrix in (("old", old), ("new", new), ("delta", new - old)):
                value = float(np.mean([np.average(matrix[g].mean(0), weights=weights) for g in GROUPS.values()]))
                np.testing.assert_allclose(value, expected[label], atol=1e-10, rtol=0)
                endpoints[p].setdefault(name, {})[label] = value
            np.testing.assert_allclose(np.average(new-old, axis=1, weights=weights),
                                       expected["delta_by_order"], atol=1e-10, rtol=0)
    hashes = json.loads((EVIDENCE / "lineage419_result.json").read_text())["sha256"]
    assert all(lineage.LOCK[path] == digest for path, digest in hashes.items())
    return delta["all_loso"], delta["seen_session"], cell_counts["all_loso"], endpoints, shared


def normalized(weights):
    return weights / weights.sum(axis=-1, keepdims=True)


def evaluate(new, ret, cells, person):
    n = normalized(person * cells) @ new.T
    r = normalized(person[..., SEEN] * RETURN) @ ret.T
    return NEW * n + (1 - NEW) * r


def summarize(values):
    return {s: float(values[g].mean()) for s, g in GROUPS.items()}


def main():
    new, ret, cells, endpoints, shared = verified_pairs()
    weights = np.ones(len(PEOPLE))
    point = evaluate(new, ret, cells, weights)
    np.testing.assert_array_equal(evaluate(new*0, ret*0, cells, weights), np.zeros(len(ORDERS)))
    loo, doubles = {}, {}
    for count, target in ((1, loo), (2, doubles)):
        for omit in itertools.combinations(range(len(PEOPLE)), count):
            if set(SEEN).issubset(omit):
                continue
            w = weights.copy()
            w[list(omit)] = 0
            target["/".join(PEOPLE[i] for i in omit)] = summarize(evaluate(new, ret, cells, w))
    rng = np.random.default_rng(450)
    draws = 20000
    per_order = evaluate(new, ret, cells, rng.exponential(size=(draws, len(PEOPLE))))
    schemes = {s: np.take_along_axis(per_order, rng.choice(g, (draws, len(g))), axis=1).mean(1)
               for s, g in GROUPS.items()}
    schemes["equal"] = np.mean(list(schemes.values()), axis=0)
    new_order = np.average(new, axis=1, weights=cells)
    return_order = ret @ RETURN
    cross_orders = {s: NEW * new_order[g, None] + (1 - NEW) * return_order[g] for s, g in GROUPS.items()}
    report = dict(all_within_scheme_order_pair_ranges={s: [float(v.min()), float(v.max())]
                                                       for s, v in cross_orders.items()},
                  people=PEOPLE, new_cells=cells.tolist(), endpoints=endpoints,
                  mixture=NEW, returning_weights=RETURN.tolist(), orders=dict(zip(ORDERS, point)),
                  by_scheme=summarize(point), shared_session_permutations_identical=shared,
                  linked_weight_interval80={s: np.quantile(v, [.1, .9]).tolist() for s, v in schemes.items()},
                  leave_one=loo, leave_two=doubles,
                  role_corners={s: summarize(v) for s, v in (("new", np.average(new,axis=1,weights=cells)),
                                                             ("return_P04", ret[:,0]), ("return_P10", ret[:,1]))})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
