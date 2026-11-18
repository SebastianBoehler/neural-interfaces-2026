"""Cross delivery and person deletion, then cross-fit the frozen release decision."""
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
spec = importlib.util.spec_from_file_location("lineage419", HERE.parents[1] / "lineage419/score.py")
lineage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lineage)

EVIDENCE = ROOT / "evidence/next_wave_16nov"
PEOPLE = list(lineage.metric.PEOPLE)
ORDERS = lineage.metric.ORDERS
GROUPS = {s: [i for i, o in enumerate(ORDERS) if o.startswith(s)] for s in ("within", "full", "runs")}
RET = [PEOPLE.index(p) for p in ("P04", "P10")]
SHARE = .566


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def equal_schemes(matrix):
    return np.mean([matrix[g].mean(axis=0) for g in GROUPS.values()], axis=0)


def stress(schemes, weights, people):
    result = {}
    for size in (0, 1, 2):
        candidates = []
        for omitted in itertools.combinations(range(len(people)), size):
            keep = [i for i in range(len(people)) if i not in omitted]
            for scheme, values in schemes.items():
                value = np.average(values[keep], weights=weights[keep])
                candidates.append((value, scheme, [people[i] for i in omitted]))
        value, scheme, omitted = min(candidates)
        result[size] = dict(low_pp=value, scheme=scheme, omitted=omitted, comparisons=len(candidates))
    return result


def decisions(n, r, cells):
    # Remove the held person's mass in both roles, renormalizing each source role.
    rw = np.zeros(len(PEOPLE))
    rv = np.zeros(len(PEOPLE))
    rw[RET] = [.2, .8]
    rv[RET] = r
    cw = cells / cells.sum()
    new = cw @ n
    ret = rw @ rv
    train = SHARE * (new - cw * n) / (1 - cw) + (1 - SHARE) * (ret - rw * rv) / (1 - rw)
    choice = train > 0
    return dict(fixed_js_pp=SHARE * new + (1-SHARE) * ret,
                nested_pp=SHARE * (cw @ (n * choice)) + (1-SHARE) * (rw @ (rv * choice)),
                js_for=[p for p, use in zip(PEOPLE, choice) if use],
                source_pp=dict(zip(PEOPLE, train))), choice


def interval(n, r, cells):
    # Resample shared order IDs; cross-fit decisions again on the recorded people.
    # Then bootstrap held evaluations in strata so both participant roles stay linked.
    rng = np.random.default_rng(420)
    new_only = [i for i in range(len(PEOPLE)) if i not in RET]
    samples = []
    for _ in range(6000):
        groups = [rng.choice(g, len(g)) for g in GROUPS.values()]
        nv, rv = (np.mean([a[g].mean(axis=0) for g in groups], axis=0) for a in (n, r))
        _, use = decisions(nv, rv, cells)
        shared = rng.integers(2, size=2)
        people = np.r_[rng.choice(new_only, len(new_only)), np.asarray(RET)[shared]]
        retweights = np.array([.2, .8])[shared]
        def score(mask):
            new = np.average((nv * mask)[people], weights=cells[people])
            returning = np.average((rv * mask[RET])[shared], weights=retweights)
            return SHARE * new + (1-SHARE) * returning
        samples.append([score(np.ones(len(PEOPLE))), score(use)])
    return dict(zip(("fixed_js", "nested"), np.percentile(samples, [10,90], axis=0).T.tolist()))


def main():
    matrices = {p: (old, new, cells) for p, old, new, cells, _ in lineage.guarded_matrices()}
    locked = read("lineage419_result.json")
    if lineage.LOCK != locked["sha256"]:
        raise ValueError("§419 source lock differs")
    for p, (old, new, cells) in matrices.items():
        owned = locked["results"][p]["official"]
        np.testing.assert_allclose(np.average(new-old, axis=1, weights=cells), owned["delta_by_order"], atol=1e-10)
        np.testing.assert_allclose(equal_schemes(new-old), owned["delta_by_person"], atol=1e-10)
    old, cap, cells = matrices["all_loso"]
    retold, retcap, _ = matrices["seen_session"]
    with np.load(lineage.RUN / "replay_data/bases.npz", allow_pickle=True) as z:
        labels, cell, owner = (z[k+"|all_loso"] for k in ("y","cell","person"))
    hashes, scores = {}, []
    for o in ORDERS:
        path = lineage.RUN / "release420" / f"JS_{o}_B2_pool.npz"
        sidecar = json.loads(path.with_suffix(".json").read_text())
        if (sidecar["arm"], sidecar["base"], sidecar["order"]) != ("JS", "B2_pool", o):
            raise ValueError(f"foreign JS sidecar: {o}")
        inputs = {Path(k).name: v for k, v in sidecar["inputs_sha256"].items()}
        for source in ("runs/replay_data/bases.npz", "runs/shuffle_perms.npz"):
            if inputs[Path(source).name] != locked["sha256"][source]:
                raise ValueError(f"foreign JS inputs: {o} {source}")
        hashes[path.name] = lineage.metric.sha256(path)
        scores.append(lineage.metric.person_accuracy(lineage.absolute(path, "scores"), labels, cell, owner, PEOPLE))
    n = np.array(scores) - cap
    owner_js = read("js321_pool_contrast.json")["bases"]["B2_pool"]
    np.testing.assert_allclose(equal_schemes(n), [owner_js["people_order_effect"]["people"][p] for p in PEOPLE])
    np.testing.assert_allclose(np.average(equal_schemes(n), weights=cells), owner_js["official_effect"]["mean"])
    sup = read("js321_returning_scheme_supplement.json")
    r = np.array([[sup["person_order_JS_minus_base_pp"][PEOPLE[i]][o] for i in RET] for o in ORDERS])
    np.testing.assert_allclose(equal_schemes(r) @ [.2,.8], sup["supplemental_equal_scheme_JS_minus_base_pp"])
    np.testing.assert_allclose(r.mean(axis=0) @ [.2,.8], sup["primary_uniform8_JS_minus_base_pp"])
    js = {s: decisions(n[g].mean(axis=0), r[g].mean(axis=0), cells)[0] for s, g in GROUPS.items()}
    js["equal_schemes"] = decisions(equal_schemes(n), equal_schemes(r), cells)[0]
    archived = read("scope416.json")
    external = {s: np.array(a["people"]) for s, a in archived["delta"].items()}
    for s, values in external.items():
        np.testing.assert_allclose(values.mean(), archived["delta"][s]["mean"])
    public = {s: (cap-old)[g].mean(axis=0) for s, g in GROUPS.items()}
    report = dict(new_stress=stress(public, cells, PEOPLE),
                  external_stress=stress(external, np.ones(len(archived["people"])), archived["people"]),
                  returning_by_scheme={s: dict(zip(("P04","P10"), (retcap-retold)[g].mean(axis=0)))
                                       for s, g in GROUPS.items()}, js=js, js_interval80=interval(n,r,cells),
                  js_sha256=hashes)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
