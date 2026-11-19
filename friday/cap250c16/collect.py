"""Seal the four conditional returning visits before computing paired scores."""

import hashlib
import json
from pathlib import Path

import numpy as np

from assets import ANCHORS, digest, meta, sha
from geometry import EEG, row_labels
from inputs import BASE, BASELINES
from protocol import score
from replay_head import AGG, AGG_SHA, ORDER, VISITS


DELIVERY = BASE / "ni26_aux431_20261118/runs/shuffle_perms.npz"
PEOPLE = ("P04", "P10")


def seal(name, data):
    file = Path(name)
    with file.open("x") as f:
        json.dump(data, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    file.chmod(0o444)
    return sha(file)


def locked_bank(root, people, days):
    manifest = sha(root/"replay.sha256")
    smoke = meta(root/"smoke/result.json")
    assert smoke["scope"] == "smoke" and smoke["visit"] == "P10|2" and smoke["n"] == 39
    assert smoke["no_held_gold"] and smoke["source"] == manifest and smoke["parents"] == AGG_SHA
    assert smoke["artifact"] == sha(root/"smoke/result.npz")
    assert sha(AGG) == AGG_SHA
    with np.load(DELIVERY, allow_pickle=False) as p:
        order = p["full|0|seen_session"]
    canonical = np.sort(order)
    labels = np.full((len(ORDER), len(order)), -1, dtype=np.int8)
    proof, scopes = {}, {}
    for i, visit in enumerate(VISITS):
        who, day = visit.split("|")
        path = root/f"visit{i}"
        result = meta(path/"result.json")
        expect = order[(people[order] == f"Dreyer2026Proteus/{who}") & (days[order] == day)]
        assert result["visit"] == visit and result["n"] == len(expect) and expect[0] == ANCHORS[visit]
        assert result["locked_native_parity"] and result["exact_prefix"] and result["no_held_gold"]
        assert result["source"] == manifest and result["scope"] == "complete" and result["parents"] == AGG_SHA
        assert result["artifact"] == sha(path/"result.npz")
        head = root/"fit"/f"{who}.npz"
        assert (result["fit"], result["fit_proof"]) == (sha(head), sha(head.with_suffix(".json")))
        cursor = np.searchsorted(canonical, expect)
        assert (labels[:, cursor] == -1).all()
        with np.load(path/"result.npz", allow_pickle=False) as z:
            np.testing.assert_array_equal(z["rows"], expect)
            base = z["native|scores"]
            latch = z["native|latch"]
            ramps = np.maximum(z["native|contrast"][:, 2], z["native|identity"][:, 2])
            t = BASELINES[visit]
            assert int(np.flatnonzero(latch)[0]) == t and not (ramps[:t] > 0).any()
            for j, arm in enumerate(ORDER):
                p, predicted = z[f"{arm}|scores"], z[f"{arm}|labels"]
                np.testing.assert_array_equal(p.argmax(1), predicted)
                np.testing.assert_array_equal(p[:t], base[:t])
                assert np.isfinite(p).all() and np.allclose(p.sum(1), 1., atol=2e-6)
                for d in ("latch", "contrast", "identity"):
                    np.testing.assert_array_equal(z[f"{arm}|{d}"], z[f"native|{d}"])
                if arm == "placebo":
                    np.testing.assert_array_equal(p, base)
                if arm != "native":
                    r = result["arms"][arm]
                    assert r["calls"] == len(expect) and r["positive"] == int((ramps > 0).sum())
                    assert r["max_mixture_error"] <= 1e-14
                assert result["arms"][arm]["native"] == result["arms"]["native"]["native"]
                labels[j, cursor] = predicted
        proof[visit] = {"receipt": sha(path/"result.json"), "output": result["artifact"]}
        scopes[visit] = result["arms"]
    assert len(canonical) == 762 and len(set(canonical)) == 762 and (labels >= 0).all()
    assert np.array_equal(labels[0], labels[1])
    proof = dict(artifacts=proof, scopes=scopes, replay_manifest=manifest, source_fit=sha(root/"fit.sha256"),
                 scorer=sha(root/"scoring.sha256"), row_sha=digest(canonical),
                 prediction_sha=hashlib.sha256(labels.tobytes()).hexdigest(), no_held_gold=True,
                 rows=762, order="full0", arms=ORDER, parent_archive=AGG_SHA)
    return canonical, labels, proof


def paired_metrics(y, labels, people, days, interface):
    stream = np.char.add(np.char.add(people, "|"), days)
    d = dict(stream=stream, interface=interface)
    by_visit, by_person, all_cells = {}, {}, {}
    for arm, p in zip(ORDER, labels, strict=True):
        def official(mask):
            return 100*score(y[mask], p[mask], d, mask)["balanced_accuracy_cell_mean"]
        by_visit[arm] = {v: official(stream == f"Dreyer2026Proteus/{v}") for v in VISITS}
        by_person[arm] = {s: official(people == f"Dreyer2026Proteus/{s}") for s in PEOPLE}
        all_cells[arm] = official(np.ones(len(y), dtype=bool))
    return by_visit, by_person, all_cells


def main():
    root = Path.cwd()
    with np.load(EEG, allow_pickle=False) as f:
        people, days, interface = (f[k] for k in ("subject", "session", "interface"))
    canonical, bank, proof = locked_bank(root, people, days)
    lock = seal("before_labels.json", proof)
    labels = row_labels(EEG, canonical)
    assert len(labels) == 762
    per_visit, per_person, absolute = paired_metrics(labels, bank, people[canonical], days[canonical], interface[canonical])
    weight = {"P04": .2, "P10": .8}
    contrasts = {}
    for a, b in (("own", "native"), ("own", "population"), ("population", "native")):
        change = {p: per_person[a][p]-per_person[b][p] for p in PEOPLE}
        key = f"{a}_vs_{b}"
        contrasts[key] = dict(person_pp=change, pp_weighted=sum(weight[p]*change[p] for p in PEOPLE),
                              pp_all_cells=absolute[a]-absolute[b],
                              by_visit_pp={v: per_visit[a][v]-per_visit[b][v] for v in VISITS},
                              refuted=all(v<=0 for v in change.values()))
    advance = True
    for control, threshold in (("native", .50), ("population", .30)):
        value = contrasts[f"own_vs_{control}"]
        advance &= value["pp_weighted"] >= threshold and min(value["person_pp"].values()) > 0
    final = dict(lock_sha=lock, scope="CONDITIONAL PUBLIC §321/D-EArN; two known people, one order",
                 official_balanced_accuracy_pct=absolute, by_visit_pct=per_visit, by_person_pct=per_person,
                 contrasts=contrasts, weighted_people=weight, qualified_for_stronger_validation=advance,
                 forecast_pp=dict(own_vs_native=.65, own_vs_population=.50), artifact_proof=proof)
    receipt = seal("result.json", final)
    print("PASS F-C16 conditional paired score", receipt, flush=True)
    print(json.dumps(final, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
