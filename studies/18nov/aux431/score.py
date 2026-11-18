"""Apply §419's locked count filter and metric to guarded returning-session pairs."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / "runs/aux431"
spec = importlib.util.spec_from_file_location("lineage", ROOT / "studies/18nov/lineage419/score.py")
lineage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lineage)
detector = lineage.detector
metric = lineage.metric


def main():
    assert (WORK / "replay.txt").read_text().splitlines()[-1] == "OK"
    new, changes, route, pins = [], {}, {}, {}

    def observe(proto, order, rows, perm, labels, cells, owners, old, historical):
        if proto != "seen_session":
            return
        path = WORK / f"{order}.npz"
        pins[order] = metric.sha256(path)
        hc = detector.J.hyp_cache(order, "wprior", detector.X, detector.LAM,
                                  sessions=detector.SESS, proto=proto)
        lat = detector.latch(order)
        with np.load(path) as data:
            assert np.array_equal(rows, data["rows"])
            route[order] = json.loads(str(data["route"]))
            assert {name.split("/")[-1] for name in route[order]} == {"P04|2", "P04|3", "P10|2", "P10|3"}
            for filename, sha in json.loads(str(data["input_sha"])).items():
                assert metric.sha256(ROOT / filename) == sha
            for tag in ("historical", "three"):
                p = np.full((detector.C.N, 3), 1/3)
                p[rows] = data[tag]
                urn = detector.J.filt_joint(p, hc, "urn")
                metric.verify_zero(p, detector.gate(p, urn, order, dict.fromkeys(lat)))
                filtered = detector.gate(p, urn, order, lat)[rows]
                if tag == "historical":
                    metric.verify_zero(filtered, historical)
                else:
                    changes[order] = int(np.sum(filtered.argmax(1) != historical.argmax(1)))
                    new.append(metric.person_accuracy(filtered, labels, cells, owners, ("P04", "P10")))

    for proto, old, historical, weights, n in lineage.guarded_matrices(observe):
        if proto == "seen_session":
            break
    assert proto == "seen_session" and n == 762 and weights.tolist() == [4, 4] and len(new) == 8
    result = {}
    for name, a, b in (("three_minus_old", old, new), ("three_minus_historical", historical, new),
                       ("historical_minus_old", old, historical)):
        result[name] = {key: lineage.summary(a, b, w) for key, w in
                        (("official", weights), ("assumed_returning", np.array([.2, .8])))}
        result[name]["order_person_delta"] = (np.asarray(b) - a).tolist()
    owner = json.loads((ROOT / "evidence/next_wave_16nov/lineage419_result.json").read_text())
    for key in ("official", "assumed_returning"):
        assert result["historical_minus_old"][key] == owner["results"][proto][key]
    result.update(people=["P04","P10"], orders=metric.ORDERS, rows=n, changed_labels=changes,
                  recognition=route, output_sha256=pins)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
