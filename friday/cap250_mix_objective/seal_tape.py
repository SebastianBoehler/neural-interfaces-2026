"""Authenticate and combine all fixed source sessions, and qualify native work metadata."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


EXTERNAL = "36fa4e8a8d630e0e79781452dc91850837eb1d4984a0c5cac8d435dca19c301a"
FIELDS = ("stock_mix","frozen","warm","active","probe48","probe52","age","stages",
          "stock_class","upstream_class","recognized")


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def stamp(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def open_signed(path):
    seal = path.with_suffix(".sha256")
    if (path.stat().st_mode | seal.stat().st_mode) & 0o222 or sha(path) != seal.read_text().strip():
        raise ValueError(f"source tape unsigned or writable: {path}")
    return np.load(path,allow_pickle=False)


def operational(root,source_sessions,replaceable):
    path = root/"before_labels.json"
    if sha(path) != EXTERNAL:
        raise ValueError("earlier independent external native work seal differs")
    receipt = json.loads(path.read_text())
    assert not receipt["held_labels_opened"] and receipt["orders"] == [8411,9411]
    lengths, steps, seen = [], [], set()
    for name,pin in receipt["workers"].items():
        path = root/name
        assert sha(path) == pin
        observed = json.loads(path.read_text())
        assert observed["source"] == receipt["source"] and observed["passed"]
        assert observed["shared_protected"] and observed["distinct_outer"] and not observed["held_labels_opened"]
        for session, r in observed["sessions"].items():
            who,visit,delivery = session.split("|")
            assert who==observed["who"] and visit in ("1","2") and int(delivery) in receipt["orders"]
            assert session not in seen
            seen.add(session)
            w = r["activity"]
            n = len(r["rows"])
            expected = len(range(20,min(251,n),10))
            assert n == w["windows"] and len(set(r["rows"])) == n
            assert w["recognized"] == -1 and w["errors"] == w["ft_over"] == 0
            assert r["warm_mix"] == n-20 and r["stop"] is None
            assert w["ft_refits"] == w["ft_relabels"] == expected and w["ft_steps"] == 20*expected
            assert r["candidate_fit_ages"] == list(range(20,min(251,n),20))
            lengths.append(n)
            steps.append(w["ft_steps"])
    if len(seen) != 36 or replaceable != 5/96 or len(source_sessions) != 14:
        raise ValueError("source and external native operational model differ")
    lens = [m["forced"]["work"]["windows"] for m in source_sessions.values()]
    return dict(reference=EXTERNAL,external_sessions=len(seen),external_window_range=[min(lengths),max(lengths)],
                external_ft_step_range=[min(steps),max(steps)],source_sessions=len(lens),
                source_window_range=[min(lens),max(lens)],fixed_native_mass=replaceable,
                common="unrecognized cap250 R15 path; strict past; first optimizer update at age 20; original 10-window cadence",
                inference_uses_no_subject_or_visit_metadata=True,mathematical_scope_only=True,
                source_recognition_suppressed=True,natural_external_recognition=True,held_labels_opened=False)


def main(args):
    with open_signed(Path("tape-smoke.npz")) as z:
        smoke = json.loads(str(z["metadata"]))
    source = sha("source.sha256")
    assert smoke["mode"] == "smoke" and smoke["conditioning_forecast_passed"] and smoke["source"] == source
    assert not smoke["held_labels_opened"] and not smoke["external_opened"]
    assert sha(args.input) == smoke["data"] and sha(args.orders) == smoke["delivery"]
    with np.load(args.input,allow_pickle=False) as z:
        names = np.array([f"{a.rsplit('/',1)[-1]}-{b}" for a,b in zip(z["subject"],z["session"],strict=True)])
        x = z["X"]
    with np.load(args.orders,allow_pickle=False) as z:
        order = z["full|0|all_loso"]
    np.testing.assert_array_equal(np.sort(order),np.arange(2655))
    assert stamp(order)==smoke["full0"]
    stored, session_reports, hashes = {}, {}, {}
    metadata = ("archive","data","delivery","source","full0","canary","replaceable_nominal_mass")
    coverage = np.zeros(len(x),dtype=bool)
    weights = rest = None
    sessions = sorted(set(names))
    assert len(sessions) == 14 and len(x) == 2655
    for i,session in enumerate(sessions):
        path = Path(f"tape-{i}.npz")
        hashes[path.name] = sha(path)
        rows = order[names[order]==session]
        with open_signed(path) as z:
            m = json.loads(str(z["metadata"]))
            assert m["index"] == i and m["session"] == session and m["mode"] == "capture"
            assert all(m[k] == smoke[k] for k in metadata)
            assert not m["external_opened"] and not m["held_labels_opened"]
            assert m["rows"]==stamp(rows) and m["work_identity_input"] and m["interleaved_prefix"]
            assert max(m["protected_max"].values(),default=1.) <= 5e-5
            assert not coverage[rows].any()
            coverage[rows] = True
            for a in ("natural","forced"):
                v = m[a]
                assert v["x_sha"]==stamp(x[rows]) and v["all_stages_zero_exact"] and v["prefix_exact"]
                assert v["gradient"]["active"] >= .95*3*len(rows)
                assert not v["work"]["errors"] and not v["work"]["ft_over"]
                np.testing.assert_array_equal(z[f"{a}|rows"],rows)
                np.testing.assert_array_equal(z[f"{a}|age"],np.arange(len(rows)))
                w,b = z[f"{a}|weights"],z[f"{a}|rest"]
                if weights is None:
                    weights,rest = w,b
                np.testing.assert_array_equal(weights,w)
                np.testing.assert_array_equal(rest,b)
                for k in FIELDS:
                    name = f"{a}|{k}"
                    p = z[name]
                    assert len(p) == len(rows) and np.isfinite(p).all()
                    if name not in stored:
                        stored[name] = np.empty((len(x),*p.shape[1:]),dtype=p.dtype)
                    stored[name][rows] = p
        session_reports[session] = m
    assert coverage.all() and len(hashes)==14
    stats = [s["forced"]["gradient"]["warm_tangent"] for s in session_reports.values()]
    warm = sum(s["rows"] for s in stats)
    rank2 = sum(s["rank2"] for s in stats)
    sigma = float(np.average([v["median_sigma2"] for v in stats],weights=[v["rows"] for v in stats]))
    if warm != 2375 or rank2 < .90*warm or sigma < 1e-4:
        raise ValueError("pooled forced warm source tangent does not meet its registered criterion")
    proof = operational(args.external,session_reports,smoke["replaceable_nominal_mass"])
    source_npz = Path("source-tape.npz")
    with source_npz.open("xb") as f:
        np.savez_compressed(f,index=np.arange(len(x)),delivery=order,session=names,weights=weights,rest=rest,**stored)
    source_npz.chmod(0o444)
    receipt = Path("source-tape.json")
    info = {k:smoke[k] for k in metadata}
    with receipt.open("x") as f:
        json.dump(dict(info,seal=sha("seal.sha256"),smoke=sha("tape-smoke.npz"),members=hashes,bank=sha(source_npz),
                       rows=len(x),folds=session_reports,operational=proof,warm_rows=warm,rank2=rank2,
                       weighted_session_median_sigma2=sigma,external_predictions_created=False,
                       held_labels_opened=False,qualified=True),f,sort_keys=True,allow_nan=False)
        f.write("\n")
    receipt.chmod(0o444)
    print("SEALED F-C31c tape",sha(receipt),sha(source_npz),warm,rank2,sigma,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path)
    p.add_argument("orders",type=Path)
    p.add_argument("external",type=Path)
    main(p.parse_args())
