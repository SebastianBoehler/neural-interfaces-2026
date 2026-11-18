"""Pair the historical public fold outputs; run from repository root with the project Python."""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs"
STORE = RUN / "lineage419"
sys.path.insert(0, str(ROOT / "studies/14nov/newmember"))
import score_s_v66p as metric

os.environ["PROTO"] = "seen_session"
sys.path.insert(0, str(ROOT / "studies/14nov/rundetect"))
import package as detector

PEOPLE = {"all_loso": metric.PEOPLE, "seen_session": ("P04", "P10")}
LOCK = {}


def pin(path, expected=None):
    digest = metric.sha256(path)
    if expected and digest != expected:
        raise ValueError(f"checksum mismatch for {path}")
    LOCK[str(path.relative_to(ROOT))] = digest
    return digest


def absolute(path, zkey):
    with np.load(path) as z:
        return z[zkey]


def summary(old, new, weights):
    old, new = np.asarray(old), np.asarray(new)
    groups = [[i for i, o in enumerate(metric.ORDERS) if o.startswith(s)] for s in ("within", "full", "runs")]
    def average(values):
        return np.mean([values[g].mean(axis=0) for g in groups], axis=0)
    delta = new - old
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(4000):
        people = rng.integers(0, len(weights), len(weights))
        draw = np.mean([delta[rng.choice(g, len(g))][:, people].mean(axis=0) for g in groups], axis=0)
        samples.append(np.average(draw, weights=weights[people]))
    return dict(old=float(np.average(average(old), weights=weights)), new=float(np.average(average(new), weights=weights)),
                delta=float(np.average(average(delta), weights=weights)), interval80=np.percentile(samples,[10,90]).tolist(),
                old_by_person=average(old).tolist(), new_by_person=average(new).tolist(),
                delta_by_person=average(delta).tolist(), delta_by_order=np.average(delta, axis=1, weights=weights).tolist())


def guarded_matrices():
    if (STORE/"replay.txt").read_text().splitlines()[-1] != "OK":
        raise ValueError("replay/check.py must pass")
    perm = RUN/"shuffle_perms.npz"
    pin(perm, "6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f")
    manifest = STORE/"control-original.sha256"
    pin(manifest, "42782d6c2b7de4a8adddfcc5282eadbc0294dd84185a31a2706a90856eb35660")
    hashes = {Path(n).name:h for h,n in (row.split() for row in manifest.read_text().splitlines())}
    for name, digest in (("v51b_uploaded","4a0bc427f2c9c862ee779db5766badeae93f783446561f4dc8abb43589933bef"),
                         ("v66_cp1_cap250","9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed")):
        pin(RUN/f"sealed/final_zips/{name}.zip",digest)
    base = RUN/"replay_data/bases.npz"
    pin(base)
    for path in ("runs/rundetect/decisions.npz","runs/rundetect/proj_d6_r1.npz",
                 "data/proteus_windows.npz","studies/14nov/rundetect/package.py",
                 "studies/14nov/rundetect/jointcount.py","studies/14nov/rundetect/structured.py"):
        pin(ROOT/path)
    if (detector.J.TAU,detector.J.FLOOR,detector.C.NDIM,detector.C.REG)!=(2,.5,6,1):
        raise ValueError("detector configuration differs")
    for protocol, people in PEOPLE.items():
        with np.load(base,allow_pickle=True) as z:
            rows, owners, cells, labels = [z[f"{k}|{protocol}"] for k in ("idx","person","cell","y")]
        before, after = [], []
        for o in metric.ORDERS:
            order = absolute(perm, f"{o[:-1]}|{o[-1]}|{protocol}")
            if not np.array_equal(np.sort(rows),np.sort(order)):
                raise ValueError(f"{protocol} {o}: order coverage")
            if protocol == "all_loso":
                raw = STORE/f"s281_v66p_control_{o}.npz"
                cp = STORE/f"control_{o}_B2_pool.npz"
                for path in (raw,cp):
                    pin(path,hashes[path.name])
                ref = RUN/f"s281_v66_reference/s242c250c_{o}.npz"
                route = RUN/f"s281_v66_route/{o}.json"
                metric.verify_route(json.loads(route.read_text()),o,pin(ref),metric.BASES,14)
                pin(route)
                for arm in ("P0r","V6Fr","V6r"):
                    key=f"B2_pool|{arm}"
                    metric.verify_zero(absolute(raw,key),absolute(ref,key))
                old = absolute(raw,"B2_pool|P0r")
                new = absolute(cp,"scores")
                zero = STORE/f"zero318_{o}_B2_pool.npz"
                pin(zero)
                metric.verify_zero(new,absolute(zero,"scores"))
            else:
                raw=RUN/f"chainout/ret_{o}.npz"
                prior=RUN/"chainout/s245_ret.npz"
                pin(raw)
                pin(prior)
                key=f"{o}|B2_pool|"
                metric.verify_zero(absolute(raw,key+"ER"),absolute(prior,key+"ER"))
                old=absolute(raw,key+"base")
                prob=np.full((detector.C.N,3),1/3)
                if not np.array_equal(rows,detector.ROWS):
                    raise ValueError("returning row identity failed")
                prob[rows]=absolute(prior,key+"T")
                lat=detector.latch(o)
                cache=detector.J.hyp_cache(o,"wprior",detector.X,detector.LAM,sessions=detector.SESS,proto=protocol)
                urn=detector.J.filt_joint(prob,cache,"urn")
                metric.verify_zero(prob,detector.gate(prob,urn,o,dict.fromkeys(lat)))
                new=detector.gate(prob,urn,o,lat)[rows]
            before.append(metric.person_accuracy(old,labels,cells,owners,people))
            after.append(metric.person_accuracy(new,labels,cells,owners,people))
        weights=np.array([len(np.unique(cells[owners==p])) for p in people])
        yield protocol, np.asarray(before), np.asarray(after), weights, len(rows)


def main():
    results = {}
    for protocol, before, after, weights, row_count in guarded_matrices():
        people = PEOPLE[protocol]
        results[protocol]=dict(official=summary(before,after,weights), equal_person=summary(before,after,np.ones(len(people))),
                               rows=row_count,cells=int(weights.sum()),people=people)
        if protocol=="seen_session":
            results[protocol]["assumed_returning"]=summary(before,after,np.array([.2,.8]))
    report=dict(orders=metric.ORDERS,scope="historical fold emulation",results=results,sha256=LOCK)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
