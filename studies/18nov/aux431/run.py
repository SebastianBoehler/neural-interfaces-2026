"""Rebuild the held-returning §245 producer on one pinned delivery order."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
RUN = ROOT / "runs"
OUT = RUN / "aux431"
ORDER = sys.argv[1]
BASE = "B2_pool"
PROTO = "seen_session"
os.environ.update(BASES=BASE, ORDERS=ORDER, ARMS="base,ER,T", PROTO=PROTO)
sys.path.insert(0, str(ROOT / "studies/14nov/pview"))
import chain as chain


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    assert (OUT / "replay.txt").read_text().splitlines()[-1] == "OK"
    receipts = json.loads((ROOT / "evidence/next_wave_16nov/lineage419_result.json").read_text())["sha256"]
    paths = [RUN / f"chainout/ret_{ORDER}.npz", RUN / "chainout/s245_ret.npz"]
    for path in paths:
        assert sha(path) == receipts[str(path.relative_to(ROOT))]
    original, prior = [np.load(path) for path in paths]
    key = f"{ORDER}|{BASE}|"
    r6 = chain.R6
    oof = r6["oof_rec"](BASE)
    perm = r6["perm"](ORDER)
    pp = perm[PROTO]
    inp = r6["inputs"](ORDER)
    for cache in (chain.ns["_BIO"], chain.A._H, chain.K.STREAMS, chain.K.REC):
        cache.clear()
    chain.S.apply(perm)
    chain.K.RANK["cur"] = {q: chain.K._rank(chain.K.IDX[q][perm[q]]) for q in chain.K.PR}
    r6["apply_members"](r6["member_deltas"](ORDER), pp)

    auxiliary = []
    changes = {}
    pins = {}
    for i, (folder, prefix, count) in enumerate([
        ("ocushuf_ss", "oc_nomix", 9),
        ("restshuf", "rest4_sh_r1", 9 if ORDER in ("within0", "within1") else 3),
    ], 1):
        paths = [RUN / folder / ORDER / f"{prefix}-{PROTO}-s{s}.npz" for s in range(count)]
        assert sorted(paths) == sorted(paths[0].parent.glob(f"{prefix}-{PROTO}-s*.npz"))
        pins.update({str(f.relative_to(ROOT)): sha(f) for f in paths})
        streams = [r6["ld"](str(f)) for f in paths]
        def mean(selected):
            x = np.mean(selected, axis=0)
            return np.log(np.clip(x, 1e-3, 1)) if i == 1 else x
        assert np.array_equal(mean(streams), inp[i])
        auxiliary.append(mean(streams[:3]))
        changes[folder] = float(np.max(np.abs(inp[i] - auxiliary[-1])))
        assert (changes[folder] > 0) == (count > 3)

    def stage(arm, inputs):
        out, routes = chain.system(BASE, arm, pp, oof, inputs, None)
        assert len(routes) == 4
        for (_, session), (rows, ramp, person) in routes.items():
            assert person in ("P04", "P10") and person in session and np.any(ramp > 0)
        return out, routes

    anchor = None
    results = {}
    for tag, inputs in (("historical", inp), ("three", (inp[0], *auxiliary))):
        for arm in ("base", "ER", "T"):
            out, routes = stage(arm, inputs)
            if anchor is None:
                anchor = routes
            assert routes.keys() == anchor.keys()
            for sk, (rows, ramp, person) in routes.items():
                arows, aramp, aperson = anchor[sk]
                assert person == aperson and np.array_equal(arows, rows) and np.array_equal(aramp, ramp)
                if arm == "T":
                    pre = rows[:np.flatnonzero(ramp > 0)[0]]
                    np.testing.assert_allclose(out[pre], er[pre], rtol=0, atol=1e-12)
                    out[pre] = er[pre]
            if arm == "ER":
                er = out
            record = out[np.argsort(pp)].astype(np.float32)
            if tag == "historical" or arm == "base":
                ref = (original if arm == "base" else prior)[key + arm]
                assert np.array_equal(record, ref), (tag, arm, float(abs(record-ref).max()),
                                                       int(np.sum(record.argmax(1) != ref.argmax(1))))
                if arm == "ER":
                    assert np.array_equal(record, original[key + arm])
            if arm == "T":
                results[tag] = record
    assert not np.array_equal(results["three"], results["historical"])
    np.savez_compressed(OUT / f"{ORDER}.npz", three=results["three"], historical=results["historical"],
                        rows=chain.IDX, input_sha=json.dumps(pins), aux_change=json.dumps(changes),
                        route=json.dumps({s:int(np.flatnonzero(r > 0)[0]) for (_,s), (_,r,_) in anchor.items()}))
    print("PASS", ORDER, changes, flush=True)


if __name__ == "__main__":
    main()
