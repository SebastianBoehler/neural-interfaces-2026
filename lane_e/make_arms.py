"""Lane E: build the capacity arms from the signed cap250 archive (no retraining of any shipped member).

python lane_e/make_arms.py STOCK.zip MEMBERS_DIR OUT_DIR [--guard SECONDS] [--arms E0,Ea,...]
Arms (evidence/wave_19nov/laneE_capacity.md): E0 = lane_e code, cap250 keys (must be bitwise cap250); Ea = Eye/Rest seeds
3+3 -> 6+6 (eye_s3-5, rest_s3-5 from MEMBERS_DIR); Eb = fine-tuning replicas 4 -> 8 (seed-1 member of each family);
Ec = the 10 supported zero-weight neural terms (nets 10-19; 55 of 65 active) at weight EC_W; Ed = Ea + Eb. --guard sets v6_ft_session_s (cap250: 1e9).
Every array not named here is copied unchanged and asserted equal after writing.
"""
import argparse
import hashlib
import io
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np

STOCK_SHA = "9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed"
SOURCE_SHA = "37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f"
ROOT = Path(__file__).resolve().parents[1]
EC_NETS = np.arange(10, 20)  # the zero-weight nets the bundle supports (FBC); 24-33 are unsupported exports
EC_W = 0.025  # raw weight of each formerly zero-weight net (the weight of nets 0-9; normalised with the rest at load)
SEED1 = {"fbraw": 48, "fb": 36, "sinc": 54, "fbl": 60}  # second member of each family slot (v6_slot_* lists seeds 0-5)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def keys_for(arm, z, members):
    new = {}
    if arm in ("Ea", "Ed"):
        files = [f"eye_s{k}.pt" for k in range(6)], [f"rest_s{k}.pt" for k in range(6)]
        assert all((members / f).exists() for f in files[0][3:] + files[1][3:]), "extra seed files missing"
        for k in range(3, 6):  # the rest input stream must be the shipped one (build6.py asserts the same)
            r = np.load(members / f"rest_s{k}.npz")
            for key in ("align", "k", "G", "scale", "shrink"):
                # the cluster's Gram mean differs from the shipped one by summation order only (3e-10 of 2269): 1e-9 relative
                assert np.allclose(r[key], z[f"v6_rest_{key}"], rtol=1e-9, atol=0), f"rest_s{k} input stream differs in {key}"
        new["v6_eye_files"], new["v6_rest_files"] = np.array(files[0]), np.array(files[1])
    if arm in ("Eb", "Ed"):
        slots = [str(s) for s in z["v6_slots"]]
        ex = np.array([[SEED1[s]] for s in slots], np.int64)
        assert all(ex[j, 0] in z[f"v6_slot_{s}"] for j, s in enumerate(slots))
        new["v6_ft_extra"] = ex
    if arm == "Ec":
        w = z["member_weights"].copy()
        off = np.flatnonzero(w[1:-1] == 0)
        assert len(off) == 20 and off[0] == 10 and off[-1] == 33, off
        w[1 + EC_NETS] = EC_W  # nets 24-33 are exports (sinc / FBC variants) this bundle cannot run: build_net raises
        new["member_weights"] = w
    return new


def build(stock, members, out, arms, guard, suffix=""):
    data = stock.read_bytes()
    assert sha(data) == STOCK_SHA, "not the signed cap250 archive"
    code = (ROOT / "submissions/lane_e_capacity/submission.py").read_bytes()
    out.mkdir(parents=True, exist_ok=True)
    with ZipFile(stock) as zf:
        assert sha(zf.read("submission.py")) == SOURCE_SHA
        items = [(i, zf.read(i)) for i in zf.infolist()]
    z = np.load(io.BytesIO(dict((i.filename, d) for i, d in items)["weights.npz"]), allow_pickle=False)
    z = {k: z[k] for k in z.files}
    for arm in arms:
        new = keys_for(arm, z, members)
        if guard is not None:
            new["v6_ft_session_s"] = np.float64(guard)
        w = dict(z, **new)
        assert list(w) == list(z) or set(new) - set(z), "key set changed unexpectedly"
        buf = io.BytesIO()
        np.savez(buf, **w)
        name = arm + suffix
        path = out / f"{name}.zip"
        with ZipFile(path, "x", ZIP_STORED) as t:
            for i, d in items:
                t.writestr(i, code if i.filename == "submission.py" else buf.getvalue() if i.filename == "weights.npz" else d)
            names = {i.filename for i, _ in items}
            for f in sorted(p.name for p in members.glob("*.pt")):
                if f not in names and arm in ("Ea", "Ed"):
                    t.writestr(f, (members / f).read_bytes())
        with ZipFile(path) as t, np.load(io.BytesIO(t.read("weights.npz")), allow_pickle=False) as back:
            for k in z:  # every array is the stock one unless this arm names it
                assert (k in new) or np.array_equal(back[k], z[k]), k
            for k in new:
                assert np.array_equal(back[k], new[k]), k
            for f in list(back["v6_eye_files"]) + list(back["v6_rest_files"]):
                assert str(f) in t.namelist(), f
        print(f"{sha(path.read_bytes())}  {path}  keys changed: {sorted(new)}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stock", type=Path)
    p.add_argument("members", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--guard", type=float)
    p.add_argument("--arms", default="E0,Ea,Eb,Ec,Ed")
    p.add_argument("--suffix", default="")
    a = p.parse_args()
    build(a.stock, a.members, a.out, a.arms.split(","), a.guard, a.suffix)
