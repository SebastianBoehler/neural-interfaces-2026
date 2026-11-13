"""Build the v5.2 package: v5.1b (runs/sealed/cand_v51b_own) plus each public person's fine-tuned SessionBN traces.

Usage: python scripts/build_v52.py [SRC_DIR] [OUT_DIR] [PNET_DIR] [CODE]
  defaults: runs/sealed/cand_v51b_own runs/sealed/cand_v52_pnets runs/pnets_jit submissions/ensemble_v52/submission.py
  (v5.2b: CODE = submissions/ensemble_v52b/submission.py, OUT = runs/sealed/cand_v52b_pnets)
Copies every file of SRC unchanged except submission.py (-> submissions/ensemble_v52/submission.py) and weights.npz
(SRC's arrays unchanged, plus pnet_member: the indices of the 30 SessionBN nets, and pnet_files[m, j]: their personal
traces p_<base stem>_<person>.pt in route_names order), and copies the 300 personal traces (ledger s96, s102).
Fails loudly if OUT exists, a trace is missing or extra, or a source array would change, or (review 13 Nov, item 5) a
personal trace does not sit on its shipped base: the personal trace carrying the shipped output layer and StreamBN affine
must equal the shipped trace (scripts/sbn_trace_base.graft_check < 1e-5); --allow-twin-bases only records the
differences (the v5.2 build of 12 Nov, whose raw FBCNet and SincShallow traces sit on retrained twins). Prints a JSON summary.
"""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "submissions/ensemble_v52/submission.py"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    twin_ok = "--allow-twin-bases" in sys.argv; a = [x for x in sys.argv[1:] if x != "--allow-twin-bases"] + [None] * 4
    src = Path(a[0] or ROOT / "runs/sealed/cand_v51b_own"); out = Path(a[1] or ROOT / "runs/sealed/cand_v52_pnets")
    pdir = Path(a[2] or ROOT / "runs/pnets_jit"); code = Path(a[3] or CODE)
    assert not out.exists(), f"{out} exists"
    z = dict(np.load(src / "weights.npz", allow_pickle=False)); assert not any(k.startswith("pnet_") for k in z)
    names = [str(n) for n in z["route_names"]]; w = z["member_weights"]
    member, files = [], []
    for i in range(int(z["n_nets"])):
        f = z.get(f"net{i}_sjit_file")
        if f is not None and str(f).startswith("sbn_"):
            assert w[1 + i] > 0, f"{f}: member weight 0"
            member.append(i); files.append([f"p_{str(f)[:-3]}_{q}.pt" for q in names])
    assert len(member) == 30, f"expected the 30 SessionBN members, found {len(member)}"
    want = {f for row in files for f in row}; have = {p.name for p in pdir.glob("p_*.pt")}
    assert want == have, f"traces missing {sorted(want - have)[:5]} extra {sorted(have - want)[:5]}"
    sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "experiments"))
    import torch
    from sbn_trace_base import graft_check
    graft = {}
    for i, row in zip(member, files):
        shipped = torch.jit.load(str(src / str(z[f"net{i}_sjit_file"])), map_location="cpu")
        graft[str(z[f"net{i}_sjit_file"])] = max(graft_check(torch.jit.load(str(pdir / f), map_location="cpu"), shipped) for f in row)
    bad = {k: v for k, v in graft.items() if v >= 1e-5}
    assert twin_ok or not bad, f"personal traces not on their shipped base (max |dp| after grafting): {bad}"
    out.mkdir(parents=True, exist_ok=False)
    for f in src.iterdir():
        if f.is_file() and f.name not in ("weights.npz", "submission.py"):
            shutil.copy2(f, out / f.name)
    shutil.copy2(code, out / "submission.py")
    for f in sorted(want):
        shutil.copy2(pdir / f, out / f)
    z2 = dict(z, pnet_member=np.asarray(member, dtype=np.int64), pnet_files=np.asarray(files))
    np.savez(out / "weights.npz", **z2)
    back = dict(np.load(out / "weights.npz", allow_pickle=False))
    for k, v in z.items():  # every v5.1b array unchanged (dtype, shape, bytes)
        assert back[k].dtype == v.dtype and np.array_equal(back[k], v), k
    assert back["pnet_files"].shape == (30, len(names)) and back["pnet_files"].dtype.kind == "U"
    for f in src.iterdir():
        if f.is_file() and f.name not in ("weights.npz", "submission.py"):
            assert sha(f) == sha(out / f.name), f.name
    size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(json.dumps(dict(src=str(src), out=str(out), graft_max=max(graft.values()), twin_bases=bad, members=member, people=names, traces=len(want),
                          files=len(list(out.iterdir())), bytes=size, submission_sha256=sha(out / "submission.py"),
                          weights_sha256=sha(out / "weights.npz"),
                          traces_sha256=hashlib.sha256("".join(sha(out / f) for f in sorted(want)).encode()).hexdigest()), indent=1))


if __name__ == "__main__":
    main()
