"""s277 prep: all external windows on the channel intersect (recorded in EVERY corpus and in PROTEUS's 41 layout), per-session EA,
float16, one array + person ids. Labels are not stored (unlabelled pretraining). Usage: prep_ext.py"""
import glob, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I drops the script dir
import numpy as np
from common import *

t0 = time.time(); pro = [str(c) for c in np.load(protocol.DATA)["ch_names"]]
files = {c: sorted(glob.glob(f"{WS}/data/{c}/windows_*.npz")) for c in CORP}
interp = set(); inv = {}
for c, fs in files.items():
    assert fs, c
    for f in fs:
        with np.load(f, allow_pickle=True) as w:
            assert [str(x) for x in w["ch_names"]] == pro, f"{f}: layout differs from PROTEUS"
            ip = {str(x) for x in w["interpolated"]}
        inv.setdefault(c, set()).add(tuple(sorted(ip))); interp |= ip
chans = [i for i, c in enumerate(pro) if c not in interp]; names = [pro[i] for i in chans]
print("interpolated sets per corpus:", {c: [len(s) for s in v] for c, v in inv.items()}, "union", len(interp), "-> intersect", len(chans), names, flush=True)
Xs, pid, cor, people = [], [], [], {}
for c, fs in files.items():
    n0 = sum(len(x) for x in Xs)
    for f in fs:
        with np.load(f, allow_pickle=True) as w:
            X = w["X"][:, chans].astype(np.float32); subj = np.char.add(f"{c}/", w["subject"].astype(str)); sess = w["session"].astype(str)
        assert X.shape[-1] == NT and np.isfinite(X).all()
        st = np.char.add(np.char.add(subj, "|"), sess)
        Xs.append(ea_noncausal(X, st).astype(np.float16))
        pid.append(np.array([people.setdefault(s, len(people)) for s in subj], np.int32)); cor.append(np.full(len(X), CORP.index(c), np.int8))
    print(f"{c}: {sum(len(x) for x in Xs) - n0} windows, {len([p for p in people if p.startswith(c + '/')])} people, {time.time() - t0:.0f}s", flush=True)
X = np.concatenate(Xs); pid = np.concatenate(pid); cor = np.concatenate(cor)
assert np.isfinite(X.astype(np.float32)).all(), "fp16 overflow"
os.makedirs(EXT, exist_ok=True); np.save(EXT / "X.npy", X); np.save(EXT / "pid.npy", pid); np.save(EXT / "corpus.npy", cor)
json.dump(dict(chans=chans, names=names, people=list(people), corpora=list(CORP), n=int(len(X)),
               per_corpus={c: int((cor == i).sum()) for i, c in enumerate(CORP)}, std=float(X[::50].astype(np.float32).std())), open(EXT / "meta.json", "w"))
print("saved", X.shape, "people", len(people), f"std {X[::50].astype(np.float32).std():.3f} {time.time() - t0:.0f}s", flush=True)
