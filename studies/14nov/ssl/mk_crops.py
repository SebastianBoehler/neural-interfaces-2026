"""s277 form 2 (Mac): cut data/proteus_windows_start{0.5,2,3,4}.npz to the 26 intersect channels (float16) as runs/crops/crop_<off>.npz. Usage: mk_crops.py OUTDIR"""
import sys, numpy as np
names = ['C2','CP2','Pz','CPz','Cz','C1','CP1','P3','CP5','CP3','C3','P4','CP6','CP4','C6','C4','FC6','FC4','FC2','F4','F3','Fz','FC5','FC3','FC1','C5']  # runs/ext27/meta.json
d0 = np.load("data/proteus_windows.npz", allow_pickle=True); chn = [str(c) for c in d0["ch_names"]]; idx = [chn.index(n) for n in names]
for nm in ("start0.5", "start2", "start3", "start4"):
    w = np.load(f"data/proteus_windows_{nm}.npz", allow_pickle=True)
    assert (w["subject"] == d0["subject"]).all() and (w["session"] == d0["session"]).all()
    np.savez(f"{sys.argv[1]}/crop_{nm}.npz", X=w["X"][:, idx].astype(np.float16), names=np.array(names))
