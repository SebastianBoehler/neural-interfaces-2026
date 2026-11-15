"""s277 (a): inventory of the external windows exports (headers only for X; small arrays read whole)."""
import glob, os, sys, zipfile
import numpy as np
W = os.environ.get("NI26_DATA", "data")
pro = [str(c) for c in np.load(f"{W}/proteus_windows.npz")["ch_names"]]
print("PROTEUS ch", len(pro))
for corp in ("dreyer2023", "lee2019_mi", "cho2017", "physionetmi", "schirrmeister2017"):
    tot, people, sess, cls, ip_sets, layouts = 0, set(), set(), set(), set(), set()
    for f in sorted(glob.glob(f"{W}/{corp}/windows_*.npz")):
        with zipfile.ZipFile(f) as z:
            with z.open("X.npy") as fh:
                v = np.lib.format.read_magic(fh); sh, _, dt = (np.lib.format.read_array_header_1_0 if v == (1, 0) else np.lib.format.read_array_header_2_0)(fh)
        with np.load(f, allow_pickle=True) as w:
            keys = w.files; subj = w["subject"].astype(str)
            people |= set(subj); sess |= set(zip(subj, w["session"].astype(str)))
            cls |= {str(c) for c in w["classes"]} if "classes" in keys else set()
            ip_sets.add(tuple(sorted(str(c) for c in w["interpolated"])) if "interpolated" in keys else ("NOKEY",))
            layouts.add(tuple(str(c) for c in w["ch_names"]) == tuple(pro))
            ylab = np.unique(w["y"]).tolist()
        tot += sh[0]
        print(f"  {os.path.basename(f)} X{sh} {dt} keys {keys} labels {ylab} people {len(set(subj))}", flush=True)
    ips = sorted(ip_sets)
    print(f"{corp}: windows {tot} people {len(people)} sessions {len(sess)} classes {sorted(cls)} same41layout {layouts} "
          f"n_interpolated {[len(i) for i in ips]} identical_sets {len(ips) == 1} interpolated {ips[0]}", flush=True)
