"""Pseudo-label files used for fine-tuning: accuracy against the true labels (a leak would show as ~1.0), whether the composite
files (stk, rts, blend), which no tracked script writes, equal the parts the ledger says they are made of, and file times."""
import glob, os, time
import numpy as np
REPO = os.path.abspath(__file__ + "/../../../.."); D = REPO + "/runs/labels"; B = np.load(REPO + "/runs/replay_data/bases.npz", allow_pickle=True)
y = {p: B[f"y|{p}"].astype(int) for p in ("all_loso", "seen_session")}; CH = np.load(REPO + "/studies/12nov/chain_final.npz")
acc = lambda a, p: float(np.mean(a.argmax(1) == y[p]))
rows = {}
for f in sorted(glob.glob(D + "/chain_*.npz")):
    z = np.load(f); n = os.path.basename(f)[6:-4]; assert (z["idx_seen"] == CH["idx_seen"]).all()
    rows[n] = z; kind = n.split("_")[0]
    if kind in ("true", "truelow", "agree", "confhalf", "indep", "stack"):
        print(f"{n:16s} new {acc(z['all_loso'], 'all_loso'):.4f} seen {acc(z['seen_session'], 'seen_session'):.4f}")
for kind in ("v51b", "v6r", "rt", "rts", "stk", "blend"):
    ks = [k for k in rows if k.startswith(kind + "_")]
    a = [acc(rows[k]["all_loso"], "all_loso") for k in ks]; s = [acc(rows[k]["seen_session"], "seen_session") for k in ks]
    seen_is_final = sum(np.array_equal(rows[k]["seen_session"], CH["seen_session"]) for k in ks)
    print(f"{kind:6s} {len(ks):2d} files: new acc {min(a):.4f}-{max(a):.4f}, seen acc {min(s):.4f}-{max(s):.4f}; seen rows identical to chain_final (recording order): {seen_is_final}/{len(ks)}; "
          f"mtime {time.strftime('%d %H:%M', time.gmtime(min(os.path.getmtime(f'{D}/chain_{k}.npz') for k in ks)))} to {time.strftime('%d %H:%M', time.gmtime(max(os.path.getmtime(f'{D}/chain_{k}.npz') for k in ks)))} UTC")
for k in sorted(rows):
    kind, o = k.split("_")[0], k.split("_")[-1]
    if kind == "rts":
        print(f"rts {o}: new == v6r {np.array_equal(rows[k]['all_loso'], rows['v6r_' + o]['all_loso'])}, seen == rt {np.array_equal(rows[k]['seen_session'], rows['rt_' + o]['seen_session'])}")
    if kind == "stk":
        rt = rows.get("rt_" + o)
        print(f"stk {o}: new == v6r {np.array_equal(rows[k]['all_loso'], rows['v6r_' + o]['all_loso'])}, seen == rt {np.array_equal(rows[k]['seen_session'], rt['seen_session']) if rt is not None else 'no rt file'}, "
              f"seen == chain_final {np.array_equal(rows[k]['seen_session'], CH['seen_session'])}")
    if kind == "blend":
        m = 0.5 * (rows["v51b_" + o]["all_loso"] + rows["v6r_" + o]["all_loso"]); print(f"blend {o}: new == mean(v51b, v6r) {np.allclose(rows[k]['all_loso'], m)}, seen == chain_final {np.array_equal(rows[k]['seen_session'], CH['seen_session'])}")
    if kind == "rt":
        print(f"rt {o}: new == v51b {np.array_equal(rows[k]['all_loso'], rows['v51b_' + o]['all_loso'])}")
