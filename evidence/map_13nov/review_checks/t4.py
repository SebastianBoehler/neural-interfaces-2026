import numpy as np, glob, os, collections
w = np.load("data/proteus_windows.npz", allow_pickle=True)
per = np.array([s[-3:] for s in w["subject"]]); ses = w["session"].astype(str); split = w["split"].astype(str); itf = w["interface"].astype(str); run = w["run"]
print("split values", collections.Counter(split))
Z = dict(np.load("runs/replay_data/inputs.npz", allow_pickle=True))
for f in sorted(glob.glob("runs/replay_data/feats_exact/*.npz")):
    z = np.load(f, allow_pickle=True); itr = z["idx_tr"].astype(int); name = os.path.basename(f)[:-4]
    te = np.concatenate([z[f"{k}|idx"].astype(int) for k in range(len(z["sessions"]))])
    trs = sorted(set(zip(per[itr], ses[itr]))); tes = sorted(set(zip(per[te], ses[te])))
    tp = set(per[te])
    leak = [x for x in trs if x[0] in tp]
    print(name, "n_tr", len(itr), "n_te", len(te), "test", tes, "| train sessions of the test person:", leak, "| overlap rows", len(set(itr) & set(te)), "| n train sessions", len(trs),
          "| te order sorted", all((np.diff(z[f"{k}|idx"]) > 0).all() for k in range(len(z["sessions"]))))
    proto = name.split("-")[0]; pos = {int(g): i for i, g in enumerate(Z[f"idx|{proto}"])}
    d = max(np.abs(Z[f"pc128|{proto}"][[pos[int(g)] for g in z[f"{k}|idx"]]] - z[f"{k}|pc"][:, :128]).max() for k in range(len(z["sessions"])))
    print("    pc128 in inputs vs fold test pc: max|diff|", d)
# onset order within session = row order?
on = w["onset"].astype(float)
for s in sorted(set(zip(per, ses))):
    m = np.flatnonzero((per == s[0]) & (ses == s[1]))
    print(s, "rows contiguous", (np.diff(m) == 1).all(), "| interfaces in order", list(dict.fromkeys(itf[m])), "| runs", list(dict.fromkeys(zip(itf[m], run[m])))[:8], "| onset monotone within run blocks", bool(all((np.diff(on[m][(itf[m]==i)&(run[m]==r)])>0).all() for i,r in set(zip(itf[m],run[m])))))
# core1 seen vs loso on the same rows
B = dict(np.load("runs/replay_data/bases.npz", allow_pickle=True))
i_s = B["idx|seen_session"].astype(int)
for k in ("core1", "mix", "deep"):
    print(k, "seen vs loso same rows: max|diff|", np.abs(B[f"B2_pool|{k}|seen_session"] - B[f"B2_pool|{k}|all_loso"][i_s]).max(), "mean|diff|", np.abs(B[f"B2_pool|{k}|seen_session"] - B[f"B2_pool|{k}|all_loso"][i_s]).mean())
for i in range(11):
    print("member", i, "seen vs loso mean|diff|", round(float(np.abs(Z["B2_pool|Ps|seen_session"][i] - Z["B2_pool|Ps|all_loso"][i][i_s]).mean()), 4))
print("GE seen vs loso same rows", np.abs(Z["GE|seen_session"] - Z["GE|all_loso"][i_s]).max())
print("core1 same across bases:", [float(np.abs(B[f"{b}|core1|all_loso"] - B["B2_pool|core1|all_loso"]).max()) for b in ("B2_s345", "B2_s3", "B2_s4", "B2_s5")])
BM = dict(np.load("runs/replay_data/bases_fresh_mix.npz")); print("fresh core1 == pool core1:", [float(np.abs(BM[f"{b}|core1|all_loso"] - B["B2_pool|core1|all_loso"]).max()) for b in ("B2_s012", "B2_s0", "B2_s1", "B2_s2")])
