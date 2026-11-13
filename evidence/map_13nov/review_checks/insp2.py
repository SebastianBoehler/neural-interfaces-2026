import numpy as np
Z=dict(np.load("runs/replay_data/inputs.npz",allow_pickle=True))
B=dict(np.load("runs/replay_data/bases.npz",allow_pickle=True))
bases=["B2_pool","B2_s345","B2_s012","B2_s0","B2_s1","B2_s2"]
for p in ("all_loso","seen_session"):
    print(p, "ws", {b: Z[f"{b}|ws|{p}"].tolist() for b in bases[:2]})
    for b in bases[1:]:
        d=[float(np.abs(Z[f"{b}|Ps|{p}"][i]-Z[f"B2_pool|Ps|{p}"][i]).max()) for i in range(11)]
        print(p,b,"max|diff| vs pool per slot",np.round(d,4))
    # pool = mean(s012,s345)?
    m=0.5*(Z[f"B2_s012|Ps|{p}"]+Z[f"B2_s345|Ps|{p}"])
    print("pool - mean(s012,s345) per slot", np.round(np.abs(m-Z[f"B2_pool|Ps|{p}"]).max((1,2)),6))
    m=(Z[f"B2_s0|Ps|{p}"]+Z[f"B2_s1|Ps|{p}"]+Z[f"B2_s2|Ps|{p}"])/3
    print("s012 - mean(s0,s1,s2) per slot", np.round(np.abs(m-Z[f"B2_s012|Ps|{p}"]).max((1,2)),6))
    # index checks
    print("idx equal inputs/bases", (Z[f"idx|{p}"]==B[f"idx|{p}"]).all(), "sess eq", (Z[f"sess|{p}"]==B[f"sess|{p}"]).all(), "y eq",(Z[f"y|{p}"]==B[f"y|{p}"]).all(), "cell eq",(Z[f"cell|{p}"]==B[f"cell|{p}"]).all(), "person eq",(Z[f"person|{p}"]==B[f"person|{p}"]).all())
    idx=Z[f"idx|{p}"]; print("idx sorted", (np.diff(idx)>0).all(), idx[:5], idx[-5:])
    sess=Z[f"sess|{p}"]; per=Z[f"person|{p}"]; cell=Z[f"cell|{p}"]; y=Z[f"y|{p}"]
    for s in np.unique(sess):
        m=sess==s; r=np.flatnonzero(m)
        print("  sess",s,per[m][0],"n",m.sum(),"contig",(np.diff(r)==1).all(),"cells",[(int(c),int((m&(cell==c)).sum()),np.bincount(y[m&(cell==c)],minlength=3).tolist()) for c in np.unique(cell[m])], "stream", B[f"stream|{p}"][m][0], len(set(B[f"stream|{p}"][m])))
w=np.load("data/proteus_windows.npz",allow_pickle=True)
print(w.files)
print({k:w[k].shape for k in w.files})
