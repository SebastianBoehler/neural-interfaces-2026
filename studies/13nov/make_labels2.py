"""Pseudo-label files for TTT12 (ledger s142), format of studies/12nov/chain_final.npz. ttt5 uses argmax as the label
and the row's max as its weight, so a weight of 0.01 switches a window off.
  chain_indep.npz    the dev chain (A.system, B2_pool) on members that exclude the three SessionBN families being
                     fine-tuned (core1, Hilbert, transformer, shallow nets, teacher): errors independent of the tuned nets
  chain_agree.npz    v3 chain labels (chain_final), weight kept where the independent chain agrees, 0.01 elsewhere
  chain_confhalf.npz v3 labels on the confident half only (weight 0.01 on the low-confidence half), label-free (s145)
  chain_truelow.npz  ORACLE diagnostic: true labels on the windows where v3's confidence is below its median, v3 labels
                     elsewhere (does fixing the hard windows carry the ceiling of s137?)
Usage: python studies/13nov/make_labels2.py"""
import os, sys
import numpy as np
REPO = os.path.abspath(__file__ + "/../../..")
sys.path.insert(0, REPO + "/replay")
import env  # noqa: E402
A = env.A; SC = A.SC; Z = A.L.Z; D = REPO + "/runs/labels"  # generated arrays live under runs/, not in tracked folders
CH = np.load(REPO + "/studies/12nov/chain_final.npz"); assert (CH["idx_seen"] == Z["idx|seen_session"]).all()
INDEP = [f for f in A.FAMS if not f.startswith("sbn")]; assert len(INDEP) == len(A.FAMS) - 3, INDEP
ind = A.system("B2_pool", INDEP)["out"]; out = {"indep": {}, "agree": {}, "truelow": {}, "confhalf": {}}
for p in A.PROTOS:
    v3, y = CH[p], Z[f"y|{p}"].astype(int); ag = v3.argmax(1) == ind[p].argmax(1); lo = v3.max(1) < np.median(v3.max(1))
    out["indep"][p] = ind[p]; out["agree"][p] = v3 * np.where(ag, 1.0, 0.01)[:, None]; out["truelow"][p] = np.where(lo[:, None], np.eye(3)[y], v3)
    out["confhalf"][p] = v3 * np.where(lo, 0.01, 1.0)[:, None]  # s145: the low-confidence half switched off (its labels are the wrong ones)
    print(f"{p}: v3 {SC.person(v3, p).mean():.2f} | independent chain ({'+'.join(INDEP)}) {SC.person(ind[p], p).mean():.2f} | agree on {ag.mean():.1%} "
          f"(label accuracy there {np.mean(v3.argmax(1)[ag] == y[ag]):.1%}, elsewhere {np.mean(v3.argmax(1)[~ag] == y[~ag]):.1%}) | low-confidence half: v3 accuracy {np.mean(v3.argmax(1)[lo] == y[lo]):.1%}", flush=True)
for k, v in out.items():
    np.savez(f"{D}/chain_{k}.npz", all_loso=v["all_loso"], seen_session=v["seen_session"], idx_seen=CH["idx_seen"]); print("written", f"chain_{k}.npz")
