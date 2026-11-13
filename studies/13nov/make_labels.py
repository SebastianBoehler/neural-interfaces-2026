"""Pseudo-label files for fine-tuning in the format of studies/12nov/chain_final.npz (ledger s135):
  chain_stack.npz  the FT+OCU stack's outputs on B2_pool (s130b design input), a better label-free chain than v3
  chain_true.npz   one-hot true labels: the ceiling of fine-tuning with perfect labels (oracle, never shippable)
Usage: python studies/13nov/make_labels.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stacklib import REPO, Z, run, SC  # noqa: E402
CH = np.load(REPO + "/studies/12nov/chain_final.npz"); D = REPO + "/studies/13nov"
assert (CH["idx_seen"] == Z["idx|seen_session"]).all(), "chain_final's seen rows are not the replay's seen rows"
out = run(("FT", "OCU"), "B2_pool", outputs=True)
for p in ("all_loso", "seen_session"):
    print(f"{p}: v3 chain_final {SC.person(CH[p], p).mean():.2f} | stack FT+OCU {SC.person(out[p], p).mean():.2f}")
np.savez(D + "/chain_stack.npz", all_loso=out["all_loso"], seen_session=out["seen_session"], idx_seen=CH["idx_seen"])
np.savez(D + "/chain_true.npz", all_loso=np.eye(3)[Z["y|all_loso"].astype(int)], seen_session=np.eye(3)[Z["y|seen_session"].astype(int)], idx_seen=CH["idx_seen"])
print("written chain_stack.npz, chain_true.npz")
