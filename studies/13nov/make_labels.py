"""Pseudo-label files for fine-tuning in the format of studies/12nov/chain_final.npz (ledger s135):
  chain_stack.npz  the FT+OCU stack's outputs on B2_pool (s130b design input), a better label-free chain than v3
  chain_true.npz   one-hot true labels: the ceiling of fine-tuning with perfect labels (oracle, never shippable)
  LABEL_OUT=name  only the stack file, under that name, from the FT source stacklib is given (FT_DIR/FT_PFX/FT_SEEDS):
                  round 3 of s137 uses round 2's fine-tuned outputs (ttt9stack) to make chain_stack2.npz
Usage: [LABEL_OUT=chain_stack2 FT_DIR=... FT_PFX=... OCU_SEEDS=...] python studies/13nov/make_labels.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stacklib import REPO, Z, run, SC  # noqa: E402
CH = np.load(REPO + "/studies/12nov/chain_final.npz"); D = REPO + "/runs/labels"  # generated arrays live under runs/, not in tracked folders
assert (CH["idx_seen"] == Z["idx|seen_session"]).all(), "chain_final's seen rows are not the replay's seen rows"
out = run(("FT", "OCU"), "B2_pool", outputs=True)
for p in ("all_loso", "seen_session"):
    print(f"{p}: v3 chain_final {SC.person(CH[p], p).mean():.2f} | stack FT+OCU {SC.person(out[p], p).mean():.2f}")
name = os.environ.get("LABEL_OUT", "chain_stack")
np.savez(D + f"/{name}.npz", all_loso=out["all_loso"], seen_session=out["seen_session"], idx_seen=CH["idx_seen"]); print("written", name)
if "LABEL_OUT" in os.environ:
    sys.exit(0)
np.savez(D + "/chain_true.npz", all_loso=np.eye(3)[Z["y|all_loso"].astype(int)], seen_session=np.eye(3)[Z["y|seen_session"].astype(int)], idx_seen=CH["idx_seen"])
print("written chain_stack.npz, chain_true.npz")
