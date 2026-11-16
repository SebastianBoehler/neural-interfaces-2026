# §281 seen-session smoke, Galvani (16 Nov 2026)

- Workspace: `/mnt/lustre/work/gehler/gfh527/ni26_s281_seen_smoke_20261116`.
  Slurm array `3110199_0` (fb_coral) and `_1` (fb_am) both COMPLETED, exit 0:0.
  full0, seed 0, full training recipe, two folds per family; not a one-epoch smoke.
- Source: production recipe revision `cc6ed2a2aecb0d017cbcf094fd040f3f1e3c394a`.
  All eight training-file MD5s printed by each task equal that revision; each task also
  reports 10/10 code/wrapper SHA-256 checks, the pinned permutation check and `EXIT 0`.
  The separately added `cell_seen.sh` SHA-256 is `7a53f5c2946572db458eca6bee99315eecc77a29049b201fef708305da415c9d`.
- Input hashes: windows `2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327`;
  permutation `6046bf5b2231d29d28bbfcf88ea7665d4aaa506c5787bd836ebec6a5b945430f`;
  `chain_final.npz` `75101510ad18245a5d34b47481eab1bd0d38f72f0710011a3a6ecee14d2ce22a`.
  The complete input manifest and source checks are in the private smoke workspace.
- Each log has exactly one P04_s23 and one P10_s23 fold and a successful family guard per fold.
  CORAL penalty was active in both; Fourier mixing was active in both.
- Reconstructed from the pinned `seen_session` protocol and the hash-matched data:
  P04 sessions 2–3 have 377 test windows, P10 sessions 2–3 have 385; each fold excludes
  its own held-out windows while keeping that person's session 1 for training. Their union
  equals `idx|seen_session` in replay `bases.npz`. This verifies the source masks and data;
  the training logs do not separately record every training index.
- Both outputs have `(762, 3)` finite probabilities in [0,1], row-sum error at most
  `2.22e-16`, and exactly the 762 replay indices in order.
  - fb_coral SHA-256 `a3c763fe0972b733a383022471af5c72c6427ea9c71b34e903a518d0250c70ca`.
  - fb_am SHA-256 `e8ff50ff1ba510282c4bd52b7e47ba0f667e933915d325b406d94c1464d16cbc`.
- Logs print large base-vs-harness differences. They compare shuffled delivery output to
  recording-order harness output, so those values are not a same-order parity test.
  This smoke does not validate the returning routed chain. The training array remains
  subject to the registered smoke and resource checks.
