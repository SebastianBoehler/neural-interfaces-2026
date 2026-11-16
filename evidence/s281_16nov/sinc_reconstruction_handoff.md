# SincShallow checkpoint reconstruction diagnostic — 16 Nov 2026

The QF archive contains held-out predictions, but the inspected `runs/diag/pdeep/nets` directory did not contain the corresponding SincShallow fold checkpoints. Other named SincShallow checkpoints found under `runs/pnets` are personalized models and are not established substitutes. This is not an exhaustive proof that no checkpoint exists elsewhere.

The local Claude transcript `~/.claude/projects/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/e43117e0-fad0-4d4b-b456-63b70df67d66.jsonl` records the 13 Nov launch as `FORMS=gC5,s15 SCHEME=within PSEED=0 CHAIN_NPZ=studies/13nov/chain_rts_within0.npz python -I studies/13nov/ttt18.py`. Its source revision `4370f1fc772d1a3377d3758d599b0b364b3231eb` is a candidate: the launch record does not establish every source file on the original machine. The candidate recipe is `sincshallow_ea_ptm7r_rot05_sbn`, seed 0, 40 epochs, learning rate and weight decay 0.001, batch 64, session groups 32, rotation 0.05, Euclidean alignment and SessionBN k=20. For each held person P04 or P10, `seen_session` trains on all other people plus that person's session 1; sessions 2 and 3 are held out.

Galvani array **3110506**, tasks 0–1 (%2), runs the pre-registered fold-specific parity diagnostic in `/mnt/lustre/work/gehler/gfh527/ni26_s281_sinc_qf_smoke_20261116`. It uses account `gehler`, one GPU, eight CPUs and 64 GB per task; the driver and OpenMP/BLAS thread settings match the CPU request. Submission occurred 16 Nov at 14:29 scheduler time. Both tasks were pending at the last check. A passing task must reproduce every held session's archived probabilities within 1e-5, with no changed labels, and saves a checkpoint only after all guards pass. No accuracy or treatment score is computed.

The immutable manifests were checked before submission and are rechecked by each task:

- Candidate source manifest SHA-256: `6f54815163d66a3432d5879baad53949131460ed5062d2ee927a3781f40d27f7`.
- Input manifest SHA-256: `fcedec3b3b1f984ac3cfed6884b02b39792cfe5e334e49ca673cb8096853cd2f`.
- QF archive SHA-256: `4a4fe30cfb2726ee2e87cf8adeabbb2ffd8c535d65d8ede2c8fbec7468c6e20b`.
- Seed-0 pretrained trunk SHA-256: `f28f2865cf666bc85d872aacea2f2fb147dfc3d3835d6bf3ff2df55aa949ab8f`.
- Batch SHA-256: `21a1aa4d697f1ac0ae0ec8ceeadffb767d354b24cc8b1938b79027ece416bbeb`.

Three non-training guard tests passed locally and with the Galvani interpreter: split leakage, parity/label failures, and nonfinite/shape rejection. The candidate reconstruction runs on Galvani GPU hardware, so device and library differences may cause parity failure. A pass establishes output parity only for the tested seed, folds and delivered order; it does not prove the original source revision, original training rows or parameter identity. Inspect both task exits and per-session results before using either checkpoint.
