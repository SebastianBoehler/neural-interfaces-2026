# §292D: learned SPD correction

Status at 16 Nov 2026, 18:04 CET: real-data smoke completed; production array queued on Galvani. No treatment score is claimed.

- Registration: `docs/LEDGER.md` §292D, committed `0034951` before production submission.
- Workspace: `/mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane4`.
- Source: isolated copy of §288's original TSMNet, experiments and scripts, with the new adaptation/runner files under `studies/16nov/lane4`. The public data and environment are read-only inputs.
- Input/source manifest: 69 SHA-256 entries; manifest SHA-256 `454f053ae902106f149820bcc6e724b1c691acd52a1c6aad80a12c6dca8c48dd`. Saved in `lane4/inputs.sha256`.
- Smoke `3112522`: completed 0:0 in 1m18s Slurm elapsed; Python training and guards took 47.32s. A nine-person fit held out P01. SPD-BN was active for 1,600 training steps. Identity extraction, zero-bias equivalence, future-prefix independence, fresh-session reset and active diagonal/full updates passed on real inputs.
- Production `3112555`: tasks `0–29%6`, ten held people × three seeds. Every fold trains once, then emits frozen/diagonal/full predictions for eight pinned delivery orders. Initial limit is six GPUs, 4 CPUs and 16 GB per task, 2-hour cap. The cap is not an expected duration.
- Expected output: `runs/lane4/fold_s<seed>_<held>.npz` plus JSON diagnostics. Arrays are indexed by order/mode; each includes original global row indices. The output guard rejects collisions.

The adaptation occurs after the frozen BiMap, in 20-dimensional SPD space. This is independently implemented and OSPDIM-inspired, not a reproduction of its pre-BiMap algorithm. Its earlier-window-only FIFO and fresh per-session optimizer are explicit. A full bias reached the registered norm cap in smoke; this is an active update, not evidence of accuracy improvement.

There were two staging errors before any job: rsync lacked an intermediate directory, and a shell-wrapped sbatch command had incorrect quoting. Neither submitted or ran a model. Both were corrected by creating the directory and using a standalone batch file. The first submitted smoke passed without a model-code repair.

The smoke did not score a treatment. Its sub-millisecond adapter-loop timings exclude the full model and feature-extraction cost. Collect all folds, check source/order/output provenance, then use the existing official cell metric and person/order uncertainty. Whole-model performance and adaptive-minus-frozen effects are separate results.
