# §292D: learned SPD correction

Status at 16 Nov 2026: all30 production tasks completed0:0; the complete matrix passed provenance and mechanism guards and was scored. These are whole-alternative results, not package gains.

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

## Completed result

The collector in `studies/16nov/lane4/score.py` validates all30 files, their remote SHA-256s, immutable source/input manifest, fit-person exclusions, order hashes, finite normalized probabilities, identity before updates and active bounded bias updates. It calls the existing official metric and §281 people/order bootstrap through the Lane2 wrapper. The full result is [track4_results.json](track4_results.json).

Official equal-cell BA, schemes equally weighted: frozen42.14%, diagonal42.73%, full43.13%; the validated incumbent unrecognized-person reference is76.22%. Every seed is argmaxed/scored separately; metrics are averaged across three seeds before uncertainty resampling. These are different aggregate choices from the older §288 base screens.

Person/order-nested contrasts in percentage points, 80% intervals: diagonal−frozen +.79 [.49,1.09], 9/10 people improve; full−frozen +1.25 [.41,2.10], 6/10; full−diagonal +.46 [−.19,1.09], 5/10. P04/P10 new-person differences: diagonal+.57/−.37, full+1.74/−1.02. They are not returning-session tests. Seed uncertainty is not resampled.

The update improves this weak geometric decoder, but the complete alternative remains far below the incumbent. This is mechanism evidence, not a current-system or sealed gain. Transferring a controlled correction to a stronger representation is a proposed follow-up, not an implemented result or a queued run.

Review limits: stream timings exclude encoder/features, so they do not measure complete decoder speed. The current runner guards NPZ collisions but not a pre-existing orphan JSON sidecar; all initial production paths were new and no overwrite occurred. Future reuse should require both output paths absent and exclusive creation. The reset guard repeats a session; state is function-local, but A→B→A interleaving would be an additional guard before packaging. No queued source snapshot was edited after pinning.
