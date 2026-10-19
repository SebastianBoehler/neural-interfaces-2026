# Optimization round 5 evidence (19 October 2026)

External training data: Dreyer 2023 (CC BY 4.0) and Scherer 2015 /
BNCI 2015-004 (CC BY-NC-ND 4.0, used by user decision; see
`docs/OPTIMIZATION_R5.md`).

- `cv_summary.json`: every round-5 harness run (`NAME@TAG`), mean/SD over
  seeds, per seed, `rev` and source. Deep runs: Galvani A100 jobs 2977845
  (Dreyer, seed 0), 2977868 (Scherer unfixed), 2977874 (`~n`, `~n~w0.5`),
  2977905 (`~w0.25`, `~w0.1`), 2977968 (`_ft`), 2979584 (`~n~w0.1`,
  `all_loso`/`seen_session`, seeds 0–4), 2979672 (`~n~w0.1` dev seeds
  3–4); all on the single-job `vs` QOS slot because the normal queue was
  full. v2 runs: M4 Max CPU (`local-v2`).
- `ensemble_selection.json`: `scripts/external_probs.py` outputs (round-3
  mix with members swapped, rebalanced α 0.5, k 5). Selection used
  `dev_loso`/`dev_session`; the chosen swap (`ens_w0.1`) was fixed before
  `ens_w0.1_eval` (`all_loso`, `seen_session`) was scored;
  `ens_w0.1_dev5` adds dev seeds 3–4.
- `diagnostics.json`: Scherer source high-pass, spectra, aligned
  eigenvalue spectra before/after `~n`, Scherer→PROTEUS transfer, v2
  statistics check, AlignedNet refactor check.
- `submissions.json`: `ensemble_all_r5` (round-3 package with the raw
  ShallowConvNet and FBCNet slots retrained with Scherer `~n~w0.1`),
  100% harness agreement, official ingestion/scoring replays on Mac CPU and
  a Linux A100-node CPU, staged ZIP checksum, timing.

Recorded `rev` values: cac7f1f (local) and 1459dcd (Galvani checkout, with
`experiments/` and `scripts/` rsynced); the round-5 code was uncommitted at
run time. Limits: native replays only, no Docker or Codabench run; the
staged package scores 1.0 on the warm-up because it trains on the test
participants. Nothing was uploaded.
