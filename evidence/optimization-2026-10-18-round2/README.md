# Optimization round 2 evidence (18 October 2026)

- `cv_summary.json`: round-2 runs, latest per seed (`summarize_results.py`).
  Deep and ensemble runs: Galvani A100 array jobs 2975257 (dev protocols)
  and 2975282 (`all_loso`, `official`), seeds 0–4. Linear runs (v1, v2,
  `v2_ks*`, `v2_dc`): M4 Max CPU; v1/v2 reproduce round 1 exactly.
- `combine_<protocol>.json`: probability-averaging ensembles scored from
  saved per-window probabilities (`scripts/combine_probs.py`). Selection
  used `dev_loso`/`dev_session` only; the chosen ensemble
  (`v2+both_x5`) was fixed before `all_loso` was scored. Other rows in
  the `all_loso`/`official` files are side output, not used for choice.
- `chosen_ensemble_eval.json`: per-participant `all_loso`/`official`
  scores of v1, v2 and the chosen ensemble, with participant/cell counts.
- `slow_diagnosis.json`: `scripts/diagnose_slow.py` output: block ablation
  (covariance/slow/both) on `all_loso`, per-cell tables for P04/P09/P10,
  per-session slow-feature statistics. Uses P04/P09 labels for diagnosis
  only. `both_vs_v2_agreement` = 1.0 checks the refit against harness v2.
- `submissions.json`: exported packages, harness agreement, official
  ingestion/scoring replays (Mac CPU, Linux CPU node; A100 node only for
  the pre-fix submission.py), checksums,
  the before/after check of the single-thread fix and timing measurements.

Recorded `rev` values: 887aa6d (local) and 1459dcd (Galvani checkout,
with `experiments/` rsynced); the round-2 code was uncommitted at run time
and is first committed together with this evidence.
Windows: `data/proteus_windows.npz`, SHA-256 in `submissions.json`.
Limits: public data only; native replays, no Docker or Codabench run.
See `docs/archive/OPTIMIZATION.md` (round 2) for interpretation.
