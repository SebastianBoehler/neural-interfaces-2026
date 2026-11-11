# Optimization round 1 evidence (18 October 2026)

- `cv_summary.json`: every candidate × protocol, latest run per seed, with
  per-fold cell means. Deep runs: Galvani A100 jobs 2975104 and 2975118.
  Linear runs: local M4 Max CPU. Built by `scripts/summarize_results.py`.
- `submissions.json`: exported packages, harness agreement, official
  inference-only Benchopt replay metrics and checksums.

Recorded `rev` values are the parent commit (`daedce3`); the experiment code
was uncommitted at run time and is first committed together with this
evidence. Windows: `data/proteus_windows.npz`, SHA-256 in `submissions.json`.
Limits: native replays only, no Docker/Codabench run, public data only.
See `docs/archive/OPTIMIZATION.md` for interpretation.
