# Sealed candidates (26 Oct 2026)

Staged ZIPs in `runs/upload/` (not in Git) and the evidence that each passed
the official ingestion + scoring replay. Hashes: `zip_sha256.txt`.

| ZIP | Built from | Evidence |
|---|---|---|
| `sealed_candidate_ensemble_all_r6nfg.zip` | staged r6nf weights + `submission.py` @ `2008781` (nets on the harness device) | `ensemble_all_r6nfg_worker_3003405_replay.json`, `gpu_cpu_agreement_3003404.jsonl` |
| `sealed_candidate_ensemble_all_r8t4g.zip` | r8t4 weights + `submission.py` @ `2008781` | `ensemble_all_r8t4g_worker_3003405_replay.json`, `gpu_cpu_agreement_3003404.jsonl` |
| `sealed_candidate_ensemble_all_r6nf_noscherer.zip` | staged r6nf (CPU) | `ensemble_all_r6nf_worker_3002631_replay.json` |
| `sealed_candidate_ensemble_all_r8t4.zip` | `scripts/sealed_r8t4.json` @ `c625ecc` code (Galvani 3002636) | `ensemble_all_r8t4_check.json`, `..._replay_a100_replay.json`, `..._worker_3002637_replay.json` |

`*_worker_*` replays ran inside the Apptainer replica of the official worker
image (`scripts/cloud/worker_image.def`: torch 2.14.1+cu130, braindecode
1.8.1) on an A100 with CUDA visible. Scores there are on the warm-up test
windows, which the sealed packages were trained on (≈ 1.0 by design); they
check execution, timing and file integrity, not accuracy. Selection and
evaluation numbers: `docs/archive/OPTIMIZATION_R8.md`, `docs/archive/SEALED_PLAN.md`.
`ensemble_all_r6nf_clean_1631413_check.json`: audit-grade rebuild of r6nf.
