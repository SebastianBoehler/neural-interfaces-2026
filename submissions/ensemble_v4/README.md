# Routing v4.1 package code (cand_v41_bio) — checkpoint, not uploaded

`submission.py` (`45f9f2fd…`, 976 lines) = the fast package's code plus an optional label-free identity gate
(class `Bio`, the same code as `experiments/bio_gate.py`). Without `bio_*` keys in `weights.npz` it is byte-identical
to the fast package at every stage on all 2,655 public windows (`scripts/check_compat.py --post`).

- Package: `runs/sealed/cand_v41_bio/` (33 files; per-file SHA-256 in `v4_files.sha256`); gate threshold 0.26.
- Staged ZIP: `runs/upload/sealed_cand_v41_bio.zip`, SHA-256 `53fc16797f541a81…` (full value in docs/SEALED_UPLOADS.md),
  57,739,465 B; byte-identical to the A100 replay (3088399) and worker-image replay (3088400) ZIPs.
- Rebuild: `python scripts/add_bio_gate.py runs/sealed/cand_rec_ts2_tsem_route3geb_fast runs/sealed/cand_v41_bio
  runs/bio1/model_psd_d64.npz tau=0.26 ramp=0.05 t0=10`. The model comes from `BIO_EXPORT=… python
  experiments/bio_id.py` on the six external multi-session corpora (224 people, 673 sessions; PSD view, PCA 64).
- Checks: `scripts/check_bio_gate.py`; replays identical on A100 and in the worker image, files unchanged; paired
  timing 1.01x the fast package (ABBA 3088401); independent review without blocking findings (runs/bio1/review_v4.md).
- Supersedes the first build (threshold 0.25, code `cd593655…`, ZIP `94185c2f…`, deleted).
- Evidence and limits: docs/PREDICTIONS_11NOV.md §11 and §14.
