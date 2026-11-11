# Routing v4 package code (cand_v4_bio) — checkpoint, not uploaded

`submission.py` (`cd593655…`, 975 lines) = the fast package's code plus an optional label-free identity gate
(class `Bio`, the same code as `experiments/bio_gate.py`). Without `bio_*` keys in `weights.npz` it is byte-identical
to the fast package at every stage on all 2,655 public windows (`scripts/check_compat.py --post`).

- Package: `runs/sealed/cand_v4_bio/` (33 files; per-file SHA-256 in `v4_files.sha256`).
- Staged ZIP: `runs/upload/sealed_cand_v4_bio.zip`, SHA-256 `94185c2fd5f2e4de0d8107809fe51653213c15073156c54017a8149f824a9722`,
  57,739,427 B; byte-identical to the A100 replay (3087470) and worker-image replay (3087471) ZIPs.
- Rebuild: `python scripts/add_bio_gate.py runs/sealed/cand_rec_ts2_tsem_route3geb_fast runs/sealed/cand_v4_bio
  runs/bio1/model_psd_d64.npz tau=0.25 ramp=0.05 t0=10`. The model comes from `BIO_EXPORT=… python
  experiments/bio_id.py` on the six external multi-session corpora (224 people, 673 sessions; PSD view, PCA 64).
- Checks: `scripts/check_bio_gate.py` (gate decisions equal the stand-alone class; outputs equal the base package before
  the first routed window of every session); replays identical on A100 and in the worker image, files unchanged;
  paired timing 1.02x the fast package (ABBA 3087472).
- Evidence and limits: docs/PREDICTIONS_11NOV.md §11 and §14.
