# Fast package code (cand_rec_ts2_tsem_route3geb_fast)

`submission.py` (`ccbce506…`, 904 lines) is the runtime code of the staged fast package. It uses the same 32 data
files as the shipped route3geb package (submission 1040847) and is the base for every future upload.

- Outputs: bitwise equal to the shipped package at every stage on all 2,655 public windows (A100, worker image and venv).
- Speed: A100 solver time 0.41x, whole run 0.53x (ABBA job 3086004).
- ZIP: `runs/upload/sealed_cand_rec_ts2_tsem_route3geb_fast.zip`, SHA-256 in `fast_zip.sha256`. Not uploaded.
- Details and audit recipe: docs/AUDIT_SBN.md; ledger: docs/PREDICTIONS_11NOV.md §1.
