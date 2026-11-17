# §326 conditional-logit source pilot

Registered before source-query scores in LEDGER §326. Implementation commit `4911d47`; source manifest SHA-256 `f35b41103b37098de243e0a59312b1f54dc76bc4a2959d22f7dba1aeb55b811d` in `/mnt/lustre/work/gehler/gfh527/ni26_s326_conditional_head_20261117`. The seven-person CPU preparation is job `3117223`; the no-score-guarded A100 fit and subsequent source-query scorer are `3117224`, dependent afterok on preparation. At submission the account had 82 active jobs and 10 running. Preparation requests six CPUs/48 GiB/20 minutes, pilot four CPUs/48 GiB/one A100/10 minutes. Actual runtime is not yet known.

Local `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q studies/17nov/conditional_head/test_loss.py` passed both tests. The private cluster snapshot's isolated Python imports and same CPU loss tests passed. Real-data source scope, stream/reset, zero/native-objective, gauge and finite-difference checks execute before source-query scores. The pilot must fail closed if any guard fails. Existing §312 and §323 full-chain jobs are independent.

This is a source-only pilot, not an outer or current-system score. P01 is excluded; P02/P03 are both query-only; P04–P10 fit the prior, scaler, conditional matrix and head. No package or upload is authorized by this result.
