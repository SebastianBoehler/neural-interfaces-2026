# §281 held-out FBLight adapter: seed-0 evidence

The adapter uses the archived P04_s23 and P10_s23 population checkpoints under `runs/diag/pdeep/nets`, not the shipped public-data trace. It permits only within0 with B2_s0 or B2_pool, each of which uses seed 0 and has its own verified prefix/route artifact. It rejects other combinations. For each requested session it checks the current seen-session fold masks, pinned checkpoint and QF archive hashes, delivery order, recognition trace and prefix provenance. It builds causal inputs from that fold's saved alignment, exports a StreamBN trace and verifies both eager-to-QF and exported-trace-to-eager frozen predictions within 1e-5 before running the package FineTune implementation. The consumer resets per session and stops at first recognition. Returned arrays contain only the optional FBLight replacement probabilities; NaN means no replacement. The CLI writes no predictions and computes no treatment score.

Guarded frozen-output comparison, all four held-out sessions, zero changed labels:

| Fold/session | Windows | Max absolute difference from QF |
| --- | ---: | ---: |
| P04/2 | 189 | 2.14449827e-6 |
| P04/3 | 188 | 1.83790061e-6 |
| P10/2 | 186 | 1.25842507e-6 |
| P10/3 | 199 | 1.45619291e-6 |

The P10/2 non-scoring smoke passed: the exported trace and eager model matched exactly, and package fine-tuning refitted at 20, 30, 40 and 50 before recognition at 60. Its largest parameter change was 0.0197390169. No prediction artifact or accuracy was produced. Three narrow adapter guard tests passed.

```sh
.venv/bin/python -m unittest discover -s studies/14nov/newmember -p 'test_s281_seen_fbl_dev.py'
scripts/run_capped.sh runs/s281_seen_fbl_checkpoint_guarded.log .venv/bin/python -I studies/14nov/newmember/s281_seen_fbl_checkpoint.py
scripts/run_capped.sh runs/s281_seen_fbl_dev_smoke.log .venv/bin/python -I studies/14nov/newmember/s281_seen_fbl_dev.py smoke
```

The P04 checkpoint SHA-256 is `c45b30f8650204346c23ac8e73a4a9534ce00d09157646d0b91f716ca2cd5ad2`; P10 is `1d5460ed0977199251026696912758491c3b5207993244a44539da270c20d47e`; the within0 QF archive is `220649ed1f6b86836d09860965a5f0115d5faeee4f8e0e3784255ddcaf8dfb96`. Source SHA-256: `s281_seen_fbl_dev.py` `59d5e8d2c39cff4f853f64ae255eb54e8ce91b8513e97f784103c511dce555e6`; `s281_seen_fbl_checkpoint.py` `87cfe1ae77a15e36a301ec6cb054da4b0a189bbc940f699645594434ef1a66dc`; `test_s281_seen_fbl_dev.py` `04798ad85270267bda8f1fb05e3485e0ed2d8d638718f9939b2fb59551d60ab8`.

The archived checkpoint format contains weights and alignment but does not record the original training-index hash. The pdeep source describes the seen-session folds and its logs report both folds, but the frozen-output match cannot independently prove which rows trained those files. The other five bases need matching per-base prefixes and/or seed-specific checkpoint validation; the other seven orders need their own prefixes and frozen-order checks. This adapter alone does not validate the other network families, GPU, t=250 on real data, the routed tail, or the full §281 system.
