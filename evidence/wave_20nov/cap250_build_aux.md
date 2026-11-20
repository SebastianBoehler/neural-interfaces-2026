# cap250 build: eye, rest, cbramod families

20 Nov 2026. Code: `cleanslate/cap250/families/{eye,rest,cbramod}.py`, helpers `_aux_v6.py`, `_aux_cb.py`, `_aux_compare.py`, launcher `_aux_slurm.sh`
(git dd6d50de). All jobs on Leonardo (AIFAC_P02_1636, boost_usr_prod, A100-64GB) through `worker.py`, release dir
`/leonardo_work/AIFAC_P02_1636/ni26_aux_20261120`. Smoke 64930471 (nine-person fold, `CAP250_SMOKE=1`) passed; first smoke 64930078 failed on a
missing `import time` (eye) and on `worker.py` needing `CAP250_REV` (no `.git` on nodes); both fixed.

Jobs: all-ten 64930473, nine-person fold (P02 held new) 64930475, returning fold (all ten, P04 first session only) 64930476.

## (1) All-ten reproduction of the archived artifacts
- cbramod s0-2: stage A reused (archived `external_C_{0,1,2}.pt`, copied from the 1675 C2 bank), 25-epoch stage B, `export_blob` half. Exported net,
  head and prior tensors **bitwise equal** to `export_blob(reve_full_C_{s}_all.pt, half=True)` of the archived fits (max |d| = 0).
- eye s0-2: not bitwise (SHA differs; same A100 type, GPU nondeterminism). Posteriors on all 2,655 public windows vs archived traces: max |dp|
  .0044/.0073/.0041, mean |dp| ~8e-5, argmax agreement 1.0. Raw weight max |dw| 2.6-4.1 (not a similarity measure for BN nets; posterior is the test).
- rest s0-2: not bitwise (weights max |dw| .07-.13). `rest_sN.npz` `align,k,G,scale,shrink` equal the archived npz exactly (diff 0) and agree across seeds.
  In-sample accuracy .9906/.9917/.9902 vs archived .9917/.9902/.9898 (same scale; the archive's seed ordering is not reproduced run to run).
  Package-form input stream and trace-vs-dev checks passed (input diff 9.5e-7, trace diff 0).
- Not run: end-to-end decision equality against sealed ZIP 9c4d... with these parts (assembler is the coordinator's).

## (2) Fold runs (SPEC format `train --people ... [--first-session-only]`, worker receipts with row digest)
Nine-person fold (2,466 rows) and returning fold (P04 first session only, 2,278 rows) ran for all family x seed (18/18 passed).
Rest rows come only from Graz source rows (pre-cue file indexed by source row index). CBraMod: `laneC.data.public` replaced by a source-scoped
function (same prior, rng 20261119, two aligned views); Lane C `stage_b` itself unchanged.

## (3) GPU time (seconds per seed, A100; includes startup/export; one fold = sum over seeds)
| fold | eye | rest | cbramod |
| --- | --- | --- | --- |
| all-ten | 113/114/114 | 150/150/150 | 208/325/318 |
| nine-person | 114/114/116 | 75/82/138 | 294/302/182 |
| returning P04 | 117/81/113 | 138/138/138 | 171/286/279 |

About 0.10 GPU-h (eye) + 0.12 (rest) + 0.08-0.23 (cbramod) per fold, ~0.3-0.45 h total; each array task asks for a whole node-share (8 CPU).
Queue wait dominated (about 1 h). Stage A (external, 960 steps) is not included.

## Notes
- Needs `CAP250_CB_CHECKPOINT` (pinned CBraMod safetensors) and stage A at `$CAP250_ASSETS/runs/cb_stage_a` (or `CAP250_CB_STAGEA`).
- Stage A is eligible for any PROTEUS fold (external data only); it is not refit per fold.
