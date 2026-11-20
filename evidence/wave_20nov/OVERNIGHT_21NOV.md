# Overnight queue · Fri 20 → Sat 21 Nov 2026 (submitted ~23:35–00:30 CET; read at ~08:00, decision ~10:00)

All lanes below pre-registered in `docs/LEDGER.md` ("Overnight 21 Nov", two entries). Nothing uploaded. Existing lanes untouched: R4b (Leonardo `r4b-run`, 1675), K2 (Galvani `ni26-laneK2-chain`), B1 sparse20 (`ret-sparse20`, `retcol-sparse20`).

## 1. Real per-fold package comparison, new orders (Leonardo, account 1636) → card `cap250_pipeline.md` + `final_stack.md`
Instrument: the 24 fold packages of run `production-20261120-32a94ab7` (12 folds × stock / hardcb), unchanged `stream_fold.py` + `score.py`; hardcb arm = `hardcb_safe` (hardlinked copies, only `submission.py` carries the `patch_dead.py` guard; prep job 64984988). Run dir `/leonardo_work/AIFAC_P02_1636/ni26/cap250/runs/overnight-21nov-pkgcmp`, scripts `.../cap250/overnight21/` (repo `scripts/overnight21/`).
- Smoke 64985012 (hardcb_safe P01 order 71000 vs stored production hardcb): 380 values, 0 changed → guard is zero-change.
- Wave A (orders 71004–71015, 3 chunks of 4): streams stock 64985043–64985163, hardcb_safe 64985053–64985248; scorers stock 64985052 (o71004) / 64985101 (o71008) / 64985167 (o71012); hardcb_safe 64985076 / 64985141 / 64985249.
- Wave B (orders 71016–71023, 2 chunks): streams 64985396–64985456; scorers stock 64985408 / 64985441, hardcb_safe 64985423 / 64985457.
- Due: scorers done ~02:00–04:00 CET (72+48 streams of ~3–10 min, GPU queue dependent).
- Third arm dsu_l5 NOT run: the fold pipeline has no lite5/dispersion fold fits (stated in the ledger entry).
- Collect (local): `scp` the scores into scratch, or on Leonardo (CPU job, not login): `sbatch --account=AIFAC_P02_1636 --partition=boost_usr_prod --gres=gpu:1 --time=10 --wrap="/leonardo_work/AIFAC_P02_1636/ni26/.venv312/bin/python /leonardo_work/AIFAC_P02_1636/ni26/cap250/overnight21/pkgcmp_collect.py /leonardo_work/AIFAC_P02_1636/ni26/cap250/runs/production-20261120-32a94ab7 /leonardo_work/AIFAC_P02_1636/ni26/cap250/runs/overnight-21nov-pkgcmp /leonardo_work/AIFAC_P02_1636/ni26/cap250/runs/overnight-21nov-pkgcmp/summary.md"` then `ssh cineca-leonardo cat .../overnight-21nov-pkgcmp/summary.md`. The collector pools 71000–71003 (production hardcb) with every finished chunk, lists missing chunks, and gives order and people-nested 80 % intervals; P04/P10 separate; run it on whatever is finished (it lists the rest as missing).
- Refutation (registered): 12-order mean hardcb_safe − stock ≤ 0 on new people refutes; (0, +0.2) undecided. Prediction +0.33. Collector self-check on 71000–71003 reproduces +0.19/+0.49/+0.30/+0.34.

## 2. NI-V on 8 further orders, cap250 vs hardcbB (Galvani) → card `laneN_niv_final.md`
Release copy `/mnt/lustre/work/gehler/gfh527/ni26_niv_x8_20261121` (SEEDS 71008–71015, setup job 3144801 asserted it; zips and enrolled packages are symlinks to the pinned originals).
- Smoke 3144803 (hardcbB, person 5, order 71008). B-new cap250 array 3144804 (0–35), hardcbB 3144806 (0–35); B-ret cap250 3144805 (0–8), hardcbB 3144807 (0–8); collector 3144808 (afterany all four) → `out/report_hardcbB_vs_cap250.json`.
- Due ~03:00–05:00 CET (2080 Ti queue). Collect: `ssh mlcloud-galvani 'cd /mnt/lustre/work/gehler/gfh527/ni26_niv_x8_20261121 && python3 -c "import json;d=json.load(open(\"out/report_hardcbB_vs_cap250.json\"));print(json.dumps(d,indent=1)[:3000])"'` (if collector ran before a task failed: rerun the failed task, then `sbatch collect.sbatch hardcbB cap250`).
- Veto rule (registered): 16-order pooled B-new full-session upper 80 % bound < 0 or worst person < −1.0 pp.

## Not queued (reason)
- R4b / K2 extra orders: scripts take no order argument (K2 `final_k2.py` is `full0|within0` only; R4b waves are fixed arrays); R4b already holds ~40 Leonardo tasks. Left to their cards (due 08:00).
- B8 (ds_all as extra fold variant) and B4 (seeds): need a fold DAG rebuild, not safe to start unattended; open in BACKLOG.
