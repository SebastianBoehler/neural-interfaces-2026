# Lane N · NI-V on every finalist (registered Fri 20 Nov 2026 ~10:20 CET, before any new arm below was scored)

Instrument: NI-V v1 (`evidence/wave_19nov/NIV.md`), release `/mnt/lustre/work/gehler/gfh527/ni26_niv_v1_20261119` (Galvani), exact packages
through the real `Solver`, Scherer 9 people x 2 visits, 8 fresh full-session orders 71000-71007 (headline) plus within-run (secondary),
arms B-new and B-ret (+ zero-routing `fa`, run by `score_candidate.sh`). RTX 2080 Ti. Limits as in NIV.md: 22 of 41 channels
interpolated, stock BA about 47 %, null (cap250 seed 1) about -0.08; sign/veto evidence only, magnitude not trusted.

## Tags (all CBraMod members Scherer-free)
- Already scored, all 8 orders, both arms, reused unchanged (hashes in `inputs.sha256`): `cap250` (v66_cp1_cap250.zip), `hard` (F-C25 hard),
  `hardcbB` (= final stack with the Scherer-free all-public CBraMod B fits, `final_stack_hardcb_B.zip`, sha 47eacf8d...).
  The arm-C `final_stack_hardcb.zip` is NOT scored on NI-V (its CBraMod fits saw Scherer windows: leak).
- New, built here with `laneT/build_ttt.py FORM OUT runs/sealed/final_zips/final_stack_hardcb_B.zip` (only submission.py differs from hardcbB;
  the lane B1 T zips were built on the leaky arm-C base, so they are rebuilt on the B base): `tttlite5`, `tttsparse20`, `tttlight`, `tttheavy`
  (zips `hardcbB_ttt*.zip` on Galvani `zips/`).

## Predictions and rules (paired deltas, pp, people+orders bootstrap 80 %)
- vs hardcbB, B-new full-session: lite5 -0.05, sparse20 -0.10 [-0.5, +0.3] (as B1 card), light -0.10, heavy -0.10. B-ret: equal within +-0.3.
- vs cap250: hardcbB about 0 (seen -0.058 new, +0.107 ret), hard about -0.05.
- Reading: veto-only. A form is flagged if its B-new or B-ret full-session upper 80 % bound < 0 or its worst person change < -1.0 pp;
  nothing counts as support if the interval contains 0 (undecided). Per-person worst change reported for every arm.
- Zero-change sanity: tttlite5/sparse20 equal hardcbB exactly on any session where FT never runs (recognised people); on unrecognised B-new
  sessions they may differ. A tag whose B-new absolute equals hardcbB to all digits is a no-op to be investigated.

## Run
Waves, job ids and dependency chain are in the results section below.

### Queued 20 Nov ~10:30 CET (smoke 3140966 lite5, 3140967 sparse20 passed: 120 windows, refits 5, errors 0)
Per tag (B-new array 0-35 / enrol / ret 0-8 / fa 0-8 / collectors vs cap250 and vs hardcbB, collectors `afterok` on B-new, ret, fa):
lite5 3140980/81/82/83 -> 3140984/85; sparse20 3140986/87/88/89 -> 3140990/91; light 3140992/93/94/95 -> 3140996/97; heavy 3140998/99/3141000/01 -> 3141002/03.
References: hard vs hardcbB 3140965, cap250 vs hardcbB 3141004 (hard/hardcbB vs cap250 exist). Summary 3141005 (afterany on all collectors, lists missing
reports) writes `out/laneN_summary.json` / `.md` in the release dir (code `laneN/summarize.py`, `laneN/queue.sh`).
Note: if a collector with `afterok` does not run because an array task failed, rerun the failed task and `sbatch collect.sbatch TAG REF`, then `sbatch summary.sbatch`.
