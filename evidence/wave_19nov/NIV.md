# NI-V v1 · validation instrument (built 19 Nov 2026, registered in docs/LEDGER.md "NI-V")

Code `niv/`; release `/mnt/lustre/work/gehler/gfh527/ni26_niv_v1_20261119` (Galvani, immutable inputs pinned in `inputs.sha256`).
**Status at hand-over (~15:00 CET): all jobs queued, no NI-V score exists yet; results land in the release dir by ~18:00 CET.**

## What it is
- **Exact-package arms** (every member re-streamed by construction: the real `Solver.predict`, one window per call, fresh copy
  per visit, RTX 2080 Ti, no STRICT exemption possible). Headline = **full-session shuffle**, 8 fresh orders (seeds 71000-71007,
  none run-contiguous); within-run shuffle (same seeds) is secondary. Labels are opened only by `niv/collect.py`.
- **B-new**: 9 Scherer 2015 people, both visits, ZIP unchanged (`score.sbatch`, 36 tasks per ZIP).
- **B-ret**: per Scherer person k the ZIP with visit 1 of k enrolled (`enrol.py`: gallery, personal head, own-space, TSEM prior as
  for a public person); visit 2 streamed (`enrol_run.sbatch ... ret`). **Zero-routing**: the 8 other people through package k
  (`... fa`), recognitions counted. Self-check vs the stored P01 rows: bio, own-space, TSEM prior bit-identical; contrast gallery cos
  0.997, head cos 0.996 (`check_enrol.py`; stored galleries came from the offline feature cache).
- **A**: 10 held PROTEUS people from the corrected re-streamed replays on file, **only the two legacy orders full0 (full) and within0
  (run-contiguous, secondary)**, v5.1b-equivalent incumbent vs cap250 (`cohort_a.py`, reproduces the stored official scores exactly).
- **Null**: cap250 with `v6_ft_seed` 0 -> 1 only (`make_null_zip.py`), same orders. MDD = 2.49 x bootstrap SE (alpha .05 one-sided, power .8).
- **Stats** (`stats.py`): person-equal cell mean, bootstrap resamples people AND orders (10k draws), 80 % and 95 % intervals,
  decisions changed, sealed-weighted combination 0.57 new (Scherer-new + A) / 0.43 returning (Scherer-ret).

## Score a new candidate ZIP (exact command)
```
ssh mlcloud-galvani   # eduVPN on
cd /mnt/lustre/work/gehler/gfh527/ni26_niv_v1_20261119
cp /path/to/candidate.zip zips/CAND.zip
bash score_candidate.sh CAND cap250        # reference tag cap250; add a third arg to pass a null tag
# -> B-new, enrolment, B-ret, zero-routing and a collector; report: out/report_CAND_vs_cap250.json
```
Reports already queued: `out/report_cap250_vs_v51b.json` (cap250 vs v5.1b with null `cap250_seed1`), `out/parity_result.txt`
(worker reproduces archived Friday decisions for G, order 22411).

## Known limits (state them with every number)
- Scherer: 22 of 41 channels interpolated, stock BA about 47 %; external instruments mispredicted cap250 magnitude 4x before.
  Treat B as sign and veto evidence; magnitude claims need A (2 orders only) and the sealed weights.
- A has 2 orders (not 8): each further order needs a ~15-job corrected chain (frozen banks, bridges, 4 refit families, terminal)
  hard-wired to full0/within0 and their legacy anchors. Not built today. P04/P10 returning is not recomputed (existing repo numbers only).
- No further external 3-task cohort has data and pipeline in the repo (Shin2017B/Shin2018/Ajra lack WORD; interpolated montage).
- Enrolled-package gallery is built on the live causal view, not the stored offline cache (cos 0.997 on P01).

## Zero-routing / false-recognition arm (`enrol_run.sbatch ... fa`): usage and 19 Nov fix
- Bug: the `fa` tasks (visits "1,2") died in 1 s with the apptainer help text: `--env ...,NIV_VISITS=1,2` is split on the comma, so
  `2` was read as a malformed key=value (ret tasks use visits "2" and were unaffected). Fix: `export APPTAINERENV_NIV_VISITS=$visits`
  and an explicit non-empty check of persons/visits/arms. Same class of bug for any env value containing a comma.
- Any candidate: `bash score_candidate.sh CAND cap250` already submits enrolment, `ret` and `fa` arrays and the collector. Manual rerun for an
  existing tag (enrolled zips `zips/enrolled/TAG_k0..8.zip` present): `sbatch --array=0-8 --exclude=galvani-cn122,galvani-cn116
  enrol_run.sbatch TAG fa`, then `sbatch collect.sbatch TAG cap250` (reports `false_recognition` per package).
- Lane E timing (3136813/4) failed separately: `zips/v66_cp1_cap250.zip` was a symlink into another release dir, dangling inside the
  container. Now copied; `build_arms.sbatch` copies, `timing.sbatch` fails on any symlink in zips/.
