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

## 3. R1a on the hard-only base, and R1a on hardcb at 16 orders (Leonardo, account 1636) → card `laneR1a_bases.md`
Registered in `docs/LEDGER.md` ("R1a on the hard-only base…"). Exact final solver (pinned parent and R1a ZIPs of `final-r1a-21nov/zips`, `check_r1a.load_templates` with the real `production-20261120-32a94ab7` fold packages `ret_P04`/`ret_P10`, visit 1 enrolled), full-session orders 71000–71015, four arms per order (hard, hard_r1a, hardcb, hardcb_r1a). Run dir `/leonardo_work/AIFAC_P02_1636/ni26/cap250/runs/overnight-21nov-r1abases` (`jobs.tsv`, `logs/`, `out/`); scripts `scripts/overnight21/r1a_bases.{py,sbatch}`, `r1a_bases_collect.py`.
- Smoke 64996733 (hardcb P04 order 71000 visit 2): 0 mismatches against the stored hardcb and R1a tapes, recognised at 29, one late refit at 100. Held-new smoke 64996735 (hard P01 71000): recognition −1, 0 labels changed.
- Item 1 (priority): 64997021–64997024 (orders 71000–71003: hard/hardcb × P04/P10), 64997025–64997029 (71004–71007; 64997027 is not ours). Item 2: hardcb 64997030/31/34/35 (71008–71015), hard 64997032/33/36/37. Held-new zero-change P01–P03 visit 1, orders 71000–71007: hard 64997038–40, hardcb 64997041–43. Each returning chunk is 4 orders × 2 visits × 2 models (~15–25 min on one GPU).
- DONE (all 24 jobs completed): result in `laneR1a_bases.md` (R1a −0.14 [−0.32, +0.02] on hardcb at 16 orders, −0.11 [−0.31, +0.06] on hard; not supported).
- Collect (local, numpy only): `scp -r cineca-leonardo:/leonardo_work/AIFAC_P02_1636/ni26/cap250/runs/overnight-21nov-r1abases/out /tmp/r1ab_out && python scripts/overnight21/r1a_bases_collect.py data/proteus_windows.npz /tmp/r1ab_out`. The collector lists absent chunks implicitly (orders both arms lack are skipped); the info files report recognition, late refits, held-new changed labels and stored-tape mismatches (hardcb 71000–71003 must be 0).
- Registered rule: a base is supported if its people-nested lower bound > 0, refuted if the mean ≤ 0, undecided otherwise. Prediction hardcb +0.15, hard +0.15. Not a gate: runtime margin still decides which base can ship.
- Backlog item 3: nothing in `BACKLOG.md` was safe to start unattended (B4/B8 need a fold DAG rebuild, B9/B11 wait for packages to qualify on the sealed GPU class).

## 4. Final hard-only Team B confirmation (submitted 21 Nov 04:00-04:30 CET; collect by 08:30 → `final_hard_teamb_confirm.md`)
Registered in `docs/LEDGER.md` ("Final hard-only Team B confirmation"). Nothing uploaded. Galvani release `/mnt/lustre/work/gehler/gfh527/ni26_final_confirm_20261121` (`zips/` pinned by `zips.sha256`, `out/`, `logs/`), scripts `scripts/overnight21/confirm_{gpu.sbatch,niv_setup.sh,niv_queue.sh}`. Galvani caps a user at 300 submitted tasks (arrays count).
- **RTX 2080 Ti timing** (`confirm_gpu.sbatch`, worker SIF). Smokes 3145716 (hard nine25: load faults, dead guard, 72-window eager/graph parity both orders) and 3145717 (hardcb nine25 both graphs: graph 8.2 ms vs original 27.3 ms on the 2080 Ti, graph = eager posterior error 0.0, forced capture failure falls back to eager). Qualification pairs (cap250, cand, cap250, cand) 3145739 nine25, 3145740 nine15, 3145741 hardcb nine25 cbgraph tbgraph, 3145742 hardcb cbgraph. Per-window benches, two repeats on shared nodes: hard 3145743/47 (nine25), 3145744/48 (nine15); cb plain 3145745/49; cb graph (nine25 both graphs) 3145746/50. Third repeat with `--exclusive` (nodes were shared with other jobs, CPU contention distorts the CPU-side actors): 3145757 / 3145758 / 3145759 / 3145760. 2080 Ti terminal graph/eager label parity, nine25, order 71000, parts 0/1: 3145751/52.
- **NI-V** (releases `ni26_niv_finalA_20261121` orders 71000-71007, `ni26_niv_finalB_20261121` orders 71008-71015; tags hss = hard_safe_st, hss_n25, hss_n15; setup `confirm_niv_setup.sh`, queue `confirm_niv_queue.sh`). A: smokes 3145718/22/26, B-new arrays 3145719/23/27, enrolments 3145720/24/28, B-ret 3145721/25/29, collectors 3145730 (n25 vs hss) / 3145731 (n15 vs hss). B: hss and hss_n25 submitted (smokes 3145732/35, B-new 3145733/36, B-ret 3145734/37); hss_n15 and the two B collectors hit the 300-task cap and are queued by a second pass (below).
- **Confirmation orders** (Leonardo `cap250/runs/laneTB-confirm-21nov`, `verify.py --base hard`, orders 71156-71171, 2 per chunk): smoke 65003908 (base arm must equal the stored hard tape, nine25 must change labels) → eight arrays 65003909-65003916 (12 folds each, 55 min) → collector 65003988 (`confirm_collect.py`, new 16 orders and pooled with the 8 Lane TB-verify orders 71124-71131). Collector self-check on the Lane TB-verify output (must reproduce hard nine25 +0.97 / nine15 +0.77 held-new): 65003984.

## 5. Finalists: seed robustness and 16-order head-to-head (submitted 21 Nov 05:45–06:15 CET; collect by 09:30 → `finalists_21nov.md`)
Registered in `docs/LEDGER.md` ("Finalists seed robustness and head-to-head"). Nothing uploaded. Leonardo run `cap250/runs/finalists-21nov` (account 1636), scripts `scripts/overnight21/fin_{fit,verify}.sbatch`, `finalists_collect.py`, `scripts/laneTB/{verify,grow_run}.py` (`--weights/--arms`, `finalists` phase).
- **S1 seed 1** (the TB-grow fits `laneTB-grow-21nov/b`, roster audited by `verify.audit`): smoke 65007320 (first smoke 65007266 failed on shell quoting, 65007317 cancelled; dependents resubmitted) → score arrays, nine15 and nine25 on orders 71124–71131: 65007328 / 65007329 / 65007331 / 65007332 (2 orders each, 12 folds). Base and seed-0 tapes: `laneTB-verify-21nov/out/hard`.
- **S1 seed 2**: fit smoke 65007269 → 12 fits 65007279 (`b_s2/<fold>/weights.pt`) → score arrays 65007292 / 65007294 / 65007299 / 65007300 (same orders).
- **S2 head-to-head, orders 71200–71215**: `finalists` smoke = task 0 of 65007268 (new_P01 chunk 0), then tasks 1–95 of 65007268's follow-up array (job name `fin-hh-grow`: hard, hard+nine25, ba_terminal); verify arrays `fin-hh-v-<first order>` (hard, nine15, nine25) for orders 71200, 71202 … 71214, 12 folds each, dependent on smoke 65007320.
- **Galvani NI-V release C** `ni26_niv_finalC_20261121` (orders 71200–71207; hss / hss_n25 / hss_n15; A's enrolled packages): smokes 3146194 / 3146197 / 3146200 → B-new 3146195 / 3146198 / 3146201 (36 each), B-ret 3146196 / 3146199 / 3146202 (9 each); collectors 3146203 (n25 vs hss), 3146204 (n15 vs hss). Scripts `fin_niv_{setup,queue}.sh`. ba_terminal not included (its NI-V is in `final_ba_terminal.md`).
- Collect: `ssh cineca-leonardo`, then `$CAP250_PYTHON scripts/overnight21/finalists_collect.py RUN --data $CAP250_DATA --verify .../laneTB-verify-21nov/out/hard --json X --part seeds|h2h` (numpy only; fails if a chunk is missing).

Update 06:50 CET: all Section 4 jobs have finished. Timing and Solver end-to-end reruns after the first exclusive attempts failed (the `--exclusive` allocation listed eight GPUs and broke the apptainer `--env` list; fixed with `APPTAINERENV_`) and starved: e2e jobs 3146075/77/177/190/191, A100-PCIe 3146334/35, repeats 3146345-47; NI-V enrolment first failed (symlinked zips not bound; fixed in `enrol.sbatch`, re-run 3146003-05 and ret arrays 3146006-11), B hss_n15 B-new re-run as 3146083. Results: [final_hard_teamb_confirm.md](final_hard_teamb_confirm.md).

## 6. Leonardo fill through 18:00 CET · dose and bank growth

Run `cap250/runs/leonardo-fill-21nov` on account 1636. Preregistration and receipts are in [leonardo_fill_21nov.md](leonardo_fill_21nov.md). The finalists S1/S2 reports were read first. S1 65011278 met the three-seed stability rule. S2 65011279 used only 71200–71215.

- Dose 71400–71431, source `4a3d7872`. Smokes 65011374–65011376 passed v3, exact pinned labels and zero-mixer equality. Dose arrays 65011605/65011606 then collectors 65011607/65011608/65011609 write `dose1.json`, `dose2.json` and `dose.json`.
- Growth, source `be01260a`. Live smokes 65011663–65011665. Next are nine remaining fixtures, paired A100 timing jobs, screen 71500–71507, then a single frozen choice on 71508–71523 if released. All use actual folds and the full Solver.
- Queue at 08:08 CET: 26 running, 270 pending across the account. The already-running Fable Round 3 and Galvani runtime work are independent. The Codex heartbeat starts 08:57 CET and checks every 50 minutes. Lane report due by 17:30 CET. Final candidate due by 18:30 CET.
- 08:15 CET growth update. Smokes 65011663–65011665 passed before the remaining fixtures 65011888, six timings 65011889, and 48 screen tasks 65011890. Freeze 65011891 waits for screen and timing. Release 65011892 waits for freeze. Its job receipts appear in `grow/{screen,held}-jobs.tsv`. Collect `grow/selection.json` and `grow/confirmation.json` inside the same run.

## 7. Galvani standing queue (submitted 08:30-08:45 CET, runs unattended to about 13:00)

Registered in `docs/LEDGER.md` ("Galvani standing queue 21 Nov"). Nothing uploaded. Scripts `scripts/overnight21/{fin_niv_setup_gen.sh,fin_niv_queue_gen.sh,standing_queue21.sh,release_e_submit.sbatch}`; release dir `/mnt/lustre/work/gehler/gfh527/ni26_final_confirm_20261121`.

- **NI-V release D** (`ni26_niv_finalD_20261121`, orders 71016-71023; hss / hss_n25 / hss_n15, A's enrolled packages): smokes 3146467 / 3146470 / 3146473, B-new 3146468 / 3146471 / 3146474 (36 each), B-ret 3146469 / 3146472 / 3146475 (9 each), collectors 3146476 (n25 vs hss), 3146477 (n15 vs hss).
- **NI-V release E** (`ni26_niv_finalE_20261121`, orders 71024-71031, set up, not yet submitted): Slurm cpu job 3146514 (`s21-relE-submit`) runs `fin_niv_queue_gen.sh E` after both D collectors, once at most 150 tasks are submitted (300-task cap). Its job ids appear in `logs/slurm-3146514.out` of the release dir.
- **Chained e2e timing** (`e2e_timing.py`, signed ZIPs, afterany chains of 4, about 15 min each; node in each `out/e2e-<jobid>.json`): 2080 Ti hard family (cap250, nine25, nine15) 3146478-81, 3146482-85, 3146486-89, 3146490-93; 2080 Ti hardcb cbgraph (cap250, cbgraph) 3146494-97, 3146498-3146501; A100-PCIe hard family 3146502-05, 3146506-09; A100-PCIe cbgraph 3146510-13.
- Not run: `hard.zip` identity sanity on NI-V and the `fa` arm (no hard.zip is in `zips/`; the fused-conv `_fc` variants belong to the Team B overhead lane, `teamb_overhead.md`).
- Collect NI-V: `ssh mlcloud-galvani 'cd /mnt/lustre/work/gehler/gfh527/ni26_niv_finalD_20261121 && python3 -c "import json;d=json.load(open(\"out/report_hss_n25_vs_hss.json\"));print(json.dumps(d,indent=1)[:3000])"'` (same for hss_n15, and finalE). Collect timing: `ssh mlcloud-galvani 'cd /mnt/lustre/work/gehler/gfh527/ni26_final_confirm_20261121 && python3 -c "import json,glob;[print(f,json.load(open(f)).get(\"summary\",\"\")) for f in sorted(glob.glob(\"out/e2e-31464[7-9]*.json\")+glob.glob(\"out/e2e-31465*.json\"))]"'` (check keys in one file first).
- 09:26 CET results. Both dose arrays and all three reports completed. Growth timing/screen completed, froze `ba_six` and released held array 65013453. Its collector 65013454 completed. No growth arm met the registered quality and runtime requirements together. Nine25 remains the lane lead. The returning block sensitivity released the registered 71600–71631 follow-up under `cap250/runs/leonardo-fill-retest-21nov`. [Results](leonardo_fill_21nov.md)
- 09:32 CET upkeep. Retest smokes 65014371–65014373 passed, arrays 65014554/65014555 and report 65014556 accepted. The report writes `leonardo-fill-retest-21nov/doseall.json`. The second bank wave is staged separately at `leonardo-fill-roster-21nov`, source `30074abb`, with the same smoke and freeze dependency gates.
- 09:34 CET roster dispatcher. Smokes 65014561/65014571/65014573 feed controller 65014651. It checks the smoke receipts and records the screen sequence under `leonardo-fill-roster-21nov/grow/screen-jobs.tsv`.

## 8. Galvani afternoon refill

Submitted at 12:15 CET. 42 Slurm job IDs (171 tasks including arrays), pre-registered in [LEDGER](../../docs/LEDGER.md#galvani-afternoon-refill--21-november-registered-before-submission). Local v3 replay printed `OK`. Prior paired timing job 3146946 exercised all three signed archives through the same timing script. Release F has its own fresh and pinned-order smoke gate. At submission, timing on cn102/cn116/cn121 and Release F setup were running. A100 first repeats were pending resources. Later repeats become eligible at 14:25 and 16:35 CET after the prior repeat on each node. Target finish ~17:00 CET, subject to Slurm placement.

- **E2E** (cap250 / nine25_fc / nine15_fc in every job). 2080 Ti cn102 3147371–73, cn116 3147374–76, cn121 3147377–79. A100 cn205 3147380–82, cn206 3147383–85, cn218 3147386–88, cn221 3147389–91, cn225 3147392–94, cn226 3147395–97, cn229 3147398–3147400. One GPU per node per repeat. cn102 is preemptable. cn116 adds paired nine15_fc Solver timing.
- **12:17 CET placement update.** To avoid first-start forecasts beyond the target, moved the pending timing chains with the same job IDs to cn235 (3147383–85), cn234 (3147386–88), cn231 (3147395–97), and cn239 (3147398–3147400). cn231 uses the preemptable A100 partition. The first cn234 run retains its original cn218 job name. Use Slurm’s actual NodeList for collection. The other six chains retain their original nodes.
- **NI-V Release F** (71032–71039). Setup 3147359. Smokes hss 3147360, nine25 3147361, nine25_fc 3147362 → fresh and pinned-order gate 3147363. B-new / B-ret arrays hss 3147364/65, nine25 3147366/67, nine25_fc 3147368/69 → collector and complete fused/unfused label and row identity 3147370. Arrays cap simultaneous use at 12 GPUs total. No ba_six jobs.
- **Job receipts:** `ssh mlcloud-galvani 'cat /mnt/lustre/work/gehler/gfh527/ni26_niv_finalF_20261121/jobs.tsv'`. For node and final states, query `sacct -X -j` with the job IDs.
- **Collect** on the local machine after completion:

  ```bash
  mkdir -p /tmp/ni26-refill21/{timing,niv}
  ni26_remote=/mnt/lustre/work/gehler/gfh527
  for job in $(seq 3147371 3147400); do
    scp "mlcloud-galvani:$ni26_remote/ni26_final_confirm_20261121/out/e2e-$job.json" /tmp/ni26-refill21/timing/
  done
  scp "mlcloud-galvani:$ni26_remote/ni26_niv_finalF_20261121/out/report_hss_n25_fc_vs_hss.json" /tmp/ni26-refill21/niv/
  scp "mlcloud-galvani:$ni26_remote/ni26_niv_finalF_20261121/logs/refill-niv-3147370.out" /tmp/ni26-refill21/niv/
  jq '{headline, parity_vs_nine25, scherer_new, scherer_returning, sealed_weighted}' /tmp/ni26-refill21/niv/report_hss_n25_fc_vs_hss.json
  ```
- 12:15 CET resumption. The morning dose retest and second bank wave completed. The user accepts a positive paired lower bound in the target regime and at most a 0.30 pp loss in the other. Report the official 50/50. Galvani `ni26_ba_six_fc_20261121` runs two signed builds, zero/active identity, official pair, six requested RTX nodes and separate Scherer NI-V. Leonardo `leonardo-fill-ba-six-21nov` runs the fused 71800–71815 confirmation. Exact dispatches and receipts are in [leonardo_fill_21nov.md](leonardo_fill_21nov.md).

## 9. Leonardo afternoon confirmation

Run `cap250/runs/ni26-afternoon-71900` on AIFAC_P02_1636. Submitted at 12:36 CET with 2 running and 194 pending tasks in this run. The final fold wave is eligible from 16:25 CET, with collection targeted for about 17:00 CET subject to placement. [Preregistration](../../docs/LEDGER.md#leonardo-afternoon-confirmation--21-november-registered-before-dispatch).

- **Paired A100-SXM E2E.** Three repeats of `e2e_timing.py` under Slurm 65024312 (one GPU and one rank per node, three nodes), after the shortened graph replay smoke 65024311. Each node loads and streams cap250, `final_stack_hard_safe_st_team_b_nine25_tbgraph_fc.zip` (`7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07`), and `final_stack_hard_safe_st_team_b_nine25_ba_terminal_fc.zip` (`1ef25b365c609354f275295f7e36fa9d1d71bbb02c8815ff4362ead078e6b23e`). Timing source snapshot `f309cc4a`.
- **Real fold confirmation.** Frozen Lane TB-verify code, hard / nine15 / nine25, one full shuffled delivery per order and 12 folds per array. The pinned-order smoke is 65024411. Arrays in order are 65024412/13/14/16 (71900–71903), 65024417/18/19/20 (71904–71907, eligible 13:35), 65024421/22/25/27 (71908–71911, eligible 14:45), and 65024434/36/37/40 (71912–71915, eligible 16:25). Collector 65024442 waits for the last wave and writes the paired 50/50 new/returning report. Returning covers P04 and P10. Local v3 replay printed `OK`. The collector reproduced the prior 71156–71171 reference on real fold outputs. Initial smoke 65024273 rejected an unsupported wrapper argument. Its fold chain was cancelled by Slurm before scoring. The replacement uses the original `verify.sbatch` entry point.
- **Collect completed results by 17:00 CET.** Check `State`, `End` and timing node allocation before copying. Rank-to-node receipts are in `logs/ni26-719-e2e-65024312_*.out`.

  ```bash
  remote=/leonardo_work/AIFAC_P02_1636/ni26/cap250/runs/ni26-afternoon-71900
  ssh cineca-leonardo "cd '$remote' && cat jobs.tsv && sacct -X -j 65024312,65024442 -o JobID,State,End,NodeList"
  mkdir -p /tmp/ni26-afternoon-71900
  scp "cineca-leonardo:$remote/out/e2e-65024312-*.json" /tmp/ni26-afternoon-71900/
  scp "cineca-leonardo:$remote/out/confirmation.json" /tmp/ni26-afternoon-71900/
  jq 'to_entries[] | select(.value | type == "object" and has("projected_s")) | {arm: .key, warm_ms: .value.ms_per_window, added_ms: .value.added_vs_ref_ms, projected_s: .value.projected_s}' /tmp/ni26-afternoon-71900/e2e-*.json
  jq '.contrasts | map_values({mean_50_50, ci80_50_50, new: .new.mean, returning: .returning.mean})' /tmp/ni26-afternoon-71900/confirmation.json
  ```
- 12:40 CET afternoon receipts. Galvani paired Scherer arrays 3147546/3147547 and report 3147548. Official cap250 pair 3147535 ran all four replays and missed the symmetric score tolerance at +0.542 pp. Signed fused-lead official pair 3147630 queued. Leonardo prep 65023987, confirmation 65023988 and report 65023989 were released after all three fused-fold smokes.
- 12:55 CET qualification. CPU zero, active identity, Scherer NI-V and signed-lead official pair completed. Profiled cn122 exceeds the 3,450 s target with a normal cap250 baseline. The ordinary Solver e2e repeats `3147645–3147650` are queued on six distinct nodes, including cn116 and cn122. No model bytes changed.
- 13:12 CET timing. Six normal unprofiled runs complete. Worst is cn114 at 3,468.3 s, so fused `ba_six` misses 3,450 s. The measured signed nine25 lead passed. Galvani FP16 MANAS pilot 3147737/3147738 followed the registered timing miss. Leonardo confirmation remains in progress.
- 13:22 CET Turing pilot. Actual FP16 MANAS passed the 379-window comparison on cn114 and released seven full identity shards, zero, paired warm-up, six e2e nodes, Scherer NI-V and an A100 unchanged-dtype check. [Dispatch details](leonardo_fill_21nov.md)
- 13:36 CET confirmation. Fresh fused `ba_six` collector 65023989 completed with 50/50 +0.36 pp [80% +0.17, +0.54]. Turing FP16 label identity and native BF16 identity passed. The official fused-lead pair passed. Turing Scherer arrays 3147783/3147784 feed 3147785. A100 native-BF16 timing is 3147847.
- 13:41 CET RTX result. FP16 fused `ba_six` passed all six normal nodes, worst 3,259.2 s on cn116, and both CPU zero comparisons. Original fused `ba_six` missed on normal nodes. Only the repeat Scherer report and supplemental A100 timing remained.

## 10. Galvani FP16 decision repeat

Submitted at 13:57 CET from source `682677e0`. Galvani showed 47 active job records and 82 tasks with arrays expanded.
The local v3 replay printed `OK`. Signed archives and the prior zero receipt were reused. Fold packages for the pack-level
PROTEUS confirmation were not found at the Galvani run locations checked. [Preregistration](../../docs/LEDGER.md#galvani-fp16-decision-repeat--21-november-registered-before-dispatch)

- **Scherer NI-V 71848–71855.** B-new and B-ret pinned-order and fresh-order smokes 3147901/3147902.
  B-new 3147904–3147912 cover four two-order tasks per person in three waves of three people. B-ret 3147913 covers
  nine enrolled people. Collector 3147914 waits for both cohorts and reports paired 50/50 with the primary full-session
  shuffle. The same people appeared in the first release. These job IDs belong to the disjoint order release.
- **RTX 2080 Ti two-arm e2e.** Smoke 3147903 (cn111) gates the first repeats. The second repeat on each node depends
  on its first and becomes eligible at 16:55 CET, targeting completion around 17:30 CET subject to placement.

  | Node | Repeat 1 | Repeat 2 |
  |---|---:|---:|
  | cn111 | 3147915 | 3147916 |
  | cn114 | 3147917 | 3147918 |
  | cn115 | 3147919 | 3147920 |
  | cn116 | 3147921 | 3147922 |
  | cn122 | 3147923 | 3147924 |
  | cn123 | 3147925 | 3147926 |

- **Collect** on the local machine. Run the NI-V and timing copies when their respective jobs complete.

  ```bash
  remote=/mnt/lustre/work/gehler/gfh527/ni26_fp16_decision_20261121
  dest=/tmp/ni26-fp16-decision
  mkdir -p "$dest"
  ssh mlcloud-galvani "cd '$remote' && cat jobs.tsv && sacct -X -j \$(cut -f2 jobs.tsv | paste -sd,) -o JobID,State,Elapsed,End,NodeList"
  scp "mlcloud-galvani:$remote/niv-results/ba_six_fc_niv_fp16_turing/report.json" "$dest/niv-71848.json"
  jq '{headline, new: .scherer_new.delta.primary_full, returning: .scherer_returning.delta.primary_full, half: .sealed_weighted.combined}' "$dest/niv-71848.json"
  scp 'mlcloud-galvani:'"$remote"'/out/timing-31479*.json' "$dest/"
  jq --arg arm final_stack_hard_safe_st_team_b_ba_six_fc_fp16_turing '
    {file: input_filename, cap_ms: .v66_cp1_cap250.ms_per_window,
     candidate_ms: .[$arm].ms_per_window, projected_s: .[$arm].projected_s, load_s}
  ' "$dest"/timing-31479*.json | jq -s '
    if length == 12 then {runs: ., worst_normal: (map(select(.cap_ms <= 190.5)) | max_by(.projected_s))}
      else error("expected all 12 timing runs") end
  '
  ```
- 13:56 CET lane recommendation. The exact local `final_stack_hard_safe_st_team_b_ba_six_fc_fp16_turing.zip` qualifies from this lane, with 50/50 +0.36 pp on fresh fused folds and 3,259.2 s worst normal compute proxy. Turing Scherer collector 3147785 and A100 compute check 3147847 completed. The negative Scherer returning point remains a flag. Both cluster queues were rechecked and the heartbeat continues through 18:00 CET. [Lane decision](leonardo_fill_21nov.md)
- 14:01 CET Leonardo heartbeat. Account snapshot 39 running and 115 pending. Dose, retest, roster and fused confirmation receipts verified. The scheduled afternoon waves cover the next check, with releases eligible at 14:45 and 16:25 CET.
- 14:07 CET heartbeat. Leonardo 11 running and 98 pending after a successful retry. Galvani 11 running and 67 pending. Smokes 3147901–03 completed. Repeat timings and Scherer report remained queued. Original cn111 job 3147536 completed at 3,289.3 s. [Verified receipt](leonardo_fill_21nov.md)
- 15:00 CET heartbeat. Scherer collector 3147914 and first FP16 timings on cn111, cn114 and cn115 completed. Worst normal FP16 projection 3,280.7 s. Cn116, cn122 and cn123 first repeats wait for resources, and six second repeats remain gated. Leonardo A100 triplicate 65024312 completed and fold collector 65024442 waits for the last wave. [Verified receipts](leonardo_fill_21nov.md#1500-cet-follow-on-receipts)
- 15:47 CET heartbeat. The nine pending Galvani timings produced no new receipt. Slurm estimated 02:01 CET on 22 Nov for resource-bound first repeats 3147921/23/25. The second repeats on those nodes still wait on their first jobs. Leonardo's 71908–71911 wave completed and the last wave remains scheduled for 16:25 CET. [Queue receipt](leonardo_fill_21nov.md#1547-cet-queue-receipt)
- 16:37 CET heartbeat. Cn122 first repeat 3147923 started early and completed at 16:38 CET with a normal 3,050.0 s projection. First repeats on cn116 and cn123 still await resources. Leonardo completed the first 12 final-wave fold tasks. [Receipts](leonardo_fill_21nov.md#1637-cet-queue-receipt)
- 17:20 CET verification. Final Leonardo collector 65024442 completed at 17:07 CET. FP16 second repeats 3147916/18/20/24 completed, all normal and under 3,450 s. Cn116 and cn123 still wait for resources, with their second repeats dependent. The Leonardo account fell below target with four active tasks. [Decision and receipts](leonardo_fill_21nov.md#1720-cet-decision-verification)
