# Dispatch log · 21 Nov

- Hard-only Team B · 02:49–02:50 CET, source `abc700f0`. Leonardo `cap250/runs/final-hard-teamb-21nov` verified all four staged ZIP hashes. Local v3 replay and both real-window CPU smokes passed. Nine15 A100 smoke 65001147 → CPU zero 65001148, terminal array 65001149, A100 timing 65001150 and cap250 pair 65001151. Nine25 A100 smoke 65001152 → CPU zero 65001153, terminal array 65001155, A100 timing 65001156 and cap250 pair 65001157. All full jobs use `afterok`. Galvani rejected batch authentication. [Archive and run record](final_hard_teamb.md)
- Hard-only Team B · completed 03:04:44 CET. Both ZIPs passed all load faults, 379-window bitwise zero parity, full graph/eager label parity in both orders and the official cap250 pairs. A100 additive runtime projections including graph capture and actor restore were 2,989.9 s for nine15 and 2,930.7 s for nine25. [Final evidence](final_hard_teamb.md)

- CBraMod speed · 01:20 CET, registered and staged at `4efd052f`. Leonardo `cap250/runs/cb-speed-21nov`: GPU smoke 64996748 → paired member benchmark 64996749 and public label array 64996751 (both `afterok`). Local v3 replay and both exact-archive CPU smokes passed. Galvani batch SSH denied authentication. [Run record](cb_speed.md)
- CBraMod mechanism and worker probes · 01:22 CET: optional vectorization and whitening smoke 64996905 → comparison 64996906, source `e4cc0584`. Worker-image CUDA probe 64996880. Native GPU smoke 64996748 stopped at an indirect harness import before either archive loaded. The dependent benchmark and label array were canceled by Slurm.
- CBraMod speed retry · 01:24 CET, source `bd6ebeee`: use the staged harness directly. Native smoke 64996944 → benchmark 64996948 and label array 64996949. The optional comparison uses the same corrected import. Worker-image probe 64996880 confirmed that the cluster driver reports CUDA 12.2 and cannot initialise the CUDA 13.0 wheel.
- CBraMod worker image retry · 01:28 CET, source `26cd457d`: Slurm binds NVIDIA's CUDA 13.0 compatibility libraries into the existing SIF for the job process. Worker smoke 64997117 → benchmark 64997118 and terminal array 64997119. The ZIPs and SIF are the same bytes as before the retry.
- CBraMod plain-only scope · 01:46 CET, source `eb54dc3a`: the 16-order result refuted the expected R1a gain. Retired the derived R1a archive and canceled its outstanding terminal tasks. The exact plain archive already passed terminal jobs 64996949_0–3 and 64997119_0–3. Worker smoke 64998037 → exact plain archive benchmark 64998049. [Result motivating scope change](laneR1a_bases.md) · [CBraMod evidence](cb_speed.md)
- CBraMod nine25 extension · 01:55 CET, source `79ea5b21`, Leonardo `cap250/runs/cb-speed-team-b-21nov`. Both exact derivatives passed local smoke. Worker-image CBraMod-only smoke 64998396 → benchmark 64998397 and terminals 64998398. Team B feature-graph smoke 64998399 → benchmark 64998401 and terminals 64998402. Each benchmark and terminal array depends on its own smoke. Galvani batch SSH again denied authentication. [Team B parent evidence](laneTB_blend.md) · [Speed evidence](cb_speed.md)
- CBraMod and nine25 · verified by 02:06 CET. Plain benchmark 64998049, both nine25 benchmarks, and all eight nine25 terminal tasks completed. The faster nine25 ZIP uses both CUDA graphs. Both deliverables matched every reference posterior and terminal label across the registered public checks. [Hashes, timings and projections](cb_speed.md)

- 00:20 Fable clean room (~/Documents/GitHub/ni26-cleanroom, b801a93): reviewed, approaches A-D deliberately NOT dispatched.
  A (adaptive population decoder, est. N 0.50-0.60) and B (personal experts) are weaker forms of mechanisms cap250 already
  has (cap250 held-new 73.7 % exceeds the clean room's own same-session supervised probe ceiling 0.72); B's ranking test is
  covered by redteam H1-H5 and laneRP. C (Graz cue-order prior) is void: organisers removed chronology from sealed delivery
  "to avoid tricks ... solving the track without using EEG" (13 Nov, memory sealed-order-not-chronological; FAQ L109-113).
  D (REVE probe) est. 0.36-0.50, closed earlier (foundation models). Hourly tick: do not dispatch these.

- Lane TB · real production-20261120-32a94ab7 fold packages, orders 71120–71123, Team B R1 encoder blend at 15% and 25%, with a fixed three-member trim. Registered in the ledger before scoring. Leonardo Slurm; Galvani batch SSH rejected authentication, so R1 weights will be refitted if the original artifacts remain inaccessible. Results due Sat 09:00 CET. [Lane card](laneTB_blend.md)
- 00:30 Reversed per user: Fable A+B dispatched to GPT build thread mcp:da9a9c35-9ca8-4053-b4f2-4f26cc6eb0f8 (clean room repo, results by 07:30, shuffled + chronological delivery). C still not built. 07:53 cron feeds RESULTS.md plus de-blinded main-system data points back to Fable (design only) -> PROPOSALS_2.md; hourly tick dispatches each proposal to GPT.

- RT-1 · dispatched 20 Nov 22:58–23:00 CET for the Sat 08:30 CET handoff. Leonardo H1/R0z and H5 first (64988017–64988060), then full R0z, H2 donor swap, H3 visit 1, and hardcb_safe reference (64988092–64988102). H4 reused Lane RP 64986260. Registered before scoring at `050198b1`. Slurm run `cap250/runs/redteam-21nov`; exact IDs and dependencies in its `jobs.tsv`. [Handoff](redteam_handoffs.md)

- Lane TB jobs · Leonardo `cap250/runs/laneTB-21nov`. Real-data fit smoke 64988026 completed. Fit array 64988029, P01 live smoke 64988278, score streams 64988281, collector 64988283, conditional ZIP and CPU blend-zero job 64988385. Scoring waits on the live smoke.

- RT-1 · 20 Nov 23:26 CET: kappa40 cleared its PROTEUS comparison. Registered Scherer NI-V B-ret confirmation (`9ff53b5a`), submitted Leonardo smoke 64989686 and dependent array 64989695 (nine people). [Handoff](redteam_handoffs.md)

- Lane TB retry · all 12 fits completed. Live smoke 64988278 matched pinned labels, then failed bitwise posteriors at the first GPU fine-tuning update (max 4.034e-7). No fresh orders scored. Revised smoke 64990355 → streams 64990356 → collector 64990357 → conditional ZIP/CPU zero 64990358, all afterok.

- Lane TB second retry · A100 repeated hardcb_safe pre-FT posteriors differed by 3.291e-9 (identical pinned labels). The zero arm now checks bitwise identity inside each live mixer call and retains the 2e-6 cross-run bound plus exact labels. New chain 64991339 (smoke) → 64991340 (streams) → 64991341 (collect) → 64991342 (conditional ZIP and CPU zero).

- RT-1 · 21 Nov 00:00 CET: all H1–H5 checks and all nine kappa40 NI-V confirmations completed. [Results and intervals](redteam_handoffs.md)

- Lane TB update 21 Nov 00:05 CET · 64991339 passed all pinned labels and same-call mixer identity. Its unchanged hardcb_safe control had max posterior Δ 0.021349 and zero label changes, so GPU cross-run posterior drift is logged. New smoke 64992565 → streams 64992566 with a baseline repeat on 71120 → collector 64992567 → conditional ZIP and CPU zero 64992568.

- Final R1a · 21 Nov 00:09 CET, registered in `b23f5aff`. Leonardo Slurm run `cap250/runs/final-r1a-21nov`, source `7aa4a05d`. Hard smoke 64992707 → A100 cap250 qualification 64992708, recognised replay/timing 64992709, CPU held-new and fault injection 64992739. Hardcb smoke 64992710 → qualification 64992711, recognised replay/timing 64992712, CPU held-new and fault injection 64992751. CPU checks use the Booster allocation after the CPU partition rejected the account. Galvani batch SSH denied authentication. Results due 09:00 CET. [Evidence](final_r1a.md)

- Final R1a tape check · 21 Nov 00:11 CET, source `a9bc7f5e`. Hard smoke 64992918 → full tape 64992919. Hardcb smoke 64992932 → full tape 64992934. Both final archive smokes completed before the full checks. The tape jobs compare the exact ZIP and the fold-scoped solver on verified tape rows. [Evidence](final_r1a.md)

- Lane TB timing recovery · 21 Nov 00:20 CET. Smoke 64992565 completed with exact pinned labels and same-call mixer identity. Stream array 64992566 remains active. The cold-load timing was invalid. Warm A100 timing 64993397 and streams feed collector 64993398, then conditional ZIP/CPU zero 64993399.

- Final R1a CPU retry · 21 Nov 00:26 CET. Both 15-minute jobs (64992739 and 64992751) timed out after the parent had completed P01/1 with 17 stock refits and no recognition. The 45-minute retries 64993648 (hard) and 64993649 (hardcb) started after the timeouts. Both A100 qualification jobs, both recognised replay jobs and both full tape jobs started around 00:27 CET. [Evidence](final_r1a.md)

- Final R1a terminal scores · 21 Nov 00:36 CET, source `d9abd7fb`. Hard smoke 64994099 → terminal-score check 64994100. Hardcb smoke 64994101 → terminal-score check 64994102. The exact ZIPs each refitted in four recognised streams but kept all 762 hard labels. The follow-up traces scores before argmax on the same deliveries to check that the refit reaches the decision. [Evidence](final_r1a.md)

- Final R1a original tape coverage · 21 Nov 00:58 CET, source `160e7ee9`. Hard smoke 64995389 → replay 64995390. Hardcb smoke 64995391 → replay 64995392. The follow-up checks the remaining four original orders after the first 16 direct comparisons yielded no full-stream match. [Evidence](final_r1a.md)

- Final R1a · checked by 01:12 CET. All intended jobs reached terminal state. Both 45-minute CPU retries passed, as did both terminal-score and cap250 qualification jobs. The full original-tape comparisons returned 0/32 exact matches for each ZIP. Both solvers reproduced full clean H1 tapes with the matching fold assets. [Final evidence and archive hashes](final_r1a.md)

- Lane TB collector retry · 21 Nov 01:33 CET. All 12 streams and warm timing completed. Collector 64993398 failed to serialize a NumPy changed-label count, so its dependent ZIP job 64993399 was cancelled by Slurm. Atomic JSON fix `ca8947f4`. Collector 64997382 reuses the 60 existing prediction files and feeds conditional ZIP/CPU zero 64997383.

- Lane TB completed · 21 Nov 01:46 CET. Collector 64997382 and conditional ZIP job 64997383 succeeded. Nine25 passed exact CPU blend-zero on 379 windows. [Results and archive receipt](laneTB_blend.md)

- Lane TB-grow · Leonardo `cap250/runs/laneTB-grow-21nov`. Preregistered at `22601c4d`. Seed-1 Team B real-data smoke 65001030 → 12 fold fits 65001034. Team C returning smoke 65001031 → two 60-epoch fold fits 65001036. Both arrays depend on their smoke. Local v3 replay printed OK. Galvani batch SSH denied authentication. [Protocol](../../docs/LEDGER.md#lane-tb-grow--dose-insertion-point-and-independent-sources-registered-21-nov-2026-before-scoring)

- Lane TB-grow replay · source `ba604057`. Leonardo live smoke 65001193 → paired A100 timing 65001195 and 48 selection tasks 65001196 → selection freeze 65001197 → 48 single-choice confirmation tasks 65001198 → report collector 65001199. Every arrow is `afterok`; the confirmation array reads the frozen choice and replays its fixed comparators.

- Lane TB-grow hook review · held pending smoke 65001193 before any live replay. The inherited mixer would still call a loaded Team B encoder at zero weight, so native and terminal arms now bypass that hook and check that Team B consumes each window exactly once. Updated source before releasing the smoke. Job IDs are unchanged.

- Lane TB-grow · all 14 new fits finished and live smoke 65001193 started from `075bc70f`. Slurm time limits for pending timing, selection and confirmation jobs were raised to 2:00, 2:15 and 1:00 hours respectively after measuring P01 smoke throughput. Order sets, dependencies and arms are unchanged.
- 03:15 Fable round 2 sent early (clean-room RESULTS.md ebfa4aa ready; A 55.5 % shuffled LOSO) with de-blinded data points incl. Team B blend; ask -> PROPOSALS_2.md (design only). 07:53 cron cancelled. Hourly tick dispatches each proposal to GPT.

- Lane TB-grow replay smoke 65001193 completed with 380 pinned labels exact and prefix/reset checks on all arms. The selection and confirmation throttles were raised to 48. The first returning selection tasks stopped before Team C because `train.py` defaulted to three members, which the six-member receipt check rejected. Source `d1fc6d5c` passes the registered E6 roster explicitly. Leonardo 65003079 → corrected fits 65003081 → both returning live smokes 65003089 → missing returning parts 65003090. The repair also waits for the original array to exit and reuses only complete prediction/receipt pairs. Timing 65001195 and repair feed freeze 65003091 → confirmation 65003092 → report 65003093. Original downstream jobs 65001197–65001199 were canceled while pending. No selection or confirmation score was opened.

- Lane TB-grow additional confirmation · preregistered at `70a42885` before selection. The Galvani batch probe passed. Leonardo already has the exact fold fixtures and banks, so it runs this block. Extra array 65003770 waits on freeze 65003091 and covers only the winner and fixed comparators on 71172–71179. Collector 65003772 waits on the extra array and primary confirmation report 65003093. No jobs use 71156–71171.
- 04:0x Fable PROPOSALS_2 (cleanroom 6a608db: P1 eye-position view g, P2 class-map arbiter, P3 scheduled hedge) dispatched as one GPT-6.2 Sol task (decisive tests first, then build what passes on hard+TB nine25); card fable_round2.md.
