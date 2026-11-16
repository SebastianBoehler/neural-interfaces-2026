# Overnight campaign: 16–17 November

User objective: combine findings, push promising directions and test new methods from first principles.
Target decision: 17 November evening, provisionally 18:00 CET. Top five is the objective, not a measured claim.
New compute is Galvani-only. Keep all user jobs within 32 running / 272 submitted, preserving 28 slots.
Use `mlcloud-galvani`; access requires eduVPN/SSH authentication. Never touch unrelated `d11-*` jobs.

## Evidence at 16 November, about 23:30 CET

- Latest official published board, updated 15:19 UTC: fifth displayed entry .839; fifth distinct team .837.
  Source: https://neural-interfaces26.github.io/leaderboard.html . No live full-board rank is verified tonight.
- Recorded uploaded v5.1b remains .8197668759. Prepared v6.6 cap250 is unchanged; no upload authorized.
  Earlier .824–.829 projection is an unvalidated scenario, not an official score or current top-five result.
- §294 all 15 arms completed. Stronger EEGNet/distillation doses worsened; no established new gain.
- §295 geometric posterior deltas were near neutral; none established a gain in this strong-system form.
- §299 FBL2 versus control +.096 pp [−.072,+.262]; versus FBL15 +.052 [−.050,+.157]. Exploratory.
- §300 locates EEG loss before count/CP: E05 −.248 pp at V6Fr, then −.174 after CP.
  Brier +.0028 and true-class margin −.0043 at V6Fr. It cannot separate mix/rebalancing/EM/TSEM.
- §302 temperature2 improves E05 by +.110 [.022,.180], but is −.064 [−.169,+.076] versus control.
  Temperature.5 worsens it; causal centering .25/.5 does not demonstrate a repair.
- §301 strong native fixed-head port passed CPU/A100, P01/P10 × within0/full0 ×62windows:
  relative feature error5.59e−9, probability error3.75e−8, no non-tied label changes, prefix/reset pass.
- Original §301 Ntrain failed objective parity. Training-only P01 audit shows identical features/scalers
  and different stopping: sklearn16iterations versus tight237; native/tight loss differ9.6e−10.
  §305 is a new registered converged-head test; never relabel the failed original parity as passing.
- Native paired P01 inference: cached-coefficient Python113.1–117.0ms, nativeCPU62.9–64.0ms;
  probability difference≤2.20e−8. Training full solve: Python19.2–19.6s, nativeCPU243.4–244.1s.
  Same objective target, different stopping tightness; no intrinsic-language or worker claim.

## Live owning documents and jobs

- §302 repairs: evidence `repair_302_complete_results.txt`; neutral ENU control `neutral_302.md`.
  ENU/scorer3115011–12 complete: ENU−control+.0047pp; ET2−ENU−.0684[−.1815,+.0731].
  Recovery from negative E05 does not establish useful EEG information above this neutral control.
- §303 compact-native adaptation repair: `lane5_repair_s303.md`; 80tasks3114754/scorer3114755 complete: connectivity flatten49.78% vsstatic48.90%.
  Original2080 checkpoints on A100 require TF32off; originalA100 retains its recorded-route defaults.
  Allguards passed. Reverse−repairperson+1.363[.842,1.905],10/10up; whole49.78%stillbelow76.22%.
- §304 conditional CORAL: `coral304_assembly.md`; first80fits3114483%6; continuation3114855.
  First80fits complete; seeds2–3 array3115042 and callback3115043 queued. Source-locked
  callbacks then queue seeds4–5 and complete240fold assembly→192streams→guarded chain.
  Source `coral_repair/`, chain subdirectory owned separately; all source/evidence committed.
- §305 native convergence: `native_core_s301.md`; corrected P01 smoke3114867, guard3114944
  also requires all-person feature/scaler audit3115026. Then3114945→3114946→3114947→
  zero3114948→24treatmentcells3114949→scorer3114950. Arrays sequential, group at most2.
- Paired native benchmarks:3114673; cached-coefficient Python smoke3114848→3114849.
  Both benchmarks complete; evidence in native_core_s301.md. §309 GPU-resident head: smoke
  3115174→paired benchmark3115175, private ni26_s309_device_20261116; original§305 untouched.
- Agents: routing_audit owns native/convergence/benchmark; galvani_failures owns resource/CORAL
  assembler/controllers; thread_context owns repair/neutral and CORAL full-chain integration.
  Inspect their latest state before duplicating work or changing their files.

## Overnight portfolio and decisions

1. Read complete guarded arms as they land; queue next tests without waiting for unrelated waves.
   Push small consistent positives, but report their uncertainty and repeated-selection bias.
2. Combine FBL2 with repairs/new members only through registered matched full-system tests,
   with singles and interaction controls. ENU separates useful EEG information from generic softening.
3. §308 strong structured tanh/linear and random/kernel heads registered; native bridge and
   realCPUgate3115212 passed. Full producer/scorer being completed; reserve-aware staged dispatch. Group≤2; posterior-only integration
   has an explicit old-stage limitation. Keep full strong linear controls and source-only selection.
4. §306 GPUv3DAG3115203–07 pending, CPUfixgate3115383→84→controller85 queued;≤2allocations. Rank audit3115158 and style
   3115162–63 passed: all14sessions have exactly one CAR null, no extra null at1e−4trace/41.
   This is descriptive only. Preserve trace power in future CAR40 controls; no target interpolation shown.
5. §307 real-dataCPUcrossfit/meta pilots3115208/3115210 passed. Array3115215 runs90crossfits%4
   →verifier3115216→10CPUmeta-fits3115217%2. §310strong-coreentropySIGNtransferregistered7f4992b; DAG
   is being completed, not yet queued. See meta_adaptation_design.md and source evidence.
6. User grants codebase ownership: revise architectures, decisions, splits and stages as useful.
   Whole alternatives whole; chain is a comparator, not a search boundary. Research block
   22:51:07–23:06:18CET completed; research_4h_16nov.md records new9–13Nov papers and next
   representation/soft-target learning tests. Both50min and4h heartbeats active through17Nov evening.

Read docs/STATE.md, docs/LEDGER.md and OFFICIAL_CONTRACT.md before new work. Register every run
before scoring. Preserve original evaluator/metric/tensor, immutable snapshots, package/submodule hashes,
all-session reset and no-future guards. Smoke every new script; use narrow real-data checks.
New representations need their own downstream stages or an explicit posterior-only limitation.
Benchmark paired warm runs, CPU binding, memory, transfers, training quality and end-to-end scope.
Before proposing an upload, build and locally validate the concrete best package, including actual
worker-image CPU/A100 compatibility for any new binary. The user uploads; no external messages.
