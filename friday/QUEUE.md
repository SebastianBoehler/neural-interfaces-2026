# Friday Track 2 queue

Branch `codex/friday-clean-stack`. Active through Friday 20 November, 18:00 CET. The existing Track 2 Friday experiment queue heartbeat is active at 50-minute intervals in this chat. Optimize for the highest eligible sealed Track 2 rank. Evaluate complete submission candidates on held-new and returning people, independent external data and worker timing. Standalone held-new scores guide the raw-waveform lane. Prefer stronger held-person and external evidence to break alternate ties inside the development noise band.

## Wednesday 18 November, 13:39 CET

| Lane | Scheduler evidence | Next dependency |
| --- | --- | --- |
| F1/F2 waveform, Ferranti | 509981 completed 0:0 in 9 s, with both 4-step source smokes. Arrays 509982 and 509983 accepted. 27 checkpoint/receipt pairs relayed, including the two smokes. | Relay remaining pairs with `bash friday/wave/relay.sh`. Ferranti manifest SHA-256 `58e6e23425f0ee195a62916d130dd406c7a56b75aa77b3b35323846755af8fc2`. |
| F1/F2 held-person, Galvani | 3127060 completed 0:0 in 91 s. Prototype smoke 3127061 and 460 scorer 3127062 released. Centered smoke 3127064 released. Scorers 3127063/65/66 and comparisons 3127067/68 have dependencies. | Validate ten checkpoint pairs per kind/seed before releasing. The Galvani work directory holds `jobs.score.tsv`. Ownership remains with the scorer until it returns completion. |
| F-C1 cap250, Galvani | 3127070 running on its verified source snapshot. | Causal post-latch prior smoke, then eight historical order pairs and returning comparison. |
| F-C2 cap250, Galvani | Registered. | Smoke the actual CSP refit estimator and queue the matched order pairs. |
| Original §458, Galvani | 3127016–18 completed, both P04/P10 timing reports and the blind held-person score verified in the owning release. | Use [the original evidence](../../neural-interfaces-2026/evidence/next_wave_16nov/native458.md). |
| Original §459, Galvani | Real smoke 3127026 completed. External worker array 3127027 was partially complete with continuing tasks at this check. | Read the owning snapshot `ni26_external459_v2_20261118` and official collector before using external scores. |

Both original frozen archives and their SHA-256 manifest are preserved. Friday local package smoke and historical v3 replay reported OK. [Experiment ledger](../docs/LEDGER.md)

## Wednesday 18 November, 13:45 CET

- F1/F2 Ferranti finished both 20-fit arrays and all 42 checkpoint pairs were relayed. The first Galvani held smoke rejected a NumPy string stored as donor metadata under the weights-only loader. No held labels were read. Scope-limited recovery preflight 3127106, smokes 3127114/17, seed scorers 3127115/16/18/19, and contrasts 3127120/21 are in `ni26_friday_wave_20261118v2`. All checkpoints were authenticated unchanged.
- F3/F4 real EEG CPU tests passed. Ferranti 510022 ran the F3 smoke when observed. Arrays 510023/24 and F4 smoke 510025 followed it, then F4 arrays 510026/27. Actor manifest `8e3c09dc6545886a1e9b89c5030fc266484b668d9e981b490dab6fa970274af1`. Relay with `bash friday/wave/relay.sh time`. Galvani release `ni26_friday_wave_time_20261118`.
- Cap250 F-C1 finished 3127070/3127088/3127089. Held-new −0.110 pp and true-returning −0.098 pp, both intervals crossing zero. [Report](cap250-c1-result.md)
- Cap250 F-C2 smoke 3127102 accepted. F-C3 registered for clean-worker external evidence about neural refits.
- F3/F4 Galvani 3127132 preflight, four kind-specific smokes and eight seed scores 3127150–61, and matched comparisons 3127162–67 accepted. F3 smoke checkpoint receipts transferred. F-C2 smoke completed, paired array 3127133 and collector 3127134 accepted. Original §459 completed. Four-order external seed 454 −0.212 pp [80% −0.625,+0.193], seed 1454 0.000 [−0.291,+0.288] vs 48.586% cap250. [Original shared decision handoff](../../neural-interfaces-2026/evidence/next_wave_16nov/EVENING_HANDOFF_18NOV.md)

## Wednesday 18 November, 14:03 CET

- F1–F4 have final scores. All reported full-system implementations fell below cap250. [Paired waveform findings](wave/results-18nov.md)
- F5 teacher 3127357 completed, bank SHA-256 `df80f1d5cbb514e750beeeb84f953fb16d3fa4eec5aa98ad21949b11602f4435`. Ferranti smoke 510110 and six 510111 fits completed. All six production models reached Galvani and passed authentication. Exact worker smoke 3127364 completed. Production 3127365 and collector 3127366. First production task completed.
- F6 Galvani pretrain smoke 3127409 pending Priority, fit array 3127410 pending Dependency. Ferranti 510118 held for four verified pretraining pairs, 510119/120 depend on its real smoke. Galvani held scoring 3127397 and 3127399–408 accepted. Pretrain source SHA-256 `29bba6c9f91880cfb6a25df6739c9e192110a5575094f6d0e457e94b486b7ccb`, fine-tune source `e880f323149946ad98d7a89e9b59ab616e80572967aa656f00f88c5dadd80dbb`.
- F-C2 completed and remains undecided. F-C3 3127173 smoke completed, 3127268 paired A100 production underway, 3127269 collector pending. F-C4 real smoke 3127413 accepted. [Cap250 F-C3 proof](cap250-c3-queue.md)
- The Wednesday decision brief has a [working version](DECISION_18NOV.md). Refresh it and post the result by 18:00 CET.

## Wednesday 18 November, 14:17 CET

- F5 completed, exact staged stock 48.456% and both fit-excluded corrective decoder seeds lower by 3.284 and 3.975 pp on independent Scherer participants. [Exact external repeat](errorfit/external-result.md)
- F6 Galvani pretraining smoke and all four 1200-step fits completed and their manifests/encoder checkpoints reached Ferranti. Ferranti 510118 eligible, fit arrays 510119/120 await smoke. Galvani held preflight 3127397 completed.
- Sebastian selected best whole-model score on unseen people as the new exploration priority. F7 standalone raw temporal decoder tests a matched masked self-supervised training term. Ferranti 510122 smoke eligible and pending, 510123/124 follow. Ferranti actor manifest SHA-256 `1c246675c86008fe9e1aaa09d1e41824460359145871a39a65d8da634b3348e1`. Galvani 3127437 preflight completed, held 3127438–43 and matched compares 3127444–47 queued.
- F8 tests synthetic source-session covariance against a variance-matched gain control. Ferranti 510134 smoke eligible, 510135/136 follow. Manifest `93f2dcaf3f64badc6735a40c175a777b2c11f65c1df1ac5f9a5f74c01d4b21a5`. Real local covariance, waveform, session reset, and no-op guards passed. Score release in `ni26_friday_wave_style_20261118`.
- Backfill requests for the three eligible Ferranti smoke jobs were shortened to two minutes while pending. Their scientific inputs and array limits are unchanged. Slurm forecast 14:54 CET for the F7 smoke at inspection. F6 and F8 had no start estimate.
- F-C4 passed real A100 smoke and reached production. Galvani 3127453 cap three, 3127454 afterok. F-C3 last three tasks were still live. [F-C4 receipt](cap250-c4-queue.md)

## Wednesday 18 November, 14:24 CET

- F6 and F7 Ferranti smoke 510118 and 510122 completed in 8 and 9 s. Their held smokes 3127399/7402 and 3127438/7441 also completed. Full actor tasks were queued for priority. Only each pending array's first fold reduced its time request to two minutes for backfill. Other tasks retain five minutes until first-fold elapsed is verified.
- F8 CPU preflight 3127482 completed, pinned stock parity and replay OK. Held smokes/seed scorers 3127483–88, diagonal and F7 comparisons 3127489–92 accepted. F8 Ferranti smoke forecast 15:10 CET, subject to resource availability.
- F-C3 exact comparison completed and remained undecided. [Report](cap250-c3-result.md) F-C4 smoke passed and workers 3127453 were in flight.
- The existing Friday heartbeat now includes Sebastian's whole-model objective and evidence figure updates. [Raw EEG experiments and next mechanism](RAW_MODEL.md)

## Wednesday 18 November, 14:27 CET

- Ferranti F8 smoke 510134 completed in 10 seconds. All three source smoke pairs (F6/F7/F8) completed. F6 510119/120, F7 510123/124 and F8 510135/136 are accepted and await H100 priority. F8 held smoke jobs 3127483/86 started after authenticated transfer.
- Held score and runtime figures are saved as PNG and PDF in this chat's visualization directory. The generator reads all complete seed pairs and omits unscored models with a visible message.

## Wednesday 18 November, 14:34 CET

- F8 held smokes 3127483/86 completed with exact first-window reset and active genuine/donor/blank paths. All 120 F6–F8 production fits still await Ferranti priority. Updated estimate for the first reduced F7 task was 15:12 CET, subject to availability.
- F9 learns the speed of session normalization in the raw decoder. Real local EEG checks confirmed identical baseline initialization and first-window output, causal prefix/reset parity, and active prior gradients. Ferranti smoke 510310 pending. Production 510311/12 accepted behind afterok for the smoke and both F7 arrays. Actor SHA-256 `4d601b39e4ab65dd519e1257b24829cf7ef3883c981f69834623fcc1cc7ed526`.
- Galvani F9 preflight 3127716 started. Fixed held smoke/scores 3127717–19, learned 3127720–22, learned versus fixed 3127723/24, and fixed versus independent F7 3127725/26. Relay with `bash friday/wave/relay.sh norm` and verify checkpoints before releasing. F-C4 exact worker jobs remain active.

## Wednesday 18 November, 15:24 CET

- F6–F9 all completed. Raw supervised F7 reached Galvani paired scores at 15:02:50 CET, 4.95 and 4.42 pp above F3 as whole decoders. All terminal insertions remained below stock. F9 fixed normalization reached 48.07% and 48.59%. It beat matched blank and donor on both seeds. [Paired F6–F9 results](wave/results-f6-f9-18nov.md)
- F10 isolates the F9 normalization update at the cap250 terminal. Galvani real smoke 3127918 and afterok paired jobs 3127919/20 accepted. F11 raw sensor scaling and F-C5 exact lazy covariance registered, actor smoke and dispatch in preparation.
- The updated figures include all completed seed pairs through F9. Both preserved ZIPs and the public EEG input passed hash verification. The release selection remains staged cap250 with sealed v5.1b as alternate.

## Wednesday 18 November, 15:29 CET

- F10 P04 real smoke 3127918 passed exact F9 logits and archived decisions, active isolated reads and first-window stock zero. Seed scorers 3127919/20 released.
- F11 Ferranti real smoke 510567 and afterok 20-fit arrays 510568/69 accepted. Galvani preflight 3127923 completed; held smoke 3127924 and scorers 3127925/26 depend on actor evidence. Matching actor manifest `501909d4e1623d13bb62e9f780df7d23db34f39003df3a6eba42bf2b04a12fed`.
- F-C5 Galvani matched CPU smoke 3127921 started with exact stock/selection/no-op guards.

- F11 Ferranti smoke 510567 completed in 11 s; both authenticated P01 actors reached Galvani. Source matched episode, donor, reference and initialization digests across arms.
- F-C5 3127921 completed. P01 held-new used 155 versus 190 covariance calls. Two returning samples used 21 versus 70 calls. Terminal scores and selected rows were bitwise equal. Full paired array 3127930 and collector 3127931 accepted.
- F10 completed both seeds. The isolated update changed cap250 by +0.131 and −0.031 pp with both intervals crossing zero. See verified hashes in the [ledger](../docs/LEDGER.md).

## Wednesday 18 November, 16:51 CET

- The [Wednesday decision brief](DECISION_18NOV.md) is complete and published on the campaign branch. Staged v6.6cap250 is the selection and sealed v5.1b is the alternate. Both archive SHA-256 values matched on a fresh check.
- Galvani scoring completed for [D2](wave_factorial/result-18nov.md) at 16:07, [F11](wave_input/result-18nov.md) at 16:19, [F-C7](cap250-c7-result.md) at 16:32 and [F-C8](cap250-c8-result.md) at 16:49. Each collector exited 0:0. The refreshed waveform figures include both authenticated F11 seeds.
- F-C8 found −0.020 pp for causal arbitration on new delivery orders for the same Scherer people. The effect remains undecided and did not meet the registered rule for release qualification.
- Track 2 organizers reported a slower RTX sealed worker, an increased evaluation limit and a large queue. The 4,800-second figure was reported by a participant. Galvani A100 timing remains a development measurement. [Runtime provenance](../OFFICIAL_CONTRACT.md)
- Ferranti and Galvani were idle at this handoff. F12 trainable spectral initialization and exact whole-worker qualification of the score-preserving F-C5 optimization are in preparation.

## Wednesday 18 November, 17:01 CET

- [F12 raw filter-bank initialization](wave/band-queue-18nov.md) is registered. Ferranti real smoke 510747 precedes paired production arrays 510748/49. Galvani preflight 3128300 precedes the paired held smoke 3128301. Seed scorers 3128302/03 wait for authenticated pairs. Run `bash friday/wave/band_release.sh` from this worktree to authenticate checkpoint pairs and release the appropriate holds.
- [F-C5a exact A100 worker profiling](cap250-c5a-queue.md) is registered. The first cold/warm smoke found numeric differences within stock while matching the archived first label. The revised check measures stock's cold/warm difference and still requires bitwise stock/deferred equality inside each matched phase. Both frozen archives are preserved.

## Wednesday 18 November, 17:15 CET

- [F12 active A100 release](wave/band-queue-18nov.md) passed a real paired source smoke with FP32 convolution on both arms. The pending H100 wave was canceled before it started. Galvani source arrays 3128355/56 feed the same-site held preflight/smoke and scorers 3128357–60. The first spectral production fit finished at 17:08 CET. These scheduler dependencies require no workstation relay. The historical F7/F9 references used H100.
- F-C5a smoke 3128324 matched all 151 archived labels and work counts. Recognized output score bits matched and covariance calls went from 10 to zero. Unrecognized covariance calls fell from 120 to 79 and score vectors began to differ at the first neural refit, before personalized fusion. The exact score-bit condition blocks the profiling array. Source-pinned shared-input diagnostic `5c6e4ac` is staged for dispatch after connectivity returns.
- Both Galvani and Ferranti SSH attempts timed out on port 2221. The local eduVPN connection reports disconnected. Check existing F12 job IDs before dispatching any retry. Resume the diagnostic after restoring cluster access.

## Wednesday 18 November, 17:22 CET

- Galvani SSH is restored. Ferranti responds and still requires keyboard-interactive authentication. F12 and the current cap250 diagnostic use Galvani.
- [F12 results](wave/band-result-18nov.md) are complete, with 40 successful source fits and both paired held scores at 17:16 CET. Spectral initialization scored 41.372% and 41.152% as a whole decoder. Its matched random contrasts were +0.176 and −1.847 pp. The standalone figure shows both seeds and per-person changes.
- Galvani A100 job 3128477 is running the preregistered exact shared-input diagnostic from `5c6e4ac`. The full-worker timing array depends on resolving the score-bit difference in the unrecognized route.

## Wednesday 18 November, 17:24 CET

- [F-C5a attribution diagnostic](cap250-c5a-result.md) completed at 17:23:22 CET. On identical live inputs, terminal score vectors matched bitwise in both passes and covariance calls fell from 120 to 79. The two independent stock workers diverged at neural refit age 20. The current evidence qualifies stage equivalence and leaves independent whole-worker timing unresolved.

## Wednesday 18 November, 17:28 CET

- Both Ferranti and Galvani SSH logins are verified. F12 completed on Galvani. Ferranti F12 jobs were canceled before starting during the earlier platform transfer.
- F13 is registered on the historical §323 S1C checkpoints. It bounds the contribution of unusually energetic past windows to causal normalization. The exact A100 reproduction and the matched comparison use the same archived inputs and weights.

## Wednesday 18 November, penultimate submission window

- One sealed upload is authorized by 20:00 CET. Target candidate lock is 19:35 CET. At 17:38, today's attempt was available and two attempts remained overall. The 17:39 top-ten cutoff was 0.8350952562. [Decision brief](DECISION_18NOV.md)
- [F13](fbc323/result-18nov.md) matched the original S1C archive on all 42,480 decisions. Bounded normalization gained +0.668 and +0.708 pp on held-new whole-model scores, with intervals crossing zero. F-C10 tests transfer to the six exact cap250 FBCNet members. Galvani zero qualification 3128700 precedes any changed worker.
- F-C9 exact smoke 3128678 passed at 17:47 CET. A100 array 3128702 and collector 3128703 score a causal reliability gate within complete cap250 on Scherer.
- D-RTX smoke 3128675 started the two frozen ZIPs on an RTX 2080 Ti worker and verified reset and active cap250 refits. Paired workers 3128676 and collector 3128677 continue. The development GPU tests hardware sensitivity. The official RTX model and its timing remain unverified.

## Wednesday 18 November, 18:45 CET

- F-C10, F-C9 and F-C4r completed and did not advance. The nine-person RTX proxy replay measured +0.304 pp for staged cap250 versus v5.1b, with its paired interval crossing zero. [Day review](RETROSPECTIVE_18NOV.md)
- Codabench accepted exact staged v6.6cap250, submission **1058969**, at **18:45 CET**. First observed status was Running. Today's quota is 1/1 used, total 6/7 used. Read the [signed-in submission page](https://www.codabench.org/competitions/17982/#/participate-tab) for its individual score when published.

## Wednesday 18 November, 19:14 CET

- [Causal EA preflight](ea_restream/result-18nov.md) traced the two missing historical slots to trainable-FIR FBCNet with EA and shallow EA. Trainable-FIR FBCNet reproduced the historical posteriors exactly on three source-excluded anchors. The corrected delivery changed 18 of 190 decisions on the initial member-only P01 replay. The complete-system accuracy effect remains open.
- The full trainable-FIR FBCNet reconstruction is registered and staged. Galvani assets 3129279, real eight-order smoke 3129280, source-excluded fold array 3129281 and completeness validator 3129282 form the next wave. The shallow EA reproduction remains separate. No downstream scoring follows until both slots qualify.

## Wednesday 18 November, 19:32 CET

- Sealed submission **1058969** finished at **0.8298729992** and ranked **#15** at the verified check. Runtime **2,554.9166 s**. Top-ten cutoff **0.8350952562**. One attempt remains. [Official decision brief](DECISION_18NOV.md#official-penultimate-receipt)
- The trainable-FIR FBCNet causal reconstruction completed at 19:28 with all 50 folds and 13,275 recorded posteriors bitwise exact. Its label-blind eight-delivery bank is sealed. The shallow EA reconstruction remains unresolved and gates a fully causal complete-system score. [EA reconstruction](ea_restream/result-18nov.md) · [Day review](RETROSPECTIVE_18NOV.md)

## Wednesday 18 November, evening follow-ups

- **Causal full-system validation.** The fresh Shallow EA bank finished at 19:47:42 CET. Both authenticated order means are available. The next dependency is a one-order archived zero and active mixer replay, followed by regeneration of the causal refit targets and comparison of both complete systems across orders. Returning-person validation requires its own fit exclusions. [Replay plan and hashes](causal_replay/REBUILD.md)
- **Raw-waveform decoder.** F14 finished at 20:11:46 CET. Training with causal normalization lowered held-new accuracy by 2.42 and 3.26 pp relative to original S1C when also applying bounded evaluation. Original training with bounded evaluation had the strongest scores, 59.25% and 60.26%. Use original training with both inference settings as paired controls for the next genuinely different waveform experiment. [Matched scores](fbc323_train/result-18nov.md)
- **Cap250 accuracy.** F-C11 deploys a two-view correction of low-confidence training targets inside the real cap250 refit. The two A100 smoke tasks passed. The nine-person exact external replay `3130025` and blinded collector `3130026` compare shipped cap250, geometry, and a matched control that moves the same edits to other windows. [Registration](../docs/LEDGER.md#f-c11--correct-hard-refit-targets-by-agreement-between-two-within-session-eeg-views)

## Wednesday 18 November, 20:37 CET

- **Full0 causal rebuild.** The exact historical mixer and live recognition passed. Both regenerated arms sealed 240 strict-past refit states and 24,180 targets. The 14 disagreements with older archived targets are preserved. A100 `3130981` is the archived r15c250 source/refit smoke before matched regeneration. [Causal bridge and sealed target inputs](causal_replay/bridge-result-18nov.md)
- **Returning folds.** D-EAr is registered. Verified the two source masks, eight global permutations and exact recorded means for the archived five FBCNet and six Shallow seeds. Fresh matched returning A100 smokes are in preparation. [Returning inputs](causal_replay/REBUILD.md#returning-person-inputs)
- **Hard-label correction.** F-C11 completed. Geometry gained +0.040 pp vs exact stock [80% −0.357,+0.440] and +0.423 pp vs matched edit placement [+0.066,+0.784]. The edited training targets themselves lost 10.65 pp in correctness. The current submission is preferred on this evidence. [Exact external report](cap250-c11-result.md)
- **Raw S1C.** F15 finished all 80 H100 fits. Galvani CPU smoke `3130934` passed for all 14 valid evaluation endpoints, with replay `3130935` and collector `3130936` staged. Both F14 seeds are plotted with paired people/order intervals. [F15 preregistration](../docs/LEDGER.md#f15--drop-the-whole-fixed-feature-branch-during-s1c-training) · [F14 figure source](figures/f14_plot.py)
