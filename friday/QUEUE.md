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
- **Raw S1C.** F15 finished all 80 H100 fits. Galvani CPU smoke `3130934` passed for all 14 valid evaluation endpoints, with replay `3130935` and collector `3130936` staged at this check. Both F14 seeds are plotted with paired people/order intervals. [F15 preregistration](../docs/LEDGER.md#f15--drop-the-whole-fixed-feature-branch-during-s1c-training) · [S1C figure source](figures/s1c_training_plot.py)

## Wednesday 18 November, 20:48 CET

- **Held-new full0 and remaining orders.** Paired refit smoke `3131050` reproduced both 80-window prefixes and changed one of 190 FBLight decisions using the corrected inputs. All other seven held-new orders have authenticated label-blind refit targets. Real four-family full0 refit smoke `3131127` precedes the 24-member bank. A separate archived terminal-zero check is in preparation. [Refit anchors](causal_replay/REBUILD.md#historical-refit-anchor) · [Eight-order bridge hashes](causal_replay/bridge-orders-result-18nov.md)
- **Returning folds.** Both authentic repeated fits passed. A100 `3131131` and afterok `3131132` handle the remaining 20 fits and complete banks. [Returning input scope](causal_replay/REBUILD.md#returning-person-inputs)
- **Raw S1C.** F15 completed at 20:46:45 CET. Whole-feature dropout during source training improved standalone full-feature bounded inference by +0.916 and +0.921 pp versus matched scaling, with positive paired lower limits. It scored 59.94% and 60.60% on the reused held-new cohort. Versus original training it gained +0.692 and +0.338 pp, with the second interval crossing zero. The follow-up compares the same training controls on the external Scherer cohort. [F15 matched results](fbc323_skip/result-18nov.md)

## Wednesday 18 November, 20:55 CET

- **Held-new refits.** Four-family smoke `3131127` qualified FBLight. FB raw and Sinc missed archived frozen parity, and FB with EA missed archived refit parity. Production remains deferred under the registered guard. Owning §257 job `3103245` identified the original eight CPU threads and RTX 2080 Ti for the two failing FB variants. The isolated matched-device smoke release `refit_full0_matrix_v2` uses original A100 for Sinc (`3131211_2`) and RTX 2080 Ti for FB with EA (`3131210_1`) and FB raw (`3131219_0`). FB raw's first matched attempt hit a CUDA device-unavailable error before fitting and was resubmitted off that node. Keep the terminal archived-zero smoke separate from these jobs. [Refit registration and prior failure](../docs/LEDGER.md#d-eag--complete-held-new-full0-neural-member-refit-bank)
- **Returning.** `3131132` authenticated both fresh returning banks at 20:53:03 CET. The independent returning archival-zero and live replay is in preparation. [Returning bank receipts](ea_returning/result-18nov.md)
- **Waveform.** F15's three paired contrasts are rendered from the authenticated score receipt. The exported figure is `held-s1c-feature-dropout` in this chat's visualization directory. [Figure source](figures/s1c_training_plot.py)

## Wednesday 18 November, 21:36 CET

- **Held-new complete replay.** The archived terminal and active count-prefix proofs passed 3131222 and 3131283. FB raw, FB with EA and FBLight restored archived frozen and refit parity on actual device classes. Their seed-0 banks 3131435, 3131438 and 3131437 cover all 2,655 rows each. Checkpoint reload 3131535–37 reproduced their P01 refits. Exact-node Sinc 3131432 preserved all 190 argmaxes but still exceeded the original posterior guard. Keep historical zeros separate and match on a single fresh seed-0 Sinc in both comparison arms, with the other five frozen members and fixed calibration common. Its deterministic smoke 3131531 passed. The remaining folds 3131532 and collector 3131533 are the next dependency for live incumbent, recognition and refit regeneration. [Terminal receipts](causal_replay/complete-zero-result-18nov.md) · [Sinc provenance](causal_replay/sinc-provenance-18nov.md)
- **Returning.** The full0 historical route replicated bit for bit. The two new banks changed seven terminal historical decisions, with identical owners and recognition ages. The exact packaged-style FBLight refit produced eight outputs before recognition on P10 visit 2. All 20 confidence weights changed and the eight predicted classes agreed. Authentic FB raw and FB with EA preflight remains open. [Live returning bridge](ea_returning/bridge-result-18nov.md) · [Returning refit](ea_returning/fbl-refit-result-18nov.md)
- **Waveform.** F15e completed on nine fit-excluded Scherer participants. Whole-feature dropout versus original training measured +0.142 and −0.218 pp. Both paired intervals cross zero. F16 on held-out PROTEUS people changed −0.336 and +0.970 pp versus raw-only hard CE. Trial-specific targets remain undecided against session-class derangement. F16e will test the raw-only controls on the same external cohort. [F15 external evidence](fbc323_skip_external/result-18nov.md) · [F15 transfer figure source](figures/f15_transfer.py) · [F16 whole-decoder results](fbc323_distill/result-18nov.md)

## Wednesday 18 November, 21:39 CET

- **Fresh matched full0 source.** The deterministic Sinc collector `3131533` completed at 21:37:10 CET. It authenticated ten fits and 2,655 frozen rows per order. Job `3131547` then verified both complete incumbent shadows at 21:38:50 CET with the same fresh Sinc seed 0. The original fixed recording calibration and both live routes held, with 2,515 finite scores and 0/14 recognized sessions per arm. Both sealed target banks contain 24,180 strictly earlier refit rows. [Replay plan and hashes](causal_replay/REBUILD.md)
- **Active refits.** Real P01 smokes `3131561/63/65/67` gate the four complete actor banks `3131562/64/66/68`. Each uses its own arm's newly generated targets on the same restored source fit. CPU terminal `3131569` depends on all four banks. Gold scorer `3131570` depends on the complete terminal seal.
- **Raw-only transfer.** F16e source smoke and both H100 triples completed. Galvani smoke `3131549`, nine-person held replay `3131550`, and collector `3131551` are staged behind the source relay. [F16e registration](../docs/LEDGER.md)

## Wednesday 18 November, 21:48 CET

- **Corrected held-new full0.** Jobs `3131569/70` completed. With shared fresh Sinc seed 0, corrected cap250 reached 73.550% versus 73.072% for the incumbent. The +0.479 pp difference has an 80% interval of −0.222 to +1.452 pp. Correcting delivery moved the advantage −0.222 pp versus the matched historical-input arm, with uncertainty crossing zero. Job `3131595` sealed fresh frozen Sinc across all eight orders. The next order preflight checks the other three source actors against their exact within0 anchors. [Complete-system result](causal_replay/complete-sinc-result-18nov.md)
- **Waveform transfer.** Trial-specific F16e targets did not consistently improve external accuracy. The exact same-weight Scherer ablation `3131607` found +2.70 to +3.52 pp from zeroing the fixed features under bounded normalization, across both trained models and both seeds. The same inference change lowered point scores on the PROTEUS holdouts. [External feature ablation](fbc323_external_q/result-18nov.md)
- **Search audit.** The next exact-worker pilot tests clipping extreme covariance directions during live alignment against stock and a scalar control with the same removed variance. The current sealed package remains the reference. [Method and coverage audit](SOLUTION_SPACE_18NOV.md)

## Wednesday 18 November, 22:34 CET

- **Native raw slot.** F-C13 completed at 22:32 CET. One raw-only S1C replacement gained +0.040 pp [80% −0.321,+0.370] on exact-worker Scherer. The replication gained +0.023 pp. Both lost on order 411 and gained on order 1411. Stock cap250 remains the submission reference. [Full results and timing](cap250_raw_swap/result-18nov.md)
- **Causal alignment.** F-C12 passed its one-session label-blind A100 smoke. The directional and matched scalar arms changed the same one final decision, with no accuracy test. D-EAw0 matched the P01 frozen source to tolerance for the other three refit families. The remaining matched within0 source folds and the returning-person refits are the next validation work. [F-C12](cap250c12/result-18nov.md) · [Within0 qualification](causal_replay/within0-provenance-18nov.md)

## Wednesday 18 November, 22:41 CET

- **Held-new within0.** RTX P04 smoke 3132064 passed across three visits. FB raw and FB with alignment matched the archive bit for bit. FBLight retained every label with maximum probability difference 1.188e−6. Eight remaining folds are running in array 3132093, with 3132101 after full authentication. This validates frozen actors before the dependent refits. [Registration](../docs/LEDGER.md)
- **Waveform F17.** Ferranti smoke 511174 passed with the exact hard-CE reference. Shared-frequency phase perturbation and a rigid temporal shift had matched waveform displacement and preserved cross-channel spectra. Galvani 3132104 checks independent replay. Ferranti production 511175 and Galvani held replay 3132105 wait for that result. Collector 3132106 depends on complete verified folds. [Registration](../docs/LEDGER.md)
- **Returning-person stages.** Galvani archive smoke 3132092 found a gap in a recognition ramp after the first positive value. The actual routing decision stays latched while the personal head and prior use the per-window ramp. Validate those two quantities separately before the matched tail replay. [Registration](../docs/LEDGER.md)

## Wednesday 18 November, 23:24 CET

- **Held-new within0.** All ten folds passed frozen qualification across 2,655 held rows and 14 visits. FB raw and FB with EA matched exactly. FBLight preserved every decision within 1.188e−6. Fresh Sinc matched its own same-state bank. The next comparison must verify the original within0 warm refits before regenerating both arms. [Frozen qualification](causal_replay/within0-provenance-18nov.md)
- **Returning.** D-EArT authenticated both actual gates, the routed tail and the source-fold priors on 762 later-visit windows. Legacy personal-head and TSEM views differ. A native RouteB reconstruction must prove which coordinates it consumes, and P10 visit 2 still has two missing warm refits. [Returning scope](ea_returning/tail/result-18nov.md)
- **Waveform.** F17 completed all 60 source fits, 20 held banks and scoring. The dispersive phase effect was undecided against matched rigid jitter at both seeds. The chart shows the comparison using the pinned scorer receipt. [F17 result](fbc323_phase/result-18nov.md) · [Figure source](figures/s1c_training_plot.py)

## Wednesday 18 November, 23:31 CET

| Lane | Scheduler evidence | Dependency |
|---|---|---|
| Within0 warm refits | Galvani RTX smoke 3132542 on P01 with the original §257 targets and observed AdamW steps. | Check archived first-20 and warm decisions for FB raw, FB EA and FBLight before the full array. The fresh Sinc bridge has separate provenance. |
| Returning native RouteB | Galvani 3132550 runs the actual ZIP stage on P04 visit 2. | Match split-view control to the legacy tail and regenerate source-fold native coordinates before the remaining visits. Only P10 visit 2 has the missing upstream warm path. |
| F18 spatial sensor withholding | Ferranti 511200 is resource-pending. Source array 511201 remains held. The label-blind real precheck matched local and dispersed masks on removed energy and anchor. | Authenticate Ferranti smoke, then Galvani 3132545. Keep both full arrays behind those smokes. Scorer 3132547 depends on all 20 held folds. |

### Overnight raw S1C factorial F19

- F18 completed at 23:44:37 CET. Energy-matched local sensor withholding remains undecided against both hard CE and dispersed withholding. [F18 results](fbc323_sensor/result-18nov.md)
- F19 Ferranti smoke `511265` passed in 10 seconds at 23:44:18 CET and exactly recovered both flat F17 states. Galvani smoke `3132601` passed its four-arm causal replay in 55 seconds at 23:47:13 CET. The relay released H100 array `511266` after both passed. Independent replay `3132602` receives completed folds. Galvani `3132622` checks actual waveform and head parameter changes against the exact source initialization. Collector `3132603` waits for all twenty blind banks and the weight proof before gold. [Registration](../docs/LEDGER.md#f19--jointly-learned-raw-s1c-hierarchy-crossed-with-matched-jitter)
- The running local relay is `bash friday/fbc323_hierarchy/run_dependencies.sh 511265 511266 3132601 3132602`. Remote source and held releases use `friday_s1c_hierarchy_20261118` and `ni26_friday_s1c_hierarchy_20261118`. The relay verifies both smoke completions before releasing source work. Once the source array is released, `bash friday/fbc323_hierarchy/relay.sh 511266 3132602` copies completed folds and safely releases their corresponding held tasks.

## Wednesday 18 November, overnight top-three campaign

[Target and decision evidence](OVERNIGHT_TOP3.md).

| Lane | Current evidence and queued dependencies |
|---|---|
| Full cap250 four-arm combination | F-C14 A100 smoke 3132597, nine-person 3132604 afterok smoke, scorer 3132605 afterok all banks. Fresh Scherer deliveries on the reused nine people. |
| Whole raw decoder | F19 source 511265 and independent held smoke 3132601 passed. Ferranti 511266, Galvani 3132602 and collector 3132603 use authenticated per-fold relay. See the F19 commands above. |
| Held-new within0 | Raw and FB EA historical warm controls passed exactly. Two FBLight visits exceeded numerical tolerance with identical decisions. Same-original RTX recovery 3132595 and independent terminal zero 3132606 gate fresh paired production. |
| Returning | Fresh source-excluded raw/Sinc warm consumers passed 3132599 with both target arms. Native gate and mixture still need common-fit regeneration before full-system comparison. [Receipts](ea_returning/fresh-warm-result-18nov.md) |
- F19 completed at 23:59:08 CET. All four arms and exact F17 controls authenticated. Adding hierarchy to jitter lost 1.108/0.628 pp in held-new whole-decoder balanced accuracy with paired upper bounds below zero for both seeds. [F19 results and final receipts](fbc323_hierarchy/result-18nov.md)

### Overnight raw S1C interpolation F20

- Ferranti source smoke `511294` passed at 00:19:27 CET on 19 November. It exercised both gradients and exactly 2,528 of 10,112 mixed appearances on the same donor row identities. Source matrix `511295` is held until the independent Galvani replay smoke `3132701` passes. Held matrix `3132702` and collector `3132703` require the complete blind banks and source-only donor checks. [Registration](../docs/LEDGER.md#f20--same-task-pooled-waveform-interpolation-across-source-people)
- Relay controller `bash friday/fbc323_latent_mix/run_dependencies.sh 511294 511295 3132701 3132702` is active. It verifies both smokes and relays each completed source fold and its donor tape. After the source matrix is released, `bash friday/fbc323_latent_mix/relay.sh 511295 3132702` can resume a partial production transfer. The remote releases are `friday_s1c_latent_mix_20261119` and `ni26_friday_s1c_latent_mix_20261119`.

## Thursday 19 November, 00:20 CET

- **Cap250 combination.** All nine exact F-C14 banks and scorer 3132605 passed. The combination gained +0.146 pp [80% −0.374,+0.698] versus stock. Cognitive-only gained +0.291 pp, and the combined result trailed it by 0.146 pp. All three interventions reversed sign across the two fresh deliveries. A100 timing passed. [F-C14 result](cap250c14/result-19nov.md) · [Paired figure source](figures/cap250_factorial.py)
- **Raw decoder.** F19 refuted the hierarchical head added to matched jitter on both seeds. F20 tests a different source objective using same-task representation pairs, with an exact hard-CE arm and a donor matched within the source session. Real Ferranti smoke 511294 passed. Galvani 3132701 gates Ferranti 511295 and held replay 3132702 before scorer 3132703. [F19 results](fbc323_hierarchy/result-18nov.md) · [F19 figure source](figures/s1c_training_plot.py)
- **Validation.** The within0 archive zero passed. The four matched family replays, terminal bank and scorer are queued behind the two original-node FBLight recovery tasks. Returning native mix and recognition qualification uses the fit-specific §321 coordinate context and preserves its conditional auxiliary scope. [Within0 dependencies](causal_replay/within0-overnight-queue.md)

## Thursday 19 November, 00:25 CET

- **F20.** Both real smokes passed. Ferranti 511295 is running, and authenticated completed folds are reaching Galvani 3132702. Final scorer 3132703 waits for all twenty sealed banks. [Authenticated relay](fbc323_latent_mix/run_dependencies.sh)
- **Returning native path.** Smoke 3132709 stopped at the normalized member-weight check. Corrected smoke 3132715 and dependent four-visit replay 3132716 use the pinned native normalized weights. This measures real gate and warm target generation in the conditional returning package. [D-EArN registration](../docs/LEDGER.md#d-earn--native-returning-mix-routing-and-strict-prefix-warm-labels)
- **Exact outer rebalancer F-C15.** A100 smoke 3132722 is running. It tests a robust past-confidence estimate against a matched-strength scalar control in the deployed tail. The inner gate and fine-tune targets keep their original estimator. The source and independent fresh order comparison are registered. [F-C15 registration](../docs/LEDGER.md#f-c15--bound-the-past-class-estimate-in-the-actual-outer-rebalancer)
- F20 completed at 00:30:49 CET on 19 November. Both registered cross-person contrasts remain undecided. All matched donor tapes, source fits and held banks authenticated. [Result and final receipts](fbc323_latent_mix/result-19nov.md)

## Thursday 19 November, 01:04 CET

- **Exact stock comparison.** F-C15 scorer 3132753 sealed all nine people and refuted the robust outer rebalancer on the two new orders. The change was −0.493 pp [paired 80% −0.949, −0.066] against stock and −0.420 pp [−0.810, −0.050] against the norm-matched scalar. Both passed A100 runtime checks. [F-C15 evidence](cap250c15/result-19nov.md) · [Paired plot source](figures/cap250_factorial.py)
- **Raw decoder.** F20 cross-person interpolation changed −0.379/+0.343 pp over hard CE and +0.118/−0.371 pp over within-session interpolation. Both planned contrasts remain undecided across seeds. [F20 evidence](fbc323_latent_mix/result-19nov.md) · [Paired plot source](figures/s1c_training_plot.py)
- **Returning.** Native returning controls matched all 762 §321 posteriors and decisions. Recording versus causal EA changed four final decisions. Production-derived auxiliary fits and calibration exposure remain, and gold stayed closed. [D-EArN evidence](ea_returning/native_inner/README.md)

### Overnight reconstruction consistency F21

- The two geometry-only maps preserve the 19 measured common sites and reconstruct 22 PROTEUS channels. Map SHA-256 `b4128c37dacfef9ddb38d960b314f21ee98715b57c1120ddb60e98d9a80f4da4`. Ferranti source retry `511320` passed at 01:11:30 CET with exact hard control and the separate KL callback guard. Galvani smoke `3132879` gates source array `511321`; independent production `3132880` and collector `3132881` require the complete blind matrix. [Registration and retry](../docs/LEDGER.md#f21--source-electrode-reconstruction-consistency-for-raw-s1c)
- Active controller `bash friday/fbc323_reconstruction/run_dependencies.sh 511320 511321 3132879 3132880`. After source release, `bash friday/fbc323_reconstruction/relay.sh 511321 3132880` safely resumes incomplete relays. Immutable source and held release names end in `s1c_reconstruction_v2_20261119`.

## Thursday 19 November, 01:14 CET

- **Returning source coverage.** P10 source-fold Hilbert Core2 fit 3132875 completed with the verified source exclusion and owning recipe. Independent reference retry 3132912 gates real native run 3132914. This source refit addresses one production-derived layer. Rest, Eye, event and calibration scopes remain separate. [Registration](../docs/LEDGER.md)
- **Returning prototype F-C16.** Authenticated first-visit source feature and fit job 3132877 completed. P10 visit 2 smoke 3132907 gates four native visits 3132916 and the conditional scorer 3132917. The registered prototype is compared with the actual native personal head and a class-count-matched donor head using the same feature extractor and head weight. [Registration](../docs/LEDGER.md)
- **F21 waveform reconstruction.** Corrected Ferranti source smoke 511320 and independent Galvani 3132879 passed. Ferranti 511321 is training source folds. Galvani held 3132880 and scorer 3132881 are bound to the authenticated relay. [Relay](fbc323_reconstruction/run_dependencies.sh)
- F21 completed at 01:22:43 CET on 19 November. Both held-new contrasts remain undecided. The source-only reconstruction disagreement decreased by 14.94% and 14.65% versus alternating CE. [Final results and receipts](fbc323_reconstruction/result-19nov.md)

## Thursday 19 November, 01:56 CET

- **F21 and F-C16 completed.** F21 reduced source reconstruction disagreement by about 15% on both seeds. All held-new accuracy intervals crossed zero. F-C16's conditional first-visit head gained +1.588 pp on P04 and lost −0.521 pp on P10 versus the native head. Its prespecified weighted effect was −0.099 pp. [F21 results](fbc323_reconstruction/result-19nov.md) · [F-C16 results](cap250c16/result-19nov.md) · [F-C16 plot source](figures/returning_head.py)
- **Held-new within0.** Both original-node RTX FBLight recoveries reproduced the archived labels and exceeded the 2e−5 warm posterior tolerance on the same visits as before. The recovery-dependent scoring chain canceled before reading held labels. The successor uses both fresh Sinc and common fit-excluded FBLight source states, and must regenerate warm targets and refits in both arms. [Historical queue and gate](causal_replay/within0-overnight-queue.md)
- **Returning source scope.** The source-only P10 Hilbert Core2 fit changed the native inner mixture at its actual 1/15 weight and passed exact original controls on the 31-window smoke. All 31 terminal decisions agreed. Rest, Eye, event and historical calibration exposure remain. [Native Core2 proof](ea_returning/native_aux/README.md)

### Overnight temporal representation F22

- Ferranti real P01 source smoke `511347` passed after exact hard-state isolation. H100 source array `511348` waits for independent real held smoke `3133120` on Galvani. Galvani held matrix `3133121` and collector `3133122` require authenticated matched checkpoints and the complete blind bank.
- Source actor SHA-256 `8a7d1784adbd88fd7381e9f3628a613fa1b7768c4732ca2551ae37cb9687f2c1`. Held scorer SHA-256 `979f5a2152b6e521fbeaac6108978817be6ae8f0ddddc4af4629855291b949bd`. Immutable release directories end in `s1c_temporal_v3_20261119`. The active controller is `bash friday/fbc323_temporal/run_dependencies.sh 511347 511348 3133120 3133121`. After source release, `bash friday/fbc323_temporal/relay.sh 511348 3133121` resumes unfinished relays. [Preregistered F22 protocol](fbc323_temporal/prereg-19nov.md)

## Thursday 19 November, 02:06 CET

- **F-C17 two EA histories.** A100 smoke 3133104 passed exact stock/disabled labels and work, strict-prefix checks and the exposure guard. The informative arm used fractional past covariance histories, with a block-permuted matching control. The four first production tasks in 3133114 are running. Scorer 3133115 waits for all nine people on orders 6411 and 7411. [Registration](../docs/LEDGER.md)
- **F22 within-window time.** Corrected Ferranti smoke 511347 reproduced the original hard-CE control and verified genuine six-sample pair features against scrambled pairs. Independent Galvani smoke 3133120 passed. Source 511348 releases through the authenticated relay to 3133121, then scorer 3133122. [Registration](fbc323_temporal/prereg-19nov.md)
- **Shared-source within0 successor.** Real P01 FBLight bank smoke 3133100 passed. RTX array 3133101 and collector 3133102 use the same eligible saved fold assets. Slurm forecast the remaining RTX starts at 04:24–04:41 CET at 02:05. The fresh Sinc/FBLight targets and four warm refits depend on the verified common bank. [Registration](../docs/LEDGER.md)
- Independent held smoke `3133120` passed in 45 seconds. Ferranti full array started at 02:06:16 CET. After the first two 35-second source tasks passed, only the pending request was shortened to two minutes to permit backfill. The matched source recipe and releases are unchanged.
- F22 completed at 02:20:54 CET on 19 November. Both paired contrasts remain undecided. All source and held tasks passed, and the exact F16 hard checkpoint and decision controls passed. [Sealed results and receipts](fbc323_temporal/result-19nov.md)

## Thursday 19 November, 02:44 CET

- **Held-new within0.** D-EAwcf completed the blind bank and independent official metric. Corrected cap250 78.324% vs P0r 75.779%, +2.546 pp [paired 80% +1.641,+3.406]. Causal alignment changed cap250's advantage by −0.374 pp [−0.714,+0.050]. One order, ten reused fit-excluded people, and common Sinc and FBLight seed-0 states. [Result](causal_replay/within0-overnight-queue.md) · [Figure source](figures/within0_alignment.py)
- **Waveform and external.** F22 physical lag versus exact hard CE was −0.001/+0.561 pp. Both preregistered two-seed endpoints remain undecided. F-C17 exact whole-system alignment scored −0.069 pp against stock and against its permuted control. It passed A100 timing. [F22 result](fbc323_temporal/result-19nov.md) · [F-C17 result](cap250c17/result-19nov.md)
- **Returning source context.** Three fit-excluded P10 Rest fits entered the genuine conditional native path. The terminal posterior moved on the 31-row smoke with all decisions and warm work retained. Production Eye and calibration scope remain. [Rest proof](ea_returning/native_rest/README.md)

### Overnight lag by jitter factorial F23

- Ferranti real source smoke `511387` gates held four-fit array `511388`. Independent Galvani smoke `3133377` gates source release and held replays `3133378`, then collector `3133379`. The three exact parent hashes and individual held decisions must match on every fold. [Preregistered F23 protocol](fbc323_lag_jitter/prereg-19nov.md)
- Actor SHA-256 `0ae228bed1dc7dd9d653dbe66ac51ca5d5a486e0054cd95e98eb3ecc431d276d`. Scorer SHA-256 `6e8d6a1b69ec22dcbb3db51c1a1206424060b05d7b7883dd9b57e0e8bfb68910`. The active controller is `bash friday/fbc323_lag_jitter/run_dependencies.sh 511387 511388 3133377 3133378`. After source release, `bash friday/fbc323_lag_jitter/relay.sh 511388 3133378` resumes fold transfers to the immutable `s1c_lag_jitter_20261119` release.
- Both real smokes passed. Ferranti 511387 matched all three parent checkpoint bytes. Galvani 3133377 independently passed in 54 seconds. The released H100 array waits for resources with an initial Slurm estimate of 04:42:18 CET. Pending wall requests were shortened to two minutes without changing the fit. The active relay uses `bash friday/fbc323_lag_jitter/run_dependencies.sh 511387 511388 3133377 3133378 resume` with a six-hour bound. Bank authentication and source IDs are unchanged.

## Thursday 19 November, 02:54 CET

- **Shared-source runs0.** P01 frozen smoke 3133373 passed. Nine-fold 3133374 and seal 3133375 gate the live route and target job 3133380. Independent exact zero 3133395 uses the owning 9 Eye / 3 Rest archived inventory. The within0 archived zero used 9/9. Both matched comparisons use 3 Eye / 3 Rest. Matched warm branches 3133397–3133404 gate the terminal bank 3133405 and separate scorer 3133406. [Registration](../docs/LEDGER.md)
- **Within0 stage necessity F-C18.** The real P01 smoke reproduced exact cap250 and proved the two stage bypasses and strict-past control. CPU job 3133393 runs the complete four-arm factorial, with the locked independent score in 3133394. [Registration](../docs/LEDGER.md)
- **Raw lag plus jitter F23.** Ferranti 511387 and independent held smoke 3133377 passed. All three exact parent states replayed. Source 511388 is released, with its first H100 slot estimated at 04:42 CET at this check. Held 3133378 and scorer 3133379 stay behind the authenticated relay. [Relay](fbc323_lag_jitter/run_dependencies.sh)
- Runs0 exact zero 3133395 passed all original and matched-anchor inventories, plus three interleaved count/CSP prefixes. The shared-bank and live-target graph remains gated.
- F-C18's first complete job reproduced its stage/work checks and stopped before sealing on the P0r output key. Corrected immutable chain 3133409 → 3133410 → 3133411 uses the same four arms and scoring rule.

- F23 completed at 03:27:01 CET. Slurm backfilled the H100 jobs before the first estimate. All source, held and scoring jobs passed. All three preregistered effects remain undecided. [Results and receipts](fbc323_lag_jitter/result-19nov.md)

### Overnight source calibration robustness F24

- Ferranti real P01 smoke `511409` and held production array `511410`. Galvani independent held smoke `3133772`, 20-fold replay `3133773`, collector `3133774`. The source array releases only after both real smokes. Initial Slurm estimate for the smoke was 04:58:35 CET. Its pending wall request is two minutes, with the full exact H100 source array unchanged. Estimate roughly 6–20 minutes after authentic source array start, depending on how many H100 slots open. [Registered F24 protocol](fbc323_calibration/prereg-19nov.md)
- Source actor SHA-256 `5c991bc11ffc3f92c09a4f2ca1cfa7b73199d5366c8dc45432ebb85830006b31`. Scorer SHA-256 `fa0ef9c9d8910eca5bcd3e097428bbb092d3a6ee125dd620ec28fa6e5d5b68f9`. Immutable release names end in `s1c_calibration_20261119`. Controller `bash friday/fbc323_calibration/run_dependencies.sh 511409 511410 3133772 3133773`. After source release, the same command with `resume` safely resumes incomplete relays.
- Galvani CPU preflight `3133783` passed an actual P01-excluded 64-row source batch without labels or fitting. Native readout reconstruction was bitwise exact. Both auxiliaries had zero donor gradients and active query gradients. The source tape selected 121 of 160 smoke batches. The H100 and independent held production gates remain in force.
- F24 v1 source `511409` stopped before fitting because the initialization guard counted frozen spectral filters as trainable. Source `511410` and held `3133772–3133774` are canceled. Immutable `s1c_calibration_v2_20261119` restores the ancestral trainable-only count, preserving exact fingerprints and the full recipe. Its H100 smoke `511412` passed in 12 seconds with CUDA peaks of 1.57 GB allocated and 1.95 GB reserved. Array `511413` is gated by independent `3133805`, then held array `3133806` and collector `3133807`. Controller `bash friday/fbc323_calibration/run_dependencies.sh 511412 511413 3133805 3133806`. Source and scorer manifests `107d380fe6cb5e55532c3194986d7f0a5eb0705f48666fc8383fbedabfbaa3d2` and `0acf64a3303718a1dcfe874e962e0cb36cbe71b4db22794d0b4cbc1df2e935cf`.
- F24 v2's real held smoke `3133805` passed in 54 seconds, releasing source array `511413` at 04:26:37 CET. The first full task took 46 seconds and the next finished within 47 seconds. Peak reserved CUDA memory was 1.96 GB. Only pending time requests were shortened to two minutes. The source recipe and allocations otherwise remain unchanged. At one active H100 the estimated complete held score is around 04:44–04:48 CET.

### RTX covariance parity and timings F-C5r

- Galvani RTX 2080 Ti real smoke `3133816` began at 04:29:06 CET. Paired warm timing array `3133817` is gated on shared-input score-bit identity, fixed routes, and actual covariance work. Release `ni26_friday_c5r_rtx_v1_20261119`, source SHA-256 `cfdbd89da2f1a5a87abe11074f78ac66032d0bc659073f222a83e9bbe5aae2d7`. [Registered scope](../docs/LEDGER.md#f-c5r--same-input-personalizer-parity-and-paired-rtx-worker-timing)
- F-C5r smoke `3133816` passed at 04:31:20 CET. All 151 terminal vectors were bitwise equal under identical live inputs. Covariance calls fell 10→0 recognized and 120→85 unrecognized. On this one RTX proxy stream the unrecognized stage took 2.534 s eager and 1.832 s deferred, with matched 200 warm optimizer steps and 30 fusions. Whole-worker timing jobs started at 04:31:35 CET. Smoke SHA-256 `4da9b1b595175a7134c28b23fc42d6642d5666461f740abb9d644b20e1643d32`.
- F-C5r timing array completed by 04:36:49 CET. Independent numerical parity passed. Deferred covariance lowered unrecognized whole-worker wall time 1.80% and 0.64%, with p95 higher in both runs. [Result](cap250c5r/result-19nov.md)

- F24 completed at 04:36:40 CET. All held folds and the collector passed. The 19 normal source jobs plus the authenticated completed seed-1 P07 payload from the Slurm TIMEOUT task supplied all 80 source fits. [Results, exact controls and timeout disposition](fbc323_calibration/result-19nov.md)
