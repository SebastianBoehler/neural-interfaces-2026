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

### Overnight multiresolution waveform modulation F25

- Ferranti P01 smoke `511434` passed on H100 in 11 seconds, with 0.856 GB CUDA peak reserved. Source array `511435` waits for Galvani real held smoke `3133882`, followed by independent matched folds `3133883` and collector `3133884`. [Registered F25 protocol](fbc323_modulation/prereg-19nov.md)
- Source actor SHA-256 `589b9a212cafdb0367f9f34bb840db5ff95c98ccfeea3bf77e268aa3ddb2e95c`. Held scorer SHA-256 `939a96514d432cedebf24a0c4500927f71401ed76117adc73fc76604453a779c`. Controller `bash friday/fbc323_modulation/run_dependencies.sh 511434 511435 3133882 3133883`. After source release, the same command with `resume` continues the checkpoint relay into the immutable `s1c_modulation_20261119` release.
- F25 independent held smoke `3133882` passed in 46 seconds. Source array `511435` released and started two fits at 04:41:08 CET. Both learned arms had 351,935 trainable parameters, alongside 347,903 for the exact F16 anchor. Warm 32-window H100 forward medians were 1.735 ms hard, 2.119 ms conventional and 2.110 ms multiscale. Estimate full scoring around 04:49–04:54 CET if two H100 slots stay available.

- F25 completed at 04:50:20 CET. All source and held jobs passed. The two bounded preregistered comparisons are undecided across seeds, with positive paired intervals on seed 1. [Results and receipts](fbc323_modulation/result-19nov.md)

### F25c frozen multiresolution substitution on corrected cap250

- Both real pupil smoke runs `3133948_0/1` passed. Pupil array `3133949` independently recomputes all hard, same-capacity conventional and multiresolution bounded posteriors, with exact original hard-probability anchors and held-label separation. Corrected full0 smoke `3133971`, full0 `3133972`, within0 `3133973` and separate scorer `3133974` follow in order. [Registered protocol](cap250_f25/prereg-19nov.md)
- Immutable Galvani release `ni26_friday_causal_rebuild_20261118/f25c_v1`. Pupil manifest `b15e7155853a7694387deb618a4786f095b6fc26bbe282aec2129ec2ef68dcb5`, consumer manifest `2f928083d9163a9e9530a746eada4438fc134f76b2f5122139c02fb37894caa7`, scoring manifest `094fdadfddbe1c5705e7686c84f00e46bd6c782ceb40c32e715f2bcb0e003301`. This is the corrected historical full-system instrument. The separately running F-C20e native external and returning lanes are independent.

### Physical common-average reference F26

- Ferranti H100 smoke `511455` passed at 05:22:31 CET. Hard retained exact F16 bytes. CAR and the norm-matched artificial direction each ran on all 160 fit steps. The fixed normalized common-offset probe moved CAR by 4.77e−7 and the artificial-direction control by 0.271. Relative energy mismatch between the two mapped waveforms was at most 1.19e−7. This verifies signal geometry before scoring.
- Galvani real held smoke `3134002` started at 05:25:06 CET. Ferranti source array `511456` waits for both smokes, with held replays `3134003` and collector `3134004` behind authenticated dependencies. Actor `b93711fe591771ffdd51dff50f4b07a7a21d3e1500635ebdfaecfabde818c2d2`, scorer `fdc72508bb8439512b015698da05e5c8d435fd14310ccfddfb87a57a3b80a023`. The controller is `bash friday/fbc323_reference/run_dependencies.sh 511455 511456 3134002 3134003`. [Preregistered test](fbc323_reference/prereg-19nov.md)

- F25c completed at 05:29:38 CET. Multiresolution beat matched conventional at equal mass in both corrected deliveries. Its stock and hard comparisons did not satisfy the two-order native gate. [Signed full-system results](cap250_f25/result-19nov.md)

### Overnight nonlinear electrode representation F27

- Ferranti real H100 source smoke `511476` passed in 18 seconds. Source array `511477` waits for Galvani `3134042` independent held smoke, then held folds `3134043` and collector `3134044`. The independent relay is `bash friday/fbc323_electrodes/run_dependencies.sh 511476 511477 3134042 3134043`. Both source and held waves are capped at four concurrent folds. [F27 protocol and thresholds](fbc323_electrodes/prereg-19nov.md)
- Inspected H100 32-window times were 1.825 ms hard, 2.403 ms physical and 2.388 ms scrambled with 0.856 GB peak reserved. All 1,476 coefficients activated under both topologies. F16 hard bytes match exactly. Release names end in `s1c_electrodes_20261119`. Source actor and held scorer manifests `4629f3dfa3b0d4d443de6599b7b2812e6e76cb0a479a60a0a15220c2cfae1332` and `dc0645c6d9c8f51245899943bef712c72f616fbcb2110b224100751a0aef9f10`.

- F26 completed at 05:33:58 CET. Both authentic smokes, all twenty source and held folds, and collector `3134004` passed. All registered bounded endpoints remain undecided. [F26 score and receipts](fbc323_reference/result-19nov.md)
- The independent Galvani hold `3134042` passed in 1m16. Ferranti `511477` released, with two tasks starting at 05:34:58 CET. The full source wall requests remain unchanged. If two H100 slots hold, the estimated score is 05:48–05:52 CET.

- F27 completed at 05:48:55 CET. Every source and held job passed, and both registered contrasts remained undecided. [Signed results](fbc323_electrodes/result-19nov.md)

### Adapted FBLight and multiresolution cross F-C22

- Galvani complete-system smoke `3134261` gates full0 `3134262`, within0 `3134263` and separate score `3134264`. It reuses the signed FBL2 and multiresolution probability banks and checks both parents bitwise before opening labels. The release is `ni26_friday_c22_cross_v1_20261119`, source `90e2a71bb75538a8598f61785df18722ec02861f6fd9f562fac4a43c931e9a3c`, scorer lock `18c37643313077fd0d34d162bca15d575e76a164401d136475fd5eeb943d7d91`. [Protocol](../docs/LEDGER.md#f-c22--cross-adapted-fblight-doubling-with-multiresolution-raw-substitution)

### F28 teacher-excluded waveform relations

- Ferranti `511498` passed all three genuine P01 nested teacher fits in 24 seconds. Source student smoke `511500` passed in ten seconds with 1,271 same-task source pairs. Real independent Galvani smoke `3134277` gates source teacher array `511499`, which in turn gates source students `511501`. Held array `3134278` and collector `3134279` only consume the authenticated teacher receipts and frozen native student checkpoints. Both source waves cap at four and are sequential. [Registered protocol](fbc323_relational/prereg-19nov.md)
- Teacher actor SHA-256 `467fd886eb03bc1b27ea28e42b199c89eb9d314630f584adf71b6e2db232354f`, student actor `8ab663afcf72b5a8e6668e379f1e00fda27effe983beafba48cf84e2373298c3`, held scorer `a001fba2661127171cac167fd2475f0b67d72864eb2b80a749c92284d9981a56`. Releases end in `s1c_relational_20261119`. Controller `bash friday/fbc323_relational/run_dependencies.sh 511500 511499 511501 3134277 3134278`. After teacher release the same command with `resume` continues incomplete student relays.
- F28 independent held smoke `3134277` passed at 06:13 CET in 37 seconds. The first source teachers completed in 14–19 seconds. Two H100 slots were active at 06:14. Full scoring is estimated around 06:30–06:40 CET at the present capacity; full source budgets remain in force.

- F-C22 completed at 06:21:19 CET. Both corrected orders matched the full parent banks bitwise. The joint beat exact cap250 by +0.538/+0.778 pp. It remained undecided against FBL2 and the matched conventional joint. [Signed results](cap250c22/result-19nov.md)

### Direct FBLight trial package F-C21p

- Built experimental archive `35bd2d46e129f5c29eef98ace428c92aa95693453507bf4d5fff4436c1ad688e` from exact frozen cap250. The CPU smoke matched all 62 predictions, protected and terminal posteriors, recognition and optimizer work bitwise. No gold was opened.
- Galvani RTX smoke `3134307` runs the actual archive on the two recognized and unrecognized fixtures. Immutable release `ni26_friday_c21p_package_v1_20261119`, input/source manifest `f0114de7bf6f9a8bb2588799b546a896bc52301d2cfad1e8100aa158bec82da8`. [Registered scope](../docs/LEDGER.md#f-c21p--check-the-executable-fblight-trial-zip-on-the-linux-rtx-worker)

### F29 recurrent waveform feature readout

- Ferranti real P01 smoke `511580` ran after all F28 source students and passed in 11 seconds. Exact F16 hard, caller RNG and zero-residual identity passed. Both learned arms use 432,911 trainable parameters. H100 32-window medians 1.754 ms hard, 1.913 ms chronological and 1.906 ms scrambled. The chronological residual and order-only movement were 0.1023/0.0470. Both true latent prefix errors were zero. [Registered design](fbc323_recurrent/prereg-19nov.md)
- Held smoke `3134335` gates source full array `511581`, matched Galvani `3134336` and collector `3134337`. Sources/held capped at four. Release names end in `s1c_recurrent_20261119`. Actor SHA-256 `51461fb9084c99c0ac3f607debf21babeaac2667c88135173555e21df4b143ee`, scorer `7f56f8906706c4e7990dad120d003b7518ae7f8d0612327cab7511ce9cc6ef04`. Relay `bash friday/fbc323_recurrent/run_dependencies.sh 511580 511581 3134335 3134336`.
- F29 independent held smoke `3134335` passed in 37 seconds. Ferranti `511581` released with three jobs starting at 06:26 CET. The independent F28 held array is complete and only its separate scorer remains in flight. Neither release writes to the other.

- F28 completed at 06:28:10 CET. All teacher, student and held element jobs passed. Both registered held contrasts remain undecided. [Authenticated result](fbc323_relational/result-19nov.md)

- F-C21p worker `3134307` stopped during archive extraction because its private container volume ran out of space. Immutable v2 `3134334` used the bound project volume and passed at 06:25:59 CET. All 62 decisions matched and the maximum posterior difference was 4.205e−8. [Archive and worker receipts](cap250_fbl2_package/README.md)

### Isolated FBLight weight and warm evidence F-C23

- Galvani real smoke `3134352` gates full0 `3134353`, within0 `3134354`, and independent scorer `3134355`. Exact corrected stock, identity and FBL2 are validated against the signed parent banks. Equal added frozen mass isolates the warm term. The fixed triple dose tests additional live weight. Release `ni26_friday_c23_fbl_dose_v1_20261119`. Source `141f811fc748f1353876eb4c5d90059307f7893a90c55cdfab2f1a835720407c` and scorer `5c7e5f5c32159abfcda34d7b04bf2015a974c738817a52d405faeeb4da3716f3`. [Preregistration](../docs/LEDGER.md#f-c23--separate-the-warm-fblight-contribution-from-added-frozen-weight)

- F-C23 smoke passed at 06:31:38 CET on 379 real windows. Stock, identity and FBL2 matched their signed parents. All fixed arms took 18 CSP refits and the equal frozen and live contributions both reached RouteB. Signed smoke `0b40d2a362050748cb4c165ad7d4ebce4ef2e286a474c3555287b66ccab06bcd`. Both complete orders released.

- F29 completed at 06:35:19 CET. The chronological recurrent readout beat the same-capacity shuffled readout, and its accuracy gain over exact hard CE was refuted. [Signed scores](fbc323_recurrent/result-19nov.md)

### F30 source-person episodic head transfer

- Ferranti `511601` passed the real source smoke in 10 seconds. It verified 140 disjoint and class-balanced episodes, equal support/query exposure, active gradients on both sides, independent autograd agreement within 1.8e−7 and byte-exact F16 hard. Independent held smoke `3134377` gates Ferranti `511602`, matched Galvani `3134378` and collector `3134379`. [Protocol and numeric forecasts](fbc323_episodic/prereg-19nov.md)
- Actor SHA-256 `d32d01f71b9eef27c383973238f02810486fd5b2988155e9d16039d3dfeb043a`. Scorer SHA-256 `bfb0f91a49bb951acd432b5936c367564d27efd7d8ff22bc44f5bf29ed8e4d7f`. Releases end in `s1c_episodic_20261119`. Controller `bash friday/fbc323_episodic/run_dependencies.sh 511601 511602 3134377 3134378`. This training-only episode never enters held inference.
- F30 independent held smoke `3134377` passed in 33 seconds. Ferranti `511602` released and started its first job at 06:41:36 CET. With one H100 the estimated complete score is around 06:55–07:00 CET. The full source allocation budget remains in force.

- F30 completed at 06:52:39 CET. All source and held fits passed. Both registered whole-model contrasts remain undecided. [Signed results](fbc323_episodic/result-19nov.md)

### F31 interior waveform order pretext

- Real H100 smoke `511623` passed in ten seconds. Native FIR radius 99 samples, direct real-boundary extension drift 1.252e−6 and gain/offset descriptor drift 2.981e−8. The matched heads used 20,224 opposing orientation examples. Fixed real source chronological/label-sham losses were 0.68561/0.69313 and direction accuracy 58.85/51.56%. The preregistered source threshold passed. [Protocol and predictions](fbc323_order/prereg-19nov.md)
- Galvani real held smoke `3134427` gates Ferranti paired source `511624`, independent held `3134428` and collector `3134429`. The actor `832bf16a1af007cadef7742a2ee3bb08cd27c51a9dccf4ebd9a691f6be38e24d` emits only native inference parameters. Scorer `12d8348244e7769d2e80e8e07158952185ad242e3037d63addf4d78cec368d8a`. Releases end in `s1c_order_20261119`. Controller `bash friday/fbc323_order/run_dependencies.sh 511623 511624 3134427 3134428`.
- F31 independent held smoke `3134427` passed in 35 seconds. Source `511624` released and began two fits at 06:57:37 CET. Current estimated complete held scoring is 07:06–07:10 CET if two H100 slots stay available. No source wall limits changed.

- F31 completed through immutable scorer-only Galvani recovery `3134521` at 07:09:01 CET. Initial collector `3134429` sealed all held decisions before opening labels, then stopped during JSON serialization. The retry restored the original scheme weighting and verified the same signed source and predictions. Both registered held endpoints remain undecided. [Signed result and recovery](fbc323_order/result-19nov.md)

### Optimized stock-protected archive

- Direct F-C21gpack CPU check matched all 151 classes and every tracked posterior exactly, including 30 real CSP fusions on unrecognized G. Clean RTX ZIP smoke `3134547` and combined native conformance `3134548/3134549` with dependent timing `3134550` are separate. [Archive receipts](cap250_gate_package/README.md)

### F32 concurrent external motor supervision

- Real Ferranti H100 source smoke `511647` passed in 11 seconds. Native hard reproduced F16 checkpoint bytes, and literal detach reproduced the same native state. Both private heads learned on the matched licensed PhysioNet batches, with final source CE 1.59924 detached and 1.53538 joint. The external path left native SessionBN state and affine gradients isolated. [Registered design and corpus terms](fbc323_motor_joint/prereg-19nov.md)
- Independent Galvani held smoke `3134551` gates Ferranti source `511648`. Galvani held array `3134552` consumes checkpoint pairs only after each source fold completes. Collector `3134553` uses both native inference modes and seals predictions before opening PROTEUS held labels. Both arrays cap at four. Actor `f8d074e6867ebfda82383c164c257d64f154b251b0ccdbcb5153b297e7917723` and scorer `49458dc244ca5d0583ab09aa4b84bca50ecce59f1d34777fb46d80d9e57e9762`. The live controller is `bash friday/fbc323_motor_joint/run_dependencies.sh 511647 511648 3134551 3134552`. Add `resume` after source release to restart checkpoint relay.
- F32 Galvani held smoke `3134551` passed at 07:22:51 CET in 51 seconds. The exact hard and detached decisions agree across both modes and all eight orders. All 24 first-window and 48 reset checks passed. Maximum posterior prefix drift was 4.982e−7. Ferranti `511648` released and two H100 elements started. Estimated final score with two slots is 07:34–07:41 CET.
- F32 completed at 07:33:49 CET. All source fits, held folds and collector `3134553` passed. The ordinary whole decoder improved at both seeds. Both primary bounded endpoints remain undecided. [Signed result](fbc323_motor_joint/result-19nov.md)

### F32e all-source native export

- H100 source smoke `511668` passed in 11 seconds. All 2,655 public PROTEUS rows followed the exact 172-update F16e smoke schedule. The hard checkpoint matched F16e byte for byte. Detached preserved the identical native state. Joint and detached motor diagnostic cross-entropy was 1.48955 and 1.54991. [Registered source and export checks](fbc323_motor_external/prereg-19nov.md)
- RTX source-only export smoke `3134652` gates the two final Ferranti fits `511669`. Galvani export array `3134653` follows their authenticated transfer. Controller `bash friday/fbc323_motor_external/run_release.sh 511668 511669 3134652 3134653`. Source actor `fdbc95d3649c0a55efb1c4de6fdcfcb19b8d278a6c319392a7e3872f1d90417e`, exporter `f7b940d3efa3d5de5ec7b49a827d4fb8e5d3ea0285e058bf5fcd2f2b9d3485bb`. Releases end in `s1c_motor_external_20261119`.
- F32e completed 07:56:54 CET. Both final exact H100 fits passed. The strict RTX tolerance missed on seed 0 in both renderer variants. Original-renderer CPU jobs `3134690_0/1` passed all 576 held-closed source comparisons with exact decisions and reset, and 1.12614e−6 maximum posterior deviation. [Signed native artifact receipt](fbc323_motor_external/result-19nov.md)

### F33 signed low-band spatial cofluctuation

- Ferranti v1 source smoke `511679` stopped before checkpoint publication. The zero probe ran the newly installed SessionBN in training mode. Obsolete held jobs stayed closed. Ferranti v2 `511681` passed at 08:40:49 CET after eval-only state isolation. Warm 32-window forward medians 1.754/2.121/2.119 ms for hard/aligned/permuted, 0.797 GB peak reserved. Both learned arms use 349,607 trainable parameters. Real projected gain and marginal/energy controls and native hard bytes passed. [Protocol](fbc323_covariance/prereg-19nov.md) · [Recovery](fbc323_covariance/source-recovery-19nov.md)
- Galvani held smoke `3134875` gates Ferranti source `511682`, independent held `3134876` and collector `3134877`. Controller `bash friday/fbc323_covariance/run_dependencies.sh 511681 511682 3134875 3134876`. Releases end in `s1c_covariance_v2_20261119`. Actor `984e6a7c7776fefca6fd70924ddc135b92dd53ab3fdffd206fc435ccaa5df70f`, scorer `d7f3be1774cfc61fa6744755bd53108f0ce62c76f578e65af21aae12479b653e`. Estimated final score 08:49–08:55 CET given two H100 slots.
- Independent v2 held smoke `3134875` passed at 08:41:56 CET, with 24 exact first-window checks, 48 exact resets and maximum prefix drift 4.982e−7. Source array `511682` released. The controller transfers each authenticated checkpoint trio into the held matrix as it completes.

### Thursday 19 November, 08:42 CET

- F-C25 external same-live factorial real RTX smoke 3134864 passed on recognized P10 and unrecognized G. All nine workers 3134865 are gated on that proof and rotate at cap four. The blinded collector is 3134866. F-C25d corrected held-new runs 3134879/80 depend on real smoke 3134878. Independent collector 3134881 seals both orders. [Design, release and receipts](cap250_f32_cross/README.md)
- F33 whole raw waveform tests synchronous spatial cofluctuation against a matched temporal derangement and exact hard. Ferranti source smoke 511681 and Galvani independent held smoke 3134875 passed. Active Ferranti array 511682 feeds Galvani folds 3134876 and scorer 3134877.
- F-C26 genuine peer-guided warm refit smoke 3134897 gates external array 3134898. F-C21gpackD independent RTX package and untouched-stock diagnostics use Galvani 3134870.
- At 08:48 CET, Ferranti and Galvani had completed both released folds. P03 had a Slurmd affinity launch failure and no fit artifact. Slurm requeued it for 09:02:11 CET. The next planned Ferranti start was 10:13:08 CET. The usual six-minute allocation remains, with a three-minute backfill minimum on pending elements. Independent Galvani source-only A100 canary `3134993` is running, without a held array. Canary manifest `84511237df30a0ffc25c1abe096b9c9b539418b07bbf8acab2dfa821f54186dc`. [A100 scope](fbc323_covariance/a100/prereg-19nov.md)
- Galvani A100 source-only canary `3134993` passed at 08:49:10 CET in 39 seconds. Exact native initialization agreed with H100 and the final fitted hard fingerprint differed across devices. Bare and F33 hard states were exact on the A100. Warm medians 4.019/4.453/4.506 ms. No held predictions or device substitution. Source receipt `16ce19d2812306824f4bf565a749ab95eb5e88493056ca26cc34356f5a66c556`.

- F-C25d real 379-window held smoke 3134878 passed at 08:44:17 CET with seven independent tails, genuine count latch and exact signed parents. Receipt `fed00492602618b59afb5d82e9d0f1d6e25f37c0477ed71674f259f9db1abaf0`.
- F-C27 scored the fixed motor-preserving cognitive output on closed F32c/F32x banks. Held 3134999 and external 3135000 both passed by 08:50:41 CET. [Scoped effects and signed receipts](cap250_cognitive_output/result-19nov.md)

### F25e all-public-source modulation exports

- Galvani A100 source smoke `3135003` passed in 22 seconds at 08:56:50 CET. The independently fitted native hard checkpoint matched the wrapped A100 hard bytes. Its H100 counterpart has a different trained state and preprocessing scale differed by one float32 ULP. All 2,655 official public rows and 172 smoke updates matched. Warm hard/single/multi medians 2.991/3.576/3.574 ms, peak reserved 0.847 GB. [Protocol](fbc323_modulation_external/prereg-19nov.md)
- Original worker-image CPU smoke `3135008` gates final matched A100 source array `3135009`, CPU export array `3135010` and independent seal `3135011`. Release `ni26_friday_s1c_modulation_external_a100_20261119`. Actor `b9d64a585b6f75a806a0d27c8b04fd32d41c8baf565a46ac07e7835e2c673e4b`, exporter `65583a897b96f5ac9ae76a7aac260aadd1f257086412828fe6b9049a5f8ce0ae`. Estimated 09:02–09:08 CET if the same A100 node remains available. Source/export only, with no Scherer exposure.
- F25e signed at 09:03:11 CET. Both A100 production groups and both CPU exports passed. The six actors reproduced all 576 original CPU bounded decisions and strict reset tests, maximum posterior drift 1.33910e−6. Independent bundle seal `3135011` passed. [Source-only bundle](fbc323_modulation_external/result-19nov.md)

- F-C25 held-new scorer 3134881 completed at 08:54:40 CET. External scorer 3134866 completed at 08:59:07 CET. Hard-only won externally. Its four-order pooled result finished at 09:06:40 CET. [Factorial](cap250_f32_cross/README.md) · [Pooled hard-only](cap250_hard_pooled/result-19nov.md)
- F-C22e verified the sealed six-actor A100 source bundle and source-only CPU streaming proof. Galvani 3135022 smoke passed at 09:06:59 CET, followed by nine eight-arm matched external jobs 3135023 and independent scorer 3135024. Fresh delivery seeds 20411/21411. Worker manifest `5ab684af163168f53d617c66778091bee19def82983f2c4bf407c7e713a8fc5b`, scorer manifest `bf7c82f002ce267c595ffdf29eb762be58baae3ea09a3f8a54af1441ac8fd4f7`.
- F-C26 peer-guided native refit completed 09:08:54 CET. Peer −0.185 pp against exact stock, −0.030 pp against matched sham, paired intervals spanning zero. [Result](cap250c26/result-19nov.md)
- F-C28 fixed pooling of complete C25d terminals completed 09:12:06 CET and missed qualification versus the early joint. [Result](cap250_fbl_wave_pool/result-19nov.md)
- Hard-only direct package qualification F-C25p 3135050 and dependent runtime 3135051. Trial ZIP SHA `6a569180d343c038558fe7eccb7903a36e2bdf694af0df3687bff3b17049fbcc`.

### Thursday 19 November, 09:18 CET

- F33 Ferranti source retained its original H100 equivalence rule. Two fits and their matched held folds passed. Later jobs wait for resources. Leonardo rejected the configured key at 08:46 CET.
- F33a independent A100 replication leaves the H100 run separate. Source smoke 3135069 passed at 09:17:07 CET with exact same-device hard/bare parameters. Durable DAG on Galvani 3135079 held smoke, 3135080 source fits with cap six, 3135081 held folds and 3135082 scorer. The two-seed raw comparison keeps the same people, methods and fit recipe. [F33a registration](fbc323_covariance/a100_whole/prereg-19nov.md)
- F-C25p trial v1 reached all 120 genuine G decisions and verified its same-input native stages. An untouched cold/warm stock repeat stopped the independent check. Unchanged direct ZIP and separate warm RTX conformance jobs 3135066 precede timing 3135067. The independent threshold is unchanged.
- F-C22e completed the first four matched external workers. Four more are active. The final participant remains gated by the four-job array cap, with independent score 3135024 dependent on all nine.

### F33a independent A100 whole-decoder replication

- Galvani device check `3135065` identified NVIDIA A100-PCIE-40GB on cn238. Real source `3135069` passed in 45 seconds and reproduced the earlier canary and identical bare/hard checkpoint SHA-256 `0e92dcf9294750ef7c9a5204b624d6e35dc72af521f902f551bf08b73de8e0c8`. Warm hard/aligned/permuted forwards 3.891/4.372/4.360 ms. Peak CUDA reserved 0.797 GB. [Registered matched science and scopes](fbc323_covariance/a100_whole/prereg-19nov.md)
- Same-cluster Slurm controller `launch_dependents.sh` and tracked remote `jobs.txt` record held smoke `3135079`, source array `3135080` at 20%6, matching per-fold CPU held array `3135081` at 20%6, and blind scorer `3135082`. Release `ni26_friday_s1c_covariance_a100_whole_20261119`, source `94598f4eb485b20ea40c0d7c13f4497ed07d04e9e9c062c72953e97460b782bc`, scorer `6c8e27573d2b91176ec6d2e98068e94d8e671d878c92e515f6c58899ce558dbe`. A100 hard and bare are replayed independently. Ferranti H100 F33 remains a separate wave.

- At 09:20 CET, F-C22e had seven complete external participants with the last two running. Its collector remains dependency-gated. F33a held smoke passed at 09:19:35 CET with all first, interleaved reset and bare/hard checks. Six source folds then started.
- Hard-only v2 recognized P10 passed. Same-live G checks and known hard decisions passed; independent warmed untreated stock A/A exceeded 5e−5, reaching 8.29270e−5. Both original tolerances and the failed jobs stand. The dependent paired timings were canceled.
- F33a completed 09:28:18 CET. The CPU gate passed in 59 s. All 20 A100/20 CPU elements and blind scorer `3135082` passed. Both registered bounded contrasts remained undecided. The independent bare control reproduced all 84,960 hard decisions. [Signed result](fbc323_covariance/a100_whole/result-19nov.md)

### F34 carrier-independent envelope morphology

- A100 source v1 `3135174` failed before held predictions because float32 analytic tokens moved 2.861e−6 under the fixed rotation. Matched float64 spectral/power v2 `3135190` passed in 18 seconds. Analytic phase drift 3.553e−15, real-power motion about 1.71, and peak CUDA reservation 1.087 GB. Warm hard/analytic/real-power forward medians 3.893/4.340/4.342 ms. Hard reproduced exact F33a artifact bytes. [Protocol](fbc323_envelope/prereg-19nov.md) · [Precision repair](fbc323_envelope/precision-19nov.md)
- Galvani CPU held smoke `3135208` gates six-at-a-time A100 source `3135209`, matching causal CPU folds `3135210` and blind scorer `3135211`. Same Slurm `launch_dependents.sh` as F33a, with durable remote `jobs.txt`. V2 release `ni26_friday_s1c_envelope_a100_v2_20261119`, actor `56ca60c2d190b3b51005f09cd53d66243af92ba2a319e58544594d0e9a94e60c`, scorer `899e0abb8c25227f8d0dbbb2061b53e49b8d9c5880da423ef119efb760fb6db5`.

### Thursday 19 November, 09:39 CET

- C22e 3135024 completed at 09:27:24. Selection audit 3135176 completed at 09:34:16. [Paired and simultaneous intervals](cap250c22_external/result-19nov.md)
- F33a A100 held score finished at 09:28:18. F34 real CPU held smoke 3135208 passed and A100 3135209 with held fold 3135210 and scorer 3135211 are active. Ferranti H100 F33 continues separately.
- Hard-only package v3 two-fixture timing 3135142 passed. Exact hard archive decision-level exterior smoke 3135151 passed at 09:35:53. Fresh orders 22411/23411, RTX array 3135152 cap four and blind scorer 3135153. Release `ni26_friday_c25q_exact_v2_20261119`. Worker `6c48a0cfc125bf1e4b8fe08f674959123948e19b9f1b364d560958b6b81b81e8`, scorer `88cfb37db1c63596c262dbcda3a389b2c7bc552570b077712b869f9389126588`. First smoke stopped in worker boot before inference. The new release corrected the worker entry point.
- F-C25pc forced-unrecognized long-stream RTX technical stress 3135353. Independent comparison uses the same package bytes. Release `ni26_friday_c25pc_postcap_v1_20261119`, source `ba670fcc40d10308700d9deae8f692e129a59abffc0f889364712c0099008989`.
- F-C29 source and RTX native P10/G passed. Fresh Scherer array 3135169 and scorer 3135205 run alongside F-C29d corrected full-system held arms 3135202/03 and blind scorer 3135204.
- F34 signed at 09:44:53 CET. All twenty A100 fits/CPU held jobs and scorer `3135211` passed. Analytic envelope gain against exact hard was refuted in both seeds. Both learned parallel branches lost sharply in held transfer. [Signed F34 result](fbc323_envelope/result-19nov.md)

### F35 training-only source-person contribution balance

- A100 source smoke `3135480` passed at 09:51:56 CET in 17 seconds. Exact hard bytes and zero-transfer optimizer matched F33a. Source used all 160 original updates, including 140 paired-person groups. Analytical head-gradient agreement 2.384e−7. Frozen anchor imbalance fell 65.21%. On actual branch trajectories, relative imbalance went 0.231→0.105 for balance and 0.227→0.357 for the equal mirror. [Registered scope](fbc323_gradient_balance/prereg-19nov.md)
- Independent CPU held smoke `3135486` gates Galvani A100 source `3135487` and matched CPU held `3135488`. Scorer `3135489` reauthenticates the source tape and seals all decisions before labels. Native Slurm DAG in release `ni26_friday_s1c_balance_a100_20261119/jobs.txt`. Actor `b153df05e923c0b40795adaed413d8d024087e49e4b2be51445d81b6180db42e` and scorer `eaa05cb8ee29a01106c56cf5c17fd457ce607ab7426468f7e1e2046a7c7be6a4`.

### Thursday 19 November, 09:53 CET

- Hard-only after-cap corrected witness 3135432 passed at 09:50:49 CET, including all 118 live warmed raw calls from age 260 through 377, exact cap work and original buffers. 3135152 direct full-stream first four participants completed, second four active. Independent accuracy remains blind. [Direct assay and witness](cap250_hard_exact/README.md)
- F-C29 fit-excluded full0 +0.330 pp and within0 −0.187 pp vs stock. Recovered external score 3135467 preserved all nine sealed workers and landed −0.093 pp vs stock. No advancement. [Scope and comparison](cap250c29/result-19nov.md)
- F34 completed at 09:44:53. Both matched larger-envelope arms lost sharply to the exact small hard A100 decoder. F35 trains the small original inference model with bounded participant gradient balancing using a mirrored control, and source smoke 3135480 passed. CPU held smoke 3135486 gates jobs 3135487/88/89. Ferranti F33 H100 relay continues separately.
- F25a all 20 source/held folds completed with exact independent same-device hard and real pupil checks. F-C22h actual count-latched smoke 3135428 passed. Two corrected full paths 3135429/30 and independent scorer 3135431 are active.
- Ferranti F33 recovered all missing fits and completed the original H100 matrix. Galvani collector `3134877` passed at 09:58:56 CET. Both registered endpoints remain undecided. [H100 signed result](fbc323_covariance/result-19nov.md)
- F35 completed at 09:59:56 CET. All fits and CPU holds passed. Whole-accuracy points improved with balance and mirror, with the two preregistered bounded contrasts undecided. [Signed result](fbc323_gradient_balance/result-19nov.md)

### F35e all-source native balance exports

- A100 source smoke `3135554` passed at 10:04:13 CET in 19 seconds. All 172 updates reproduced the exact independently fitted F25e hard and no-dose state. The frozen source head-imbalance reduction was 62.53%. [Registered source and CPU scope](fbc323_balance_external/prereg-19nov.md)
- Independent source-only CPU smoke `3135576` gates final A100 seeds `3135577`, per-seed CPU exports `3135578` and signed seal `3135579`. Galvani release `ni26_friday_s1c_balance_external_a100_20261119/jobs.txt`. Actor `35f846402a40dcc100ba05cec5539110542cb833c1367bf5f32d87567d822248`, exporter `4af04d73ec4a55a4897933b25571d667f717bdbddc2b73bea9d6b445db4fea35`. Estimated 10:07–10:10 CET.

### Thursday 19 November, 10:04 CET

- F-C25q independent exact archive and both real stock repeats scored at 10:04:11. Hard +0.050 pp [−0.370,+0.456]. It remains the lead experimental alternate. [Whole exact archive](cap250_hard_exact/README.md)
- F-C22h complete A100 held-new hard improved +0.425/+0.374 pp vs stock on the two corrected orders. Conventional remained below hard within0 and unresolved full0. [Matched held-new replication](fbc323_modulation_a100/result-19nov.md)
- Original Ferranti F33 and separate A100 F35 completed. F35 whole-raw balance +0.430/+0.924 pp vs hard with wider same-dose mirror uncertainty. F35c corrected complete system and F35e matched all-public CPU exports are active. New F35x external code staged on Galvani for fresh 26411/27411, pending the signed actor lock.
- F35e sealed at 10:07:45 CET, all source/CPU jobs and independent seal `3135579` passed. Six compact native CPU actors match 576 exact original decisions with 1.1992e−6 maximum posterior difference. All-source hard reproduces exact A100 F25e bytes. [Signed bundle and source audit](fbc323_balance_external/result-19nov.md)

### Thursday 19 November, 10:11 CET

- F35c completed all 20 held folds. Complete-system smoke `3135586` passed at 10:09:03 CET with exact stock, identity and independent hard. Both corrected orders `3135587/88` started at 10:09:24 CET. Blind scorer `3135589` follows both. First results estimated 10:15–10:17 CET.
- F35x starts with real recognized and unrecognized RTX smoke `3135590`. Nine paired external participants `3135591` and the independent scorer `3135592` follow in order. Fresh delivery orders 26411 and 27411 test the positive standalone balance result in the complete native stack against stock, A100 hard, the matched mirror and the H100 hard reference. [Run manifest](cap250_balance_external/README.md)

### Thursday 19 November, 10:17 CET

- F35c blind scorer `3135589` passed at 10:15:17. Balance versus exact A100 hard −0.272 pp within0 and +0.146 pp full0. The full-system comparison missed the registered advancement rule. [Signed held-new result](cap250_f35/result-19nov.md)
- F35x RTX smoke `3135590` passed at 10:11:14. Two external participants finished and four are active at the 10:16 check. The collector opens held labels only after all nine pass. [Native external manifest](cap250_balance_external/README.md)

### Thursday 19 November, 10:22 CET

- F35d real full0 bank smoke `3135625` and independent stage scorer `3135626` passed. CountFilter and CSP together shifted balance minus hard by −0.229 pp [paired 80% −0.341,−0.082] within0 and −0.002 pp full0. Every original terminal result reproduced. [Stage result](cap250_f35/result-19nov.md#where-the-standalone-effect-changed)
- F-C31 real source smoke `3135612` passed and nine bounded CPU probes `3135615` are running at cap three. Blind scorer `3135616` follows. The test uses a local sensor reconstruction against an age-matched control and the exact independent stock/hard archive decisions. [Registration](../docs/LEDGER.md#f-c31--probe-reconstruction-sensitivity-on-exact-hardstock-disagreements)

### F36 same-session reciprocal raw waveform crossfade

- A100-PCIE-40GB source smoke `3135613` passed in 17 seconds at 10:20:12 CET. The native hard checkpoint matched exact F33a bytes. Both active arms used the same 2,528 of 10,112 source appearances. Mean relative raw RMS displacement was 0.494280, maximum paired mismatch 4.63e−9, with bit-exact zero dose and unchanged edges. [Prospective protocol](fbc323_crossfade/prereg-19nov.md)
- CPU held smoke `3135629` passed at 10:23:09 CET in 44 seconds with all 24 first-window and 48 interleaved-reset checks. Maximum posterior prefix drift 4.41115e−7. Galvani native DAG `3135630` trains 20 folds at cap six, `3135631` replays matching folds after each source element, and `3135632` seals decisions before labels. Release `ni26_friday_s1c_crossfade_a100_20261119/jobs.txt`. Actor `baf7735eb9ca7f225b99ecbf58e6a16a8fea3a13813ae7c2e8434d0fff901510` and scorer `2734eeee0e398a53ffc5900b303255a2ef193df31d2adfb448b59eb1aba67367`.

### Thursday 19 November, 10:31 CET

- The reconstruction selector scored at 10:23:32. Current cue −0.112 pp versus hard and −0.023 pp versus the one-window-lag control. It fixed 11 and broke 16. [F-C31 signed result](cap250c31/README.md)
- F35x scorer `3135592` completed at 10:27:32. Balance tied matched A100 hard and trailed H100 hard by an uncertain 0.093 pp. Six-order H100 pooling with the same signed actors completed at 10:30:40. Hard gained +0.220 pp [paired 80% +0.018,+0.432] versus stock on the same nine external people. The independent exact ZIP result remains +0.050 pp and the numerical veto persists. [F35x](cap250_balance_external/README.md) · [Six-order reconciliation](cap250_hard_pooled/result-19nov.md#six-order-extension--19-november-103040-cet)
- F36 scorer `3135632` passed at 10:30:15. Reciprocal cross-trial minus native hard was −0.094/+0.619 pp across seeds. Single-trial matched raw changes had +0.389/+0.444 pp points with both intervals spanning zero. The follow-up checks the same control in the complete system.

- F36 whole decoder collector `3135632` completed at 10:30:15 CET. All 20 matched A100 and CPU folds passed. Bounded cross-trial minus exact hard −0.094/+0.619 pp and minus matched same-trial −0.483/+0.175 pp, both undecided. [Signed result](fbc323_crossfade/result-19nov.md)

### F36c corrected cap250 with both F36 pupils

- Fresh original recording-order `replay/check.py` printed OK. Native CPU P01 pupil smoke `3135705` passed both orders and bitwise F-C22h hard posteriors. Smoke bank `5936859778fe38589bf6e5bd178bc32e92a266f8bb02b5465a91de6a43ab1dcd`. [Preregistered comparison](cap250_f36/prereg-19nov.md)
- Galvani `3135713` replays all 20 fit-excluded pupil pairs. Corrected full0 native pilot `3135714` gates `3135715/3135716` within0/full0, then independent blind collector `3135717`. Release `ni26_friday_f36c_held_20261119/jobs.txt`. Pupil `577fddd5a31ddc6248e7e8ca4679acb8a948f543f726ee06c071c1693c61db83` and whole `f346e2cb9d1de08a93c631fd46be50aa978cb129ae15ac208dcb4a7979f844a1`.
### Thursday 19 November, 10:41 CET

- F36c real 379-row corrected pilot `3135714` passed. All twenty fit-excluded pupils reproduce fixed standalone decisions and the same-device A100 hard anchor. Corrected within0/full0 jobs `3135715/16` and independent scorer `3135717` follow. [Exact stack protocol](cap250_f36/prereg-19nov.md)
- F-C31c paired source-tape smoke `3135703` passed and fourteen fixed sessions are in array `3135711`. Independent reauthentication `3135712` checks the live route and source operands before any conditional training. [Mixer registration](cap250_mix_objective/prereg-19nov.md)
- F-C25pdet immutable-stock RTX smoke `3135765` and paired fresh-process diagnostic `3135766` compare native default with pre-start deterministic control. No archive change. [Technical scope](cap250_hard_determinism/README.md)
- F37 true H100 source smoke `511781` passed at 10:40:12. Independent CPU smoke `3135762` gates release of Ferranti array `511782`, Galvani held folds `3135763` and blind scorer `3135764`. Tested a distinct training-only spectral amplitude shift around alpha with the equal-dose beta and exact hard controls. [Run manifest](fbc323_spectral_shift/README.md)
