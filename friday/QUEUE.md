# Friday Track 2 queue

Branch `codex/friday-clean-stack`. Active through Friday 20 November, 18:00 CET. The existing Track 2 Friday experiment queue heartbeat is active at 50-minute intervals in this chat. Prefer stronger held-person and external evidence to break alternate ties inside the development noise band.

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
