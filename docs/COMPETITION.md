# Competition information

> **Current facts (11 Nov 2026):** sealed phase 30483 closes 21 Nov 2026, 22:59 UTC. Allowance: 7 uploads per team,
> 1 per UTC day (reset 00:00 UTC); we have used 4 (3 left). A failed run uses no quota; a manually cancelled one does. The
> cohort size is private. Runtime: 60 min written, 80 min actual worker timeout. The dates and quotas in the tables
> below are the October snapshot and are superseded by this note and docs/SEALED_UPLOADS.md.

Scope: Track 2, BCI decoding, EEG/EMG Foundation Challenge 2026.
Task: classify motor imagery, mental subtraction and word generation from EEG.
Public development dataset: PROTEUS, NEMAR nm000290 v1.0.0.
Only sealed final scores determine competition ranking.

## Dates and submission quotas

Website reviewed 17 October 2026. Codabench timestamps and total quotas below
were checked in the signed-in interface on 15 October; recheck before uploading.

| Milestone | Europe/Berlin time from Codabench |
| --- | --- |
| Warm-up closes | 25 October 2026, 02:00 CEST |
| Sealed final opens | 28 October 2026, 01:00 CET |
| Sealed final closes | 22 November 2026, 01:00 CET |

The website labels the final deadline 21 November AoE. That differs from the
earlier Codabench cutoff above. Plan against the earlier cutoff until resolved.
Registration remains open through 24 October; Track 2 approval was verified.

| Phase | Daily quota | Total quota shown in Codabench |
| --- | --- | --- |
| Warm-up | 5 | 100 |
| Sealed final | 1 | 10 |

Quotas apply to the account, so check current usage before submitting.
The local scripts never upload a submission.

## Warm-up uploads

Uploaded by the user on 18 October 2026 (Europe/Berlin). Codabench showed
2 of 5 daily and 2 of 100 total warm-up submissions used after the first two;
1000615 followed at 10:13. Worker image `lionelyneuro/neural-compet:v3`.

| ID | ZIP (SHA-256 prefix) | Model | Local replay | Codabench score |
| --- | --- | --- | ---: | ---: |
| 1000481 | `warmup1_ensemble_official.zip` (`a8f11f33`) | v2 + 10 ShallowConvNets, official train | 0.44109823028673834 | 0.44109823028673834 (23.7 s) |
| 1000482 | `warmup2_v1_official.zip` (`76fc58fb`) | v1 tangent space, official train | 0.4972515758249907 | 0.4972515758249907 (15.7 s) |
| 1000615 | `warmup4_ensemble_official_r3.zip` (`b5a3cd23`) | round-3 ensemble (v2 + ShallowConvNets + FBCNets, rebalanced), official train | 0.4939964157706094 | 0.49 (two decimals shown) |

All three finished on the worker (phase "Warm-up Phase (update 1)"). For
1000481 and 1000482, cell-mean balanced accuracy, pooled balanced accuracy
and accuracy are bit-identical to the local replays; 1000615 matches at the
two decimals shown, confirming the plain-torch FBCNet and causal
rebalancing code path on the worker. Durations are the worker-reported scoring durations.
Raw export: `evidence/warmup-2026-10-18/codabench-results.json`. Warm-up scores remain indicative: P04 is 6 of the 8 cells. No
sealed-phase submission has been made.

## Warm-up uploads after the outage (26 Oct 2026, Warm-up Phase (update 1))

Uploaded by Claude with the user's explicit authorization, quota checked
first (0/5 today, 3/100 total). Both are the exact staged sealed ZIPs; their
scores are inflated by design (trained on the warm-up test participants) and
check worker execution and runtime.

| ID | ZIP (sha256) | Cell-mean balanced accuracy | Duration | Local / replica |
|---|---|---:|---:|---|
| 1010506 | `sealed_candidate_ensemble_all_r6nfg.zip` (`9d430f0d…`) | 0.9973118280 | 24.0 s | 0.9973118280 / 40.6 s (replica) |
| 1010508 | `sealed_candidate_ensemble_all_r8t4g.zip` (`725cfcdc…`) | 1.0000000000 | 25.6 s | 1.0000000000 / 46.5 s (replica) |

| 1010748 | `warmup9_r8t4_tv.zip` (`ace07590…`; r8t4 trained on official train + validation only, **honest**) | 0.5303723852 (P04 0.493, P09 0.644) | 26.7 s | 0.5303723852 (CPU replay) |
| 1014114 | `warmup11_r9_tv.zip` (`5f5d2c96…`; r9, official train + validation, **honest**; uploaded 27 Oct, queued about 12 h on Codabench) | 0.5357966220 | 30.5 s | 0.5357966220 (replay, identical) |

Reading the board (26 Oct): the public leaderboard lists the top 50, all ≥
0.63 (38 at ≥ 0.95). Reaching 0.63 honestly would need about P04 0.60 and
P09 0.72; P04's within-session ceiling with our members is 0.67 (band-power
LDA 0.58), so such entries almost certainly used P04/P09 data. There is no
honest reference on the warm-up board; our honest number is 0.530.

Full board via the leaderboard API (27 Oct, 160 participants, best entry
each): 52 at ≥ 0.90, 16 at 0.60–0.89, 17 at 0.55–0.59, 3 at 0.53–0.54,
9 at 0.50–0.52, 63 below 0.50. Entries at ≥ 0.60 almost certainly used P04/P09
data, as argued above. The 0.55–0.59 cluster needs P04 ≈ 0.54–0.58 with P09 ≈ 0.65. That is
plausible for a stronger honest cross-subject decoder, so it is the realistic
competition. Our honest r9 is 0.536, so on the warm-up proxy we sit about
20th among possibly honest entries. Caveat: 8 cells (6 from P04) give a
noise of several points, and the sealed set is 57% unseen participants
plus 43% later sessions.

The worker runs the nets on its GPU (about half the CPU-path time). At
≈ 21k sealed windows: ≈ 11–12 min. Worker image: `lionelyneuro/neural-compet:v3`.

## Compute and rules

Training compute is uncapped. Organizer evaluation must finish within 60 minutes
on one A100. Submission storage is 15 GB per profile, shared across tracks.
Public redistributable pretraining data are allowed; disclose corpora and compute.
Sealed test data and private clinical data are prohibited. Preserve model licenses.
Top-three entries per track undergo a reproducibility audit.

The linear baseline runs on CPU. Measured training/evaluation: 7.97 seconds;
inference replay: 4.89 seconds, excluding setup and data preparation.
Local data currently occupy 2.2 GB and the environment 1.5 GB.
GPU experiments and Linux/A100 worker compatibility have not been validated.

## Sources

- [Competition and timeline](https://neural-interfaces26.github.io/)
- [Official rules and compute limits](https://neural-interfaces26.github.io/rules.html)
- [Track 2 Codabench](https://www.codabench.org/competitions/17982/)
- [Official source](https://github.com/neural-interfaces26/2026-competition)
- [PROTEUS](https://nemar.org/dataexplorer/detail?dataset_id=nm000290)

Dates and rules are snapshots, not a substitute for checking the live platform.
