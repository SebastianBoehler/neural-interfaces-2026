# Competition information

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
2 of 5 daily and 2 of 100 total warm-up submissions used afterwards, with
worker image `lionelyneuro/neural-compet:v3`.

| ID | ZIP (SHA-256 prefix) | Model | Local replay | Codabench score |
| --- | --- | --- | ---: | ---: |
| 1000481 | `warmup1_ensemble_official.zip` (`a8f11f33`) | v2 + 10 ShallowConvNets, official train | 0.44109823028673834 | 0.44109823028673834 (23.7 s) |
| 1000482 | `warmup2_v1_official.zip` (`76fc58fb`) | v1 tangent space, official train | 0.4972515758249907 | 0.4972515758249907 (15.7 s) |

Both finished on the worker (phase "Warm-up Phase (update 1)"). Cell-mean
balanced accuracy, pooled balanced accuracy and accuracy are bit-identical
to the local replays, so the worker runs these packages exactly, including
the torch ensemble. Durations are the worker-reported scoring durations.
Raw export: `evidence/warmup-2026-10-18/codabench-results.json`. Warm-up scores remain indicative: P04 is 6 of the 8 cells. No
sealed-phase submission has been made.

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
