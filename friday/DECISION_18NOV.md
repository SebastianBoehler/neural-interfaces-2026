# Wednesday Track 2 decision brief · working version

**Current selection is staged v6.6cap250. The alternate is sealed v5.1b.** The clean waveform variants tested so far trail cap250 on ten held-new public people. The fixed spectral kernel's gains did not reproduce on the independent Scherer cohort. Sebastian chose stronger held-person and external evidence as the tie-break inside the development noise band. At 18:00 CET this brief needs a fresh leaderboard check and any completed F5, F6, or cap250 refinement results. Both listed archives passed SHA and ZIP CRC checks on 18 November.

## Sealed score and rank scenario

At 13:39 CET on 18 November, Codabench sealed v5.1b entry 1050185 scored **0.8197668759**, ranked **#20** by entry, and took **1,287.04 seconds**. Cap250 is staged. Its private score is unknown. Thresholds hold other entries fixed and use the same leaderboard task (36954). [Live Codabench result](https://www.codabench.org/api/leaderboards/19997/)

| Target | Entry score to beat | Gain from v5.1b |
| --- | ---: | ---: |
| #19 | 0.820745 | +0.098 pp |
| #15 | 0.823771 | +0.400 pp |
| #14 | 0.825365 | +0.560 pp |
| #13 | 0.831416 | +1.165 pp |
| #10 | 0.835031 | +1.526 pp |
| #5 | 0.841214 | +2.145 pp |
| #1 | 0.859411 | +3.964 pp |

An entry score of 0.824 would place #15 in this snapshot. Scores of 0.826–0.829 would place #14. These are rank scenarios with fixed competitors. The public development gain has no calibrated conversion to a sealed score.

## Accuracy evidence

| Comparator and instrument | Observed difference |
| --- | --- |
| Cap250 vs v5.1b, FULL-SYSTEM DEV, ten held-new PROTEUS people, eight orders | +1.380 pp [80% +0.742,+1.988]. Three deployment seed counts matched. |
| Cap250 vs v5.1b, FULL-SYSTEM DEV, conditional returning P04/P10 | +1.483 / +0.575 pp. The assumed 20/80 returning mix gives +0.757 pp. Combined with 56.6% held-new gives +1.110 pp. |
| Exact cap250 vs v5.1b, FULL-SYSTEM EXTERNAL, nine Scherer people | +0.493 pp [−0.311,+1.286] and +0.040 pp [−0.837,+0.929], two order pairs. +0.266 pp across four orders on the same people. Montage and crop changes move the estimate. |
| Spectral kernel vs cap250, FULL-SYSTEM DEV, held-new seeds 454/1454 | +0.217 / +0.158 pp. Two-seed mean +0.177 pp [80% +0.080,+0.265]; −0.041 pp [−0.093,+0.008] versus seed 454. |
| Spectral kernel vs cap250, FULL-SYSTEM DEV, returning mix | +0.370 pp [−0.124,+0.507] and +0.268 pp [+0.141,+0.367], eight paired orders and only two people. |
| All-source spectral kernel vs exact cap250, FULL-SYSTEM EXTERNAL | −0.212 pp [−0.625,+0.193] and 0.000 pp [−0.291,+0.288], nine Scherer people and four orders, 8,520 decisions per arm. |
| F1–F4, FULL-SYSTEM DEV, ten held-new PROTEUS people | Every tested insertion was below 76.1039% cap250. Best seed point estimate −0.347 pp. Genuine history sometimes improved a weak model enough to beat its own blank control. |
| F-C1/F-C2, FULL-SYSTEM DEV | Count prior scale −0.110 pp new, −0.098 returning. Shape-normalized CSP fit −0.004 pp new. All three intervals cross zero. |

[Exact archive selection evidence](../../neural-interfaces-2026/evidence/next_wave_16nov/lunch_recommendation_18nov.md) · [Returning spectral kernel](../../neural-interfaces-2026/evidence/next_wave_16nov/kernel457.md) · [Waveform evidence](wave/results-18nov.md) · [External receipts](../../neural-interfaces-2026/evidence/next_wave_16nov/EVENING_HANDOFF_18NOV.md)

## Proven A100 runtime

- Sealed v5.1b had **2,313 seconds** of measured headroom against the 3,600-second limit. On two standardized four-session A100 process pairs, exact v6.6cap250 took **584.05 vs 349.50 seconds** and **211.22 vs 114.61 seconds**. The hidden workload, ratios on sealed streams, and actual cap250 sealed time remain unknown. [Exact process pairs](../../neural-interfaces-2026/evidence/next_wave_16nov/budget414.md)
- The native cap250 release reproduced every label and work counter. Its paired solver replay was **1.05% faster** on 2,655 ordinary public windows and **3.23% faster** on 1,178 forced-adaptation windows. Each workload has one pair. [Native release parity](../evidence/next_wave_16nov/native366_worker.md)
- Isolated C++ CUDA graph replay for two kernel seeds took **0.169–0.170 ms per EEG window** on P04/P10. The clean worker replay on two external people measured **0.31–0.38 ms per window** for the graph increment and **0.991×/1.001×** with respect to stock for complete session time. Two rotated warm pairs per person had matching stock labels and work counts. The graph is affordable in these workloads. Its accuracy evidence does not justify replacing the staged package. [Isolated timing](../../neural-interfaces-2026/evidence/next_wave_16nov/native458.md) · [Worker receipt](../../neural-interfaces-2026/evidence/next_wave_16nov/EVENING_HANDOFF_18NOV.md)

## Decision condition

Prefer the untouched staged cap250 ZIP on the current evidence. Preserve v5.1b as the sealed alternate. F5 learns against historical fit-excluded stock mistakes and runs in independent exact Scherer sessions. F6 transfers independently pre-trained PhysioNet sensorimotor weights into held PROTEUS folds. F-C3/F-C4 test active optimizer behavior in the exact cap250 worker. Their results govern any experiment that advances. See current job dependencies in the [Friday queue](QUEUE.md) and archive identities in the [frozen input manifest](inputs.sha256).
