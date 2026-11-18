# Wednesday Track 2 decision brief · working version

**Current selection is staged v6.6cap250. The alternate is sealed v5.1b.** Sebastian set an evening target above 0.845 on the sealed ranking metric. The clean waveform variants tested so far trail cap250 on ten held-new public people. The fixed spectral kernel's gains did not reproduce on the independent Scherer cohort. Sebastian chose stronger held-person and external evidence as the tie-break inside the development noise band. Both archives passed SHA and ZIP CRC checks on 18 November. The exact staged cap250 also passed the real-window reset smoke, and the chain replay printed `OK`.

## Sealed score and rank scenario

At 15:51 CET on 18 November, Codabench sealed v5.1b entry 1050185 scored **0.8197668759**, ranked **#20** by entry, and took **1,287.04 seconds**. Cap250 is staged. Its private score is unknown. Thresholds hold other entries fixed and use the sealed ranking metric, balanced accuracy cell mean (task 36954). [Live Codabench result](https://www.codabench.org/api/leaderboards/19997/)

| Target | Entry score to beat | Gain from v5.1b |
| --- | ---: | ---: |
| #19 | 0.820745 | +0.098 pp |
| #15 | 0.825365 | +0.560 pp |
| #14 | 0.831416 | +1.165 pp |
| #13 | 0.832424 | +1.266 pp |
| #10 | 0.835095 | +1.533 pp |
| #5 | 0.841214 | +2.145 pp |
| Evening target | 0.845000 | +2.523 pp |
| #4 | 0.845643 | +2.588 pp |
| #3 | 0.846932 | +2.717 pp |
| #1 | 0.859411 | +3.964 pp |

An entry score of 0.824 would place #16 in this snapshot. Scores of 0.826–0.829 would place #15. These are rank scenarios with fixed competitors. The public development gain has no calibrated conversion to a sealed score.

At 15:51 CET, the signed-in Sealed Final Phase showed zero of one attempts used today and five of seven total. The phase ends Saturday 21 November at 23:59 CET. The shared evaluation queue determines when a sealed result appears. [Codabench submissions](https://www.codabench.org/competitions/17982/#/submissions-tab) · [Track 2 submission guide](https://www.codabench.org/competitions/17982/participate.md)

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
| F6–F9, held-new WHOLE and FULL-SYSTEM DEV | Supervised raw waveform gained +4.952/+4.421 pp over F3 in the whole-decoder test. Fixed normalization reached 48.070%/48.591%. All eight raw or transferred model variants trailed cap250 as terminal insertions at both seeds. |
| F5 learned correction vs exact staged cap250, FULL-SYSTEM EXTERNAL | −3.284 pp [−4.160,−2.338] and −3.975 pp [−5.033,−2.873], nine Scherer people and two orders. |
| F10 isolated normalization update, FULL-SYSTEM DEV | +0.131 pp [80% −0.036,+0.306] and −0.031 pp [−0.226,+0.167] against stock. The matched-donor contrasts also remained undecided. |
| F-C1/F-C2, FULL-SYSTEM DEV | Count prior scale −0.110 pp new, −0.098 returning. Shape-normalized CSP fit −0.004 pp new. All three intervals cross zero. |
| F-C3/F-C4, exact FULL-SYSTEM EXTERNAL | Stable-pseudo-label weighting −0.185 pp [80% −0.463,+0.116]. Balanced pseudo-class loss +0.119 pp [−0.162,+0.417]. Both contrasts remain undecided. |

[Exact archive selection evidence](../../neural-interfaces-2026/evidence/next_wave_16nov/lunch_recommendation_18nov.md) · [Returning spectral kernel](../../neural-interfaces-2026/evidence/next_wave_16nov/kernel457.md) · [Waveform evidence](wave/results-18nov.md) · [F6–F9 paired results](wave/results-f6-f9-18nov.md) · [External receipts](../../neural-interfaces-2026/evidence/next_wave_16nov/EVENING_HANDOFF_18NOV.md)

## Proven A100 runtime

- Sealed v5.1b had **2,313 seconds** of measured headroom against the 3,600-second limit. On two standardized four-session A100 process pairs, exact v6.6cap250 took **584.05 vs 349.50 seconds** and **211.22 vs 114.61 seconds**. The hidden workload, ratios on sealed streams, and actual cap250 sealed time remain unknown. [Exact process pairs](../../neural-interfaces-2026/evidence/next_wave_16nov/budget414.md)
- The native cap250 release reproduced every label and work counter. Its paired solver replay was **1.05% faster** on 2,655 ordinary public windows and **3.23% faster** on 1,178 forced-adaptation windows. Each workload has one pair. [Native release parity](../evidence/next_wave_16nov/native366_worker.md)
- F-C5 reproduced all **27,336 historical score vectors** bit for bit. It reduced the CPU personalizer time by **17.0% on held-new people** and **79.9% on returning people**. The change has no effect on replay accuracy. Full-worker timing remains unmeasured. [F-C5 evidence](cap250-c5-result.md)
- Isolated C++ CUDA graph replay for two kernel seeds took **0.169–0.170 ms per EEG window** on P04/P10. The clean worker replay on two external people measured **0.31–0.38 ms per window** for the graph increment and **0.991×/1.001×** with respect to stock for complete session time. Two rotated warm pairs per person had matching stock labels and work counts. The graph is affordable in these workloads. Its accuracy evidence does not justify replacing the staged package. [Isolated timing](../../neural-interfaces-2026/evidence/next_wave_16nov/native458.md) · [Worker receipt](../../neural-interfaces-2026/evidence/next_wave_16nov/EVENING_HANDOFF_18NOV.md)

## Decision condition

Prefer the untouched staged cap250 ZIP on the current evidence. F11 waits for one Ferranti training retry before Galvani can complete the second matched score. Ferranti estimated a 16:17 CET start as of 15:53 CET. The additional replay work uses Galvani and stored predictions. Leonardo rejected the current SSH key, while both active clusters remained reachable. See current job dependencies in the [Friday queue](QUEUE.md) and archive identities in the [frozen input manifest](inputs.sha256).
