# Wednesday Track 2 decision brief

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

At 15:51 CET, the signed-in Sealed Final Phase showed zero of one attempts used today and five of seven total. The phase ends Saturday 21 November at 23:59 CET. An organizer reported a large Codabench queue at 16:46 CET. The queue and worker run could put the result after 18:00 CET even with an afternoon upload. [Codabench submissions](https://www.codabench.org/competitions/17982/#/submissions-tab) · [Track 2 Discord](https://discord.com/channels/1405198616934481961/1547897829253451877)

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
| F11 sensor scaling, held-new WHOLE and FULL-SYSTEM DEV | The two same-session training seeds scored 46.348% and 46.374% alone, versus 47.097% and 47.277% for F7. Blending them into cap250 lowered full-system accuracy by 3.244 and 3.372 pp, with both paired intervals below zero. [F11 results](wave_input/result-18nov.md) |
| F-C1/F-C2, FULL-SYSTEM DEV | Count prior scale −0.110 pp new, −0.098 returning. Shape-normalized CSP fit −0.004 pp new. All three intervals cross zero. |
| F-C3/F-C4, exact FULL-SYSTEM EXTERNAL | Stable-pseudo-label weighting −0.185 pp [80% −0.463,+0.116]. Balanced pseudo-class loss +0.119 pp [−0.162,+0.417]. Both contrasts remain undecided. |
| F-C6, causal branch selection vs cap250, FULL-SYSTEM DEV | +0.037 pp [80% −0.039,+0.109] held-new and −0.018 pp [−0.093,0.000] returning. The weighted result was +0.013 pp. Selection with known labels would add +2.119 pp on that same mix. [Historical arbitration evidence](cap250-c6-result.md) |
| D1, external rescue ceiling over cap250 | A label-aware oracle across four alternatives corrected 98 of 2,200 stock misses for +2.285 pp [80% +1.911,+2.659] on two pinned orders. This measures where the alternatives disagree when stock is wrong. A deployable selector would have to identify those cases from EEG. [Paired Scherer analysis](transfer/d1_result.md) |
| D2, early count plus spectral kernel, FULL-SYSTEM DEV | +0.213 pp [80% +0.097,+0.323] versus cap250 and −0.004 pp [−0.045,+0.038] versus the kernel alone. The negative interaction was −0.067 pp [−0.128,−0.014]. [Historical factorial comparison](wave_factorial/result-18nov.md) |
| F-C7, learned branch selection, exact FULL-SYSTEM EXTERNAL | −0.053 pp [80% −0.228,+0.132] versus cap250 and the source-permuted arm. The preregistered F-C6 causal comparator gained +0.208 pp [+0.023,+0.394] on the same nine previously examined Scherer participants and two orders. [Exact external evidence](cap250-c7-result.md) |
| F-C8, fixed causal branch selection, exact FULL-SYSTEM EXTERNAL | −0.020 pp [80% −0.205,+0.146] versus cap250 on two fresh orders for the same nine Scherer people. The two order effects were −0.093 and +0.053 pp. [Order replication](cap250-c8-result.md) |

[Exact archive selection evidence](../../neural-interfaces-2026/evidence/next_wave_16nov/lunch_recommendation_18nov.md) · [Returning spectral kernel](../../neural-interfaces-2026/evidence/next_wave_16nov/kernel457.md) · [Waveform evidence](wave/results-18nov.md) · [F6–F9 paired results](wave/results-f6-f9-18nov.md) · [External receipts](../../neural-interfaces-2026/evidence/next_wave_16nov/EVENING_HANDOFF_18NOV.md)

## Measured A100 runtime and sealed RTX conditions

- Sealed v5.1b finished in **1,287.04 seconds**. In two standardized four-session Galvani A100 pairs, exact v6.6cap250 took **584.05 vs 349.50 seconds** and **211.22 vs 114.61 seconds**. On 18 November the Track 2 organizer confirmed a switch to a slower RTX before the sealed phase and a longer run limit. A participant reported **4,800 seconds**. The live numerical cap and cap250's sealed RTX runtime remain unverified. [Exact A100 pairs](../../neural-interfaces-2026/evidence/next_wave_16nov/budget414.md) · [Track 2 Discord](https://discord.com/channels/1405198616934481961/1547897829253451877)
- The native cap250 release reproduced every label and work counter. Its paired solver replay was **1.05% faster** on 2,655 ordinary public windows and **3.23% faster** on 1,178 forced-adaptation windows. Each workload has one pair. [Native release parity](../evidence/next_wave_16nov/native366_worker.md)
- F-C5 reproduced all **27,336 historical score vectors** bit for bit. It reduced the CPU personalizer time by **17.0% on held-new people** and **79.9% on returning people**. The change has no effect on replay accuracy. Full-worker timing remains unmeasured. [F-C5 evidence](cap250-c5-result.md)
- Isolated C++ CUDA graph replay for two kernel seeds took **0.169–0.170 ms per EEG window** on P04/P10. The clean worker replay on two external people measured **0.31–0.38 ms per window** for the graph increment and **0.991×/1.001×** with respect to stock for complete session time. Two rotated warm pairs per person had matching stock labels and work counts. The graph is affordable in these workloads. Its accuracy evidence does not justify replacing the staged package. [Isolated timing](../../neural-interfaces-2026/evidence/next_wave_16nov/native458.md) · [Worker receipt](../../neural-interfaces-2026/evidence/next_wave_16nov/EVENING_HANDOFF_18NOV.md)

## Decision condition

Prefer the untouched staged cap250 ZIP on the current evidence. Ferranti completed the last F11 retry at 16:14:59 CET. Galvani finished F-C7 by 16:32 and F-C8 by 16:50 CET. The measured methods lacked an improvement large enough to support the evening target. Leonardo rejected the current SSH key. See the [Friday queue](QUEUE.md) and archive identities in the [frozen input manifest](inputs.sha256).

| Checkpoint (CET) | Expected evidence |
| --- | --- |
| Wednesday 16:06, completed | F-C6 causal branch selection scored +0.013 pp over cap250 on the weighted historical mix. |
| Wednesday 16:07, completed | D2 early count plus spectral retrieval matched the kernel alone. |
| Wednesday 16:19, completed | Both F11 seeds scored. Causal sensor scaling lowered the full-system score on held-new PROTEUS people. |
| Wednesday 16:32, completed | F-C7 exact Scherer transfer. The learned selector did not improve cap250. |
| Wednesday 16:50, completed | F-C8 exact repeat of the fixed causal rule scored within the noise band on two additional orders. |
| Wednesday 18:00, decision completed early | Cap250 is the selection. A new official score requires upload authorization and a completed Codabench job. The organizer reported a large queue. |
| Friday 20 November, 18:00, planned | Final candidate and alternate. Use held-person and independent external evidence to break ties inside development noise. |
