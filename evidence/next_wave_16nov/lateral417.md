# §417 Person-specific task asymmetry across interfaces

Across 2,655 public windows, 14 sessions and ten people, a person's task-related left/right band-power pattern recurred across BrainHero and Graz. Independently center each interface across people, average visits within people, and compare a person's normalized task-contrast vector to their own and all nine other people's vectors on the other interface.

| Cross-interface cosine difference | SUB−MI | WORD−MI | Both |
|---|---:|---:|---:|
| Own person minus mismatched people | +.564 [.459,.666] | +.444 [.293,.593] | +.504 [.384,.621] |
| Anatomical pairing minus mean of 16 random electrode pairings, on the preceding contrast | −.170 [−.289,−.058] | −.223 [−.378,−.078] | −.196 [−.326,−.073] |

Brackets give 80% intervals over the observed person contributions, conditional on the fitted interface centroids. One-sided reassignment of participant identities gave p=.00020 (5,000 permutations, plus-one correction). Both interfaces can share acquisition sessions. These measurements establish repeatability of spatial task contrasts on these recordings. The electrode randomizations undermine a specifically hemispheric interpretation. The measurements cannot distinguish subject physiology from acquisition effects.

## Held-person decoder and frozen endpoint

Train on six people. Evaluate P01/P02/P03/P09 on the eight pinned shuffled orders, weighting within-run, runs and full-session order families equally. The class-conditional diagonal Gaussian decoder evaluates two opposing asymmetric task patterns. Its prior uses only earlier windows from the same delivered session. Frozen B2_pool probabilities receive a 10% terminal probability blend.

| Outcome, % | Fixed native | Fixed reversed | Equal mixture | Causal mixture |
|---|---:|---:|---:|---:|
| Whole decoder | 40.360 | 35.377 | 36.428 | 36.428 |
| FULL-SYSTEM DEV | 81.246 | 81.258 | 81.225 | 81.225 |

The four-person frozen B2_pool endpoint scored 81.216%. Causal minus frozen was +.009 pp [−.060,+.084]. Eight of 6,056 order-deliveries changed. Causal minus native was −3.932 pp [−5.747,−2.117] whole and −.022 pp [−.043,0] full. Causal and equal mixtures made identical decisions across all 6,056 order-deliveries in both systems. The anatomy-specific posterior range within person/order was .0026–.0114; 23 of 32 ranges missed the initial .005 activity check. The random-pair control ranged .0092–.0772. Intervals resample people and orders. These exposed four people each contributed one session. Returning-person, independent-person and sealed effects are unmeasured.

## §417b Source temperature check

Six-person leave-one-person-out source loss selected multiplier 1 for both montages. Native loss increased from 1.0993 at 1 to 1.1029 at 2 and 3.0983 at 64. Sharpening the model therefore lacked source support. Both output banks had identical SHA-256, including every decision. The latent orientation model's weak activity remained after this source-only calibration. No submission candidate.

## Receipts

Registration and initial code 287057a; source calibration and corrected activity guard d5bdbed. CPU jobs 3124581/3124586 completed smoke/failed the activity guard before evaluation. Corrected smoke/full 3124596/3124597 and calibration smoke/full 3124598/3124599 completed, 9 min 40 s allocated CPU time including the failed run (two cores × 290 wall seconds). Replay printed OK. Pinned bank zero decisions, shuffled age/order, 29-window prefix, first-window prior and probability activity checks passed. Both prediction banks were sealed before reading the evaluation labels.

- [Primary result, per-person scores, control distribution and proof digest](lateral417_result.json)
- [Source calibration, losses and outputs](lateral417b_result.json)
- Both prediction banks SHA-256 `41b397b5736d994cf95a538e0427655a5da6d11fdeecd7677ad1b88bcebc2924`.
- Raw primary/calibrated CPU work directories on Galvani under `/mnt/lustre/work/gehler/gfh527/ni26_lateral417_v1_20261118` and `ni26_lateral417b_v1_20261118`.
