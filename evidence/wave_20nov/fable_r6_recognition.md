# Fable R6 Bet 1 · earlier recognition · 21 November

**Stage 1 stopped the bet at 16:53 CET.** The delivery-5 true-identity splice gained **+0.117 pp** at 50/50 [paired 80% −0.049, +0.290], below the prespecified +0.30 pp. Fit and held-out orders gained +0.134 and +0.099 pp respectively. The stop closed the soft-mix fit and package qualification. [Preregistration](../../docs/LEDGER.md#fable-r6-bet-1--early-soft-recognition-registered-21-november-at-1630-cet-before-the-full-order-experiment)

## Gain by first substituted delivery

Exact em_half reference `833a28721b0de0fa34f94de0328284ca9643cbaccc2dd343926f19c254949a4d` scored 77.931% at 50/50. Each splice starts at the stated delivery and ends before the native latch. Oracle outputs use true identity and a personal path warmed from delivery 1. Held-new decisions are identical by construction. Intervals resample people and then orders with 20,000 paired draws and official class-balanced cell weights. Fit and held-out orders reuse people. [Base qualification](bottleneck_review_21nov.md#recognition-gated-sessionem) · [Last Push replay](last_push_21nov.md)

| Start delivery | 50/50 Δ pp [80%], all 16 orders | Fit / held-out Δ pp | Returning P04 / P10 Δ pp |
| ---: | ---: | ---: | ---: |
| 1 | +0.127 [−0.088, +0.353] | +0.153 / +0.101 | +0.667 / −0.159 |
| 3 | +0.122 [−0.063, +0.316] | +0.135 / +0.110 | +0.599 / −0.110 |
| 5 | +0.117 [−0.049, +0.290] | +0.134 / +0.099 | +0.547 / −0.081 |
| 8 | +0.063 [−0.049, +0.181] | +0.076 / +0.049 | +0.332 / −0.081 |
| 12 | +0.037 [−0.010, +0.095] | +0.050 / +0.024 | +0.166 / −0.017 |

## Native recognition and impostor crossings

All 64 returning streams latched correctly. P04 latched on delivery 11 in 25/32 streams, then at 13, 15, 17, 18, 31, 38 and 41 once each. P10 latched on delivery 11 in 26/32 streams, then at 13, 14, 15, 16, 19 and 25 once each. The median was 11 for both. The signed reference misrecognised 1/224 held-new streams (0.45%), P10 visit 1 as P03 on fit order 71802 at delivery 40. It misrecognised 0/112 in the held-out order block.

The archived capture lacks the cumulative statistic. Replaying the signed IdentityGate on the pinned EEG and verified fold galleries recovered the native first latch timing on all 288 streams. Both diagnostic rules require the same winner above the threshold on three successive deliveries. The pooled thresholds use all orders to achieve zero impostor crossings by construction.

| Pooled threshold | First correct delivery | Correct by 5 | Correct earlier than native | Oracle splice 50/50 Δ pp [80%] | Fit threshold on held-out impostors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global, strictly above 0.273074 | 4 | 12/64 | 36/64 | +0.046 [−0.007, +0.099] | 1/112 |
| Separate threshold per delivery | 3 | 37/64 | 61/64 | +0.075 [−0.032, +0.184] | 10/112 |

Both pooled thresholds eventually crossed on all 64 genuine streams with the correct identity. The fit-only global threshold admitted P10 visit 1 as P03 at delivery 7 on held-out order 71814. The delivery-specific fit rule also admitted only P10 as P03, including three crossings by delivery 5.

## Checks

Local and Leonardo v3 replays printed `OK`. Leonardo smoke 65040479 matched the live half-weight patch and the fresh causal prefix on P04's 41-delivery example. All eight capture tasks in 65040484 completed, with the last ending at 16:51:21 CET. They matched 836 live native and em_half deliveries against the pinned references across 64 genuine prefixes and captured 772 pre-latch oracle decisions. The overlapping stored oracle decisions matched through delivery 20, before half-weight SessionEM can change emissions. The inactive splice was exact, and a direct pass of the pinned official balanced-accuracy function reproduced all cohort/time points. [Capture](../../scripts/bottleneck_21nov/r6_recognition_capture.py) · [Scorer](../../scripts/bottleneck_21nov/r6_recognition.py) · [Statistic](../../scripts/bottleneck_21nov/r6_recognition_stat.py)

Local result `work/fable_r6_recognition/stage1.json` SHA-256 `00d7b814eab2b0e565949288b7f5b9bf0bf058944e4ad30b240abb135b552f78`.
