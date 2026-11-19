# Returning full0 archived zero and causal input bridge

D-EAr0 replays the historical §431 `B2_pool` returning handler for P04 and P10 visits 2 and 3. The two fresh arms share the same D-EAr source fits. They differ only in whether slots 1 and 3 use the recording-order or causal means. The archived zero uses its original source fits and member inputs. [Preregistration](../../docs/LEDGER.md#d-ear0--archived-returning-full0-zero-and-fresh-causal-input-bridge) · [Fresh member banks](result-18nov.md)

## Archived zero

Galvani job `3131342` completed at 21:07:58 CET on 18 November. It reproduced the archived `base`, `ER`, and `T` probabilities bit for bit in their float32 representation on all 762 rows. The original three-eye/three-rest `T` output also matched exactly. Copying the archived slots through the replacement path and reversing the session blocks produced exact outputs and traces. The first attempt, `3131308`, stopped before the probability check because the diagnostic compared integer session IDs to person/visit names. Revision `c172e0b1` maps the names separately through global rows and keeps the actor's integer IDs intact.

| Session | First global row | Routed owner | First positive route age | Active refit ages | Past targets |
| --- | ---: | --- | ---: | --- | ---: |
| P04 visit 2 | 2153 | P04 | 10 | — | 0 |
| P04 visit 3 | 2383 | P04 | 10 | — | 0 |
| P10 visit 2 | 1238 | P10 | 28 | 20 | 20 |
| P10 visit 3 | 1326 | P10 | 10 | — | 0 |

Ages are zero-based. At the sole active refit, the reconstructed EM and TSEM state exactly reproduced the shadow's current emission. The exported targets match an independent truncated replay and contain only rows earlier than age 20. The original `gm.gate` obtains the age-20 gate value from the cumulative margin through age 19. The current row is used to check the reconstruction against the native emission.

## Matched fresh replay

Galvani job `3131393` completed at 21:10:26 CET. The live mixer consumed both new slots at weights 0.5 and 1.0 out of 15, respectively. Reversing the session blocks again reproduced each arm's outputs and traces exactly. Both arms retained all four routed owners and ages from the archived zero. P10 visit 2 retained the sole active refit at age 20. All 20 earlier-row posteriors passed the strict-past and truncated-prefix checks. The maximum posterior change was 0.03840349, with zero changes in the 20 hard pseudo-labels. The largest native emission reconstruction error was 1.12e−16.

| Live stage | Maximum absolute probability change | Decision changes out of 762 |
| --- | ---: | ---: |
| Slot 1, FBCNet EA mean | 0.45366832 | — |
| Slot 3, Shallow EA mean | 0.56395837 | — |
| Base mixer | 0.04169549 | — |
| Rebalanced mixer | 0.04202954 | — |
| Unrouted shadow after EM | 0.03788305 | — |
| Unrouted shadow after TSEM | 0.08382666 | — |
| `base` output | 0.10674643 | 11 |
| Three-eye/three-rest `ER` output | 0.04587403 | 9 |
| Three-eye/three-rest `T` output | 0.03590869 | 7 |

The probability and decision changes compare the two fresh arms directly. The live producer rebuilt routing and the own-space stages independently for each arm. The label guard verified the two original source masks, each excluding that person's visits 2 and 3 and retaining visit 1. Only the allowed fold source rows and matched public visit 1 supplied fitted labels. The replay did not open a held-gold metric.

## Release

The Galvani directory is `/mnt/lustre/work/gehler/gfh527/ni26_friday_ea_returning_bridge_v2_20261118/`. The shared source manifest SHA-256 is `e7c1512cec362a483cd3b28b317be4661e78e429131542ca599d95d5d21d1cb8`. The `index` column holds sorted global rows and `delivery` holds the `full0` global row permutation. The stage matrices use `index` layout. Each session's `rows`, `ages`, and `refits` arrays give its own delivery layout and refit grid. The independent bank read verified both SHA values, the common row map and all finite-target masks.

| Artifact | SHA-256 |
| --- | --- |
| `zero.npz` | `8e9395df097970abe0cd5343ef2a24698960e4aebcb9fd7ea7e7d5ca94c29fda` |
| `zero.json` | `710a5f89a0dcecd93d3b50af4f2a5c6bc8dd3494155a63bd30fc1dbdbef4e390` |
| `matched.npz` | `c8bdf3c5535f0d6445e2f7607838433061fdb08a187cbdd0dc70d8f57ae48c85` |
| `matched.json` | `1717d6d5e531719d70a4006d54973bec65d842fd7c0b8e91a7d1709d52760ca2` |

The release is a label-blind one-order input and routed-replay instrument. It establishes no returning balanced accuracy, regenerated cap250 model fit, exact worker qualification or new package.
