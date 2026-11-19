# D-EArEc · Recognition-gated Eye calibration

The base Eye ensemble uses the three original Eye nomix training recipes on 1,512 raw windows from eight public source people. Both P04 and P10 were jointly excluded from the Eye ensemble's training arrays, labels and running ocular statistics. Each balanced multinomial head then used only its own 189 or 192 first-visit calibration windows, all unseen by the base Eye ensemble. The head uses the actual frozen 128-dimensional embedding from each of three seeds and fixed L2 C=0.10. The verified logits were reproduced exactly on both first visits.

Both fresh arms mount the same three TorchScripts in the original Eye actor. The control forwards its observed evidence directly. The calibrated variant uses the actual native RouteB winning recognition identity and ramp. It forwards the same evidence exactly while the route has no eligible winner. When an enrolled identity wins, the variant blends the source posterior and the corresponding immutable personal head with calibration mass 0.5 times the native ramp. The result enters the real RouteB SessionEM Eye columns. The prefix verifies that neither the inner identity gate nor the native warm targets, updates, emissions, Rest output or outer probability mixture changes. The archived native production-Eye comparator retains its old training exposure.

| Galvani jobs | Artifact |
| --- | --- |
| `3133808 → 3133809` | Real A100 smoke and three original 60-epoch fits, release `ni26_friday_return_eye_cal_fit_v1_20261119`, source `5cd02b4a…4343`. |
| `3133815` | Two independently fitted source-only personal heads, release `ni26_friday_return_eye_cal_head_v1_20261119`, source `779d882d…833c`, receipt `b42dfef0…3b6c`. |
| `3133834 → 3133835` | Two native 31-window smokes then four whole held visits, release `ni26_friday_return_eye_cal_native_v1_20261119`, source `29238af7…9891`. |
| `3133846 → 3133847` | Separate blind seal and official score, release `ni26_friday_return_eye_cal_score_v1_20261119`, source `410e2135…6113`. |

The source smoke completed at 04:28:01 CET, all three full fits by 04:29:44 CET, and both independent heads by 04:30:25 CET on 19 November. The primary score compares calibrated and unchanged source-only Eye on the exact same three source models and held full0 orders. The original production-Eye score is a separately exposed reference. Rest, Core2, fixed classical/event priors and old Eye and gallery calibration remain in this instrument.

```bash
ssh mlcloud-galvani 'sacct -X -j 3133808,3133809,3133815,3133834,3133835,3133846,3133847 -o JobID,State,Elapsed,Start,End,NodeList'
```

Both genuine native 31-window smokes passed at 04:34:01 and 04:34:51 CET. The gate enabled the P04 calibration on 20 windows from age 10 and the P10 calibration on two windows from age 28. Native model logits and independent blend arithmetic matched exactly. The largest Eye log-evidence changes were 4.229824 and 2.179229. The largest terminal probability changes were 0.0142520 and 0.00075418. Both before-recognition sequences and both interleaved resets were bitwise exact. P10 retained all 20 native warm optimizer steps. Receipts `00d7191362fb6303432eeca2372839f8b7b3b784d5347dd0bab2adb88fa382ef` and `75f7e63faa73c40fa3f5064a48ec676cf9559794280e41f0f9d9b4ff5ed5bda7`. These smokes opened no held labels.

## Complete returning result · 19 November 04:51 CET

FULL-SYSTEM DEV on both later visits of two reused returning participants. Each value below uses the same source-only 1,512-row Eye ensemble for both columns.

| Scope | Source-only Eye | Calibrated Eye | Calibrated minus source-only and paired 80% interval |
| --- | ---: | ---: | ---: |
| P04 | 80.43011% | 80.95094% | +0.52083 pp [+0.26042, +0.78125] |
| P10 | 91.90880% | 91.63998% | −0.26882 pp [−0.40323, −0.13441] |
| Equal official cells | 86.16945% | 86.29546% | +0.12601 pp [−0.26882, +0.52083] |
| Returning 20/80 | 89.61306% | 89.50218% | −0.11089 pp [−0.40323, +0.26042] |

Calibration rescued two P04 decisions and lost one P10 decision. Both aggregate paired intervals cross zero. Calibration reached 702 of 762 rows through the real native route, the unchanged recognition identities selected both heads correctly, and all four visits passed exact before-recognition, optimizer, inner and outer mixture, and session-prefix checks.

The calibrated variant trailed the original production Eye by 3.68133 pp on equal cells [−4.73964, −2.72907] and by 3.20329 pp under 20/80 [−4.42857, −2.72907]. The original Eye's training included these held visits. The comparison isolates the calibrated versus unchanged source-only actor, while the production comparison measures different Eye training exposure and fitted weights. All arms retain the original outer EM and historical gallery calibration plus the other production-derived layers.

Job `3133846` sealed all complete predictions at 04:51:37 CET, before separate metric job `3133847` completed at 04:51:49 CET. The read-only blind receipt, prediction bank and score hashes are `a576694f35447b5c0e37639673e582b740ba34b806246ff3daf808d2029bc3e7`, `fbf7b5f52cf32327def9a8aaee9a02465e9bad224ea2cc1c10b2d1f9b2cc61fa`, and `54c7b7dfdf3488dce061fdbd526e6237c0e04d4f689e44770789f5c420abcde0`.
