# F-C2 results — Amplitude-normalized personalizer fits

Keep the original cap250 spatial fit. Normalizing covariance power before fitting the CSP spatial filters changed 210 held-new decisions. Corrections and regressions were equal, and the net full-system change was −0.004 pp. The registered external check required a positive primary point estimate and at least 20 changes. The change count passed and the score condition did not.

| Historical FULL-SYSTEM DEV, held-new people | Score | Change versus stock, pp (paired 80% interval) |
|---|---:|---:|
| Three-eye/three-rest stock | 76.1039% | — |
| Normalize covariance power for CSP fit only | 76.1003% | −0.004 [−0.081,+0.086] |
| Normalize covariance power throughout personalizer | 76.1217% | +0.018 [−0.111,+0.185] |

Fit-only minus throughout was −0.021 pp [−0.152,+0.084]. The two registered directions remain undecided. Fit-only changed held-new P04/P10 by −0.012/−0.132 pp. The historical returning-role bank records that recognition stopped personalization before it could emit in all 32 streams. Preserving that stop leaves zero eligible returning decisions. The descriptive 56.6/43.4 mixture for fit-only was −0.002 pp.

Galvani smoke 3127102 and all eight cells in array 3127133 passed. The smoke changed real fitted spatial eigenspaces, matched the stock selection and scores, and preserved prefix outputs through age 108. Full order cells reproduced all 21,240 stock score vectors and used the same selected rows, refits and counts in every arm. The earliest change was after personalization began. Replay printed OK. Collector 3127134 passed in 2 s. Source commit `42dc4fb`, manifest `de8aefe5e6c3deaba759653a2cfb890313539394862a96d4b6b826eb30ad2b5d`. Decision receipt `8d36a01adc205cc02c883d83cb42a7b28c4d2cdc3b2ade3f38f3cf691770d685`. [Paired results](cap250-c2-result.json)
