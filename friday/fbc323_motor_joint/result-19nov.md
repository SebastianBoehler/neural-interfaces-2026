# F32 · Concurrent motor supervision

19 November 2026, 07:33:49 CET. The motor objective improved the ordinary raw S1C decoder on both seeds. The preregistered bounded whole-accuracy and motor-recall gains remain **undecided** across seeds. Detached and exact hard CE produced identical native learned states, held decisions and scores. Standalone DEV on ten fit-excluded PROTEUS people, weighted across 28 cells and eight delivery orders. [Registered protocol and corpus terms](prereg-19nov.md)

| Inference | Seed | Hard and detached | Joint | Joint minus detached, pp [paired 80%] | Motor recall gain, pp [paired 80%] |
|---|---:|---:|---:|---|---|
| Bounded | 0 | 59.1613% | 59.4368% | +0.276 [−0.199, +0.790] | +0.265 [−0.732, +1.268] |
| Bounded | 1 | 58.8654% | 59.7572% | +0.892 [+0.205, +1.455] | +1.751 [+0.828, +2.640] |
| Ordinary | 0 | 58.5458% | 59.1194% | +0.574 [+0.024, +1.062] | +0.687 [−0.513, +1.637] |
| Ordinary | 1 | 58.3882% | 59.2179% | +0.830 [+0.158, +1.399] | +2.075 [+1.052, +2.913] |

Bounded subtraction/word recall gains were +0.083/+0.479 pp at seed 0 and +0.868/+0.057 pp at seed 1. Bounded P04/P10 whole gains were +0.413/−0.088 and +0.746/+2.369 pp. Full-order balanced accuracy improved on both full orders and both seeds in both modes. On bounded seed 0, three of the four within-stream orders declined. The positive ordinary intervals describe the registered secondary endpoint.

Each auxiliary arm used 619,200 external row exposures per seed across all source fits. Fixed source motor diagnostic accuracy rose from 40.67% to 50.04% at seed 0 and from 41.71% to 51.21% at seed 1. This measures learning on exposed source trials. Auxiliary spatial gradients were zero under detach and positive under joint training. Private normalization produced no original SessionBN writes, and both held inference modes used only original native weights. The real source smoke tested a 99-sample filter radius and a post-event tail drift of 1.907e−6. Independent held replay verified 672 exact first windows, 1,344 exact interleaved resets, 861,874 active bounds and maximum posterior prefix difference 1.585e−6. All hard predictions reproduced the sealed F16 bank.

Ferranti smoke `511647` and all twenty elements of `511648` passed. Galvani smoke `3134551`, all twenty `3134552` held elements and collector `3134553` passed. The collector authenticated and sealed the full held bank before opening labels. There were 60 native source fits and no competition submissions. The full source and held evidence is in the Ferranti/Galvani releases ending `s1c_motor_joint_20261119`.

| Artifact | SHA-256 |
|---|---|
| PhysioNet shard manifest | `b49b01b84ed61ad10865381a68c49d8e65b23ce99ed79527fc002071ad66a403` |
| Source actor | `f8d074e6867ebfda82383c164c257d64f154b251b0ccdbcb5153b297e7917723` |
| Held scorer | `49458dc244ca5d0583ab09aa4b84bca50ecce59f1d34777fb46d80d9e57e9762` |
| Blind held manifest | `ad428671148719a6d2e6c2cda39aaa8a7ffe27d348ea143e2668dcc30eeaa959` |
| Whole results | `1b9dd9d99b655d1dd5f1fff3655995d26593e6373bab6729fb64107b46daf5df` |
