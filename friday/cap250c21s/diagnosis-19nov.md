# F-C21e independent accuracy veto and F-C21d stock-repeat diagnosis

Seven of nine independent C21e person jobs passed. The other two breached the preregistered 5e−5 protected upstream limit in the Sinc refit output. Job `3134282_3` observed 5.203640e−5 on E visit 2 at seed 10411. Job `3134282_7` observed 5.513963e−5 on its first session at seed 10411. Independent scorer `3134284` could never start and was cancelled. The result and pre-label lock files are absent. The separate paired runtime and cap evidence retain their own registered verdicts.

C21d `3134346` completed at **06:32:19 CET on 19 November** in 3m04s. The pinned RTX worker reproduced the cold archived stock frame and replayed E visit 1 before three independent untouched stock runs on the exact failed E visit 2 order. Each repeat saw 105 windows, 180 optimization steps over nine refits, nine relabels, five CSP refits and 15 fusions. No refit input row, prior pseudolabel, confidence weight or final decision changed in the three runs. Gold remained unopened.

| Warm stock pair | Maximum Sinc posterior difference | Maximum refit weight difference | Different final decisions |
|---|---:|---:|---:|
| 1 / 2 | 7.298820e−6 | 0 | 0 |
| 1 / 3 | **5.394570e−5** | 0 | 0 |
| 2 / 3 | 4.664688e−5 | 0 | 0 |

The untouched host reproduced a breach on E with the same motif and work. This supports base-worker numerical variation in that fixture. The original F-C21e veto remains in force and the original independent accuracy comparison stays unscored. C21s uses one locked stream of actual native inputs and separate outer states to answer a distinct question.

Diagnosis `ni26_friday_c21s_native_v1_20261119/diagnose.json` SHA-256 `aa7b74cc603df8eca114fbc6818e4a696df318bec7c727fe65e2fad7ad43eac5`. Worker source `d9a626eb1322fdda0d42d787142887389742be78437cdff5521dc015792ad647`.
