# F-C21g · Keep stock during initialization and after recognition

19 November 2026, 06:40:20 CET · FULL-SYSTEM DEV · Authenticated complete prediction banks from F-C21 and F-C21r

The age-20 selection rule retained the full point gain from always-FBL2 on held-new within0 and 92.8% on held-new full0. Both paired 80% lower bounds versus stock stayed positive. On the four conditional returning visits, the rule agreed with stock on all 762 labels, as predicted from the earlier native evidence. [Registered rules](../../docs/LEDGER.md#f-c21g--protect-stock-through-initialization-and-native-recognition)

| Held-new order | Selected minus stock, pp [paired 80%] | Always-FBL2 minus stock | Selected minus always-FBL2 |
| --- | ---: | ---: | ---: |
| Within0 | +0.3832 [+0.1112, +0.7018] | +0.3832 [+0.1205, +0.7214] | +0.0000 [−0.1374, +0.1093] |
| Full0 | +0.4532 [+0.1324, +0.8435] | +0.4882 [+0.1901, +0.8779] | −0.0350 [−0.1106, +0.0434] |

The rule picked FBL2 on 2,375 of 2,655 windows in each held-new order. Actual Bio and contrast recognition stayed off. CountFilter latches were present and remained separate. Selected within0 predictions recovered 28 mistakes and lost 18 correct decisions. Selected full0 predictions recovered 42 and lost 30. Before the eligibility age, always-FBL2 made two rescues and two losses within0 and two rescues and one loss full0. The selector preserved stock decisions and terminal values on all rejected rows.

On returning full0, the eight eligible P10 visit-2 rows had identical stock and FBL2 decisions. The policy blocked all six unconditional FBL2 decision changes, including two rescues and three losses. Equal-cell, P04, P10 and 20/80 differences were exactly 0 pp. Original gallery and production auxiliaries remain conditional. The earlier returning result and the preceding forecasts already exposed this protection outcome. [Native returning evidence](../ea_returning/native_fbl2/README.md)

The selector reads only the two independently evolved complete current terminal outputs, current recognition and the session age. Its emitted decision feeds neither branch. Existing native or corrected replay receipts supply exact stock, identity, shadow, refit, prefix and delivery proofs. This experiment does not measure dual-tail runtime or establish independent external or private leaderboard behavior.

| Receipt | Galvani job | SHA-256 |
| --- | --- | --- |
| `source.sha256` | — | `a5369883c4d6c27c77e3a9aa3c56fe843a3886513d9ebcaa75ccc4661519d7b6` |
| `complete_select.json` | `3134375` | `ef38352c92e0969ba08564f7fef4ce768a7fd40a09fccb05a25d8ce9e9ff151e` |
| `complete_select_score.json` | `3134376` | `59bef3c831923f0a2ddf7d45f8878647a6703649d83c3b407da4b821fee7cbcd` |

Immutable release `ni26_friday_c21g_select_v1_20261119`. Read-only blind seal completed at 06:40:08 CET before the separate score opened gold.
