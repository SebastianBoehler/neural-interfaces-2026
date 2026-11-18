# F18 energy-matched sensor withholding

F18 did not establish a gain from withholding local sensor neighborhoods during training. The local model scored 58.8513% and 58.9461% held-new WHOLE DEV with bounded inference. The exact hard-CE control scored 59.1613% and 58.8654%. The matched dispersed arm scored 58.9251% and 59.4642%. All three arms received untouched held EEG at evaluation. [Preregistration](../../docs/LEDGER.md#f18--local-sensor-withholding-during-raw-s1c-training)

| Whole-decoder balanced accuracy | Seed 0 ordinary | Seed 0 bounded | Seed 1 ordinary | Seed 1 bounded |
|---|---:|---:|---:|---:|
| Exact hard CE | 58.5458% | 59.1613% | 58.3882% | 58.8654% |
| Local withdrawal | 58.5490% | 58.8513% | 58.3664% | 58.9461% |
| Dispersed withdrawal | 58.5689% | 58.9251% | 59.2030% | 59.4642% |

| Paired contrast in percentage points [80% interval] | Seed 0 | Seed 1 |
|---|---:|---:|
| Bounded local minus hard CE | −0.3100 [−0.9226, +0.1827] | +0.0807 [−0.6780, +0.6403] |
| Bounded local minus dispersed | −0.0737 [−0.6990, +0.5365] | −0.5181 [−1.2348, +0.0592] |
| Bounded dispersed minus hard CE | −0.2362 [−0.8648, +0.2733] | +0.5989 [−0.0687, +1.2429] |
| Ordinary local minus hard CE | +0.0032 [−0.6201, +0.5029] | −0.0218 [−0.6467, +0.4532] |
| Ordinary local minus dispersed | −0.0199 [−0.6733, +0.6396] | −0.8367 [−1.5008, −0.2611] |
| Ordinary dispersed minus hard CE | +0.0230 [−0.6572, +0.5836] | +0.8148 [+0.1563, +1.4353] |

Both registered bounded effects remain undecided. Bounded local-minus-hard was +0.8871/+0.7084 pp for P04 and −0.2378/+1.3772 pp for P10. Bounded local-minus-dispersed was +0.8798/+1.1295 pp for P04 and −0.6897/−0.5952 pp for P10. The paired intervals resample people and delivery orders.

Both masks zeroed exactly three source sensors. They shared the highest-energy sensor in the local patch. The largest per-row energy mismatch was 0.005995 of total input energy. The largest of the twenty fold means was 0.000344. Across the canonical source mask plans, the marginal sensor frequencies differed by 0.134 total variation. The comparison controls mask count, shared anchor and removed energy. The two other sensor identities follow the assigned geometry. Clean held scores measure transfer from this training policy.

Ferranti source smoke 511200 passed in 14 seconds. Galvani precheck 3132552 verified all 20 source tapes without labels, and 3132545 passed the independent held smoke in 44 seconds. Ferranti 511201 and Galvani 3132546 produced all 60 checkpoint fits and 20 held banks. The exact hard-CE scores reproduced sealed F16. The replay passed 672 first-window agreements, 1,344 exact restarts, and a maximum truncated posterior difference of 2.314e−6. Collector 3132547 completed at 23:44:37 CET on 18 November 2026.

| Artifact in `ni26_friday_s1c_sensor_20261118` | SHA-256 |
|---|---|
| `actor.sha256` | `e7fd28145f16d25d8e1fecd7d47bcac0a3820e854d0e10b18d55b265e9ecf2a9` |
| `score.sha256` | `0c25034f1bed8694172036b28ef2fc12b6b0de306b6f51fb2ce01f98bc4cf5b6` |
| `precheck.json` | `8f78e045067f0e3a67d6039884c37438ef04a9cdf07496c4eee510d77d65af64` |
| `before_labels.sha256` | `7f1c7d25be687ae8cb98ca6d44f35d0f3e1f0a359e1dca209572f001c1d2e3df` |
| `whole_results.json` | `3b759900d0add003dd10b2eb832348deab3a6e61ec1343851f81f452d53fa9ba` |
