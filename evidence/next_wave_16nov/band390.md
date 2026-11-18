# §390 — Which training rows can causal personalization exchange safely?

Replacing only rows close to the confidence cutoff gained 0.086 and 0.067 percentage points over stock across ten new-person replays. Both paired intervals included zero. Across six people outside §384, the two seeds gained 0.120 and 0.053 points. All active arms replaced exactly 6,593 historical rows over 1,008 refits.

| Difference, official percentage points | Ten people, paired 80% interval | Six outside §384 | P04 | P10 |
|---|---:|---:|---:|---:|
| Near − broad, seed 386 | +0.150 [−0.027,+0.322] | +0.240 | +0.453 | +0.045 |
| Near − broad, seed 1386 | +0.064 [−0.104,+0.237] | +0.094 | −0.075 | +0.362 |
| Near − far, seed 386 | +0.197 [+0.044,+0.338] | +0.228 | +0.356 | +0.143 |
| Near − far, seed 1386 | −0.019 [−0.225,+0.161] | +0.016 | −0.069 | +0.491 |
| Near − nearest deterministic, seed 386 | +0.062 [−0.097,+0.223] | +0.063 | +0.080 | −0.086 |
| Near − nearest deterministic, seed 1386 | +0.042 [−0.143,+0.209] | −0.004 | −0.036 | +0.170 |
| Matched-band geometry − permuted | −0.046 [−0.214,+0.131] | −0.083 | −0.358 | +0.165 |
| Matched-band geometry − random | −0.001 [−0.096,+0.112] | −0.067 | −0.117 | −0.110 |

The six-person diagnostic explains how much confidence the method discarded. Added rows averaged 4.54/4.51 confidence percentage points below removed rows under the near selector. The broad selector lost 19.08/19.43 points, and far selection lost 36.72/36.77. The true correctness gap was 5.65/6.30 points near the cutoff, 20.63/21.37 for broad draws and 37.10/36.82 for far draws. This confirms the confidence intervention. The resulting classification gains remain uncertain. [Confidence and correctness diagnostics](band390_diagnostic.json)

## §390b independent-head agreement

When both near heads supported the same departure from stock, 166 decisions changed across the eight orders. They comprised 79 rescues, 67 harms and 20 cases where both decisions were wrong. Official gain was +0.086 pp [−0.012,+0.179]. Requiring count-filter confirmation gave +0.016 [−0.073,+0.120], with 42 rescues and 45 harms. The complementary rule departed from stock only when the count filter favored stock. It yielded +0.069 [+0.002,+0.130], with 37 rescues, 22 harms and five both-wrong decisions. The complementary result motivated §390c on independent random draws. Repeated orders reuse the same people and underlying windows. [Agreement results](agree390b_result.json)

The ten-person stock control reproduced 76.2793%. The zero-change, native labels, pinned orders, prefix and recognition-stop checks passed. In all 32 archived returning-person sessions the recognizer stopped personalization at window 10–63, before the first possible output at 90. This bounds the intervention to the new-person route for those recordings. All metrics use official cell weights, with people-equal and subset summaries preserved in the results. Jobs 3123614–18 and 3123672 passed. Source snapshot c6f9f3a964c928a68fb6506081a2e4d0d5541466ee41d5e981d50fab84a5a35f reuses the pinned §386 worker and paired-summary implementation. [Full paired results](band390_result.json)

## §390c independent-seed replication

The counter-filter result did not replicate on seeds 2390 and 3390. Its change against stock was −0.006 pp [−0.063,+0.044] across ten people and −0.026 on the six outside §384. Of 58 changed decisions, 24 were rescues, 23 were harms and 11 remained wrong. The official metric equalizes cells and classes, so its sign can differ from the raw rescue count. Near consensus across both seeds gained +0.005 pp [−0.069,+0.082], with 66 rescues and 68 harms. Reusing the same ten public people limits this evidence to seed stability. Jobs 3123804, 3123805 and 3123806 passed. [Replication results](rep390c_result.json)
