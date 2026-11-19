# F-C21 · Corrected CC30 and adapted FBLight factorial

19 November 2026, 05:55:05 CET · FULL-SYSTEM DEV · The same ten held-new PROTEUS people in corrected within0 and full0

Doubling the actual adapted FBLight family gained **+0.3832 pp** within0 and **+0.4882 pp** full0 against exact corrected stock, with positive paired 80% lower bounds in both. The joint CC30 replacement plus FBLight doubling also beat stock in both deliveries. It trailed FBLight-only within0 by **0.0328 pp**, so the registered joint advancement condition failed. [Preregistration](../../docs/LEDGER.md#f-c21--requalify-the-318-cc30-and-adapted-fblight-factorial-on-corrected-deliveries)

| System or paired contrast | Within0 | Full0 |
|---|---:|---:|
| Stock balanced accuracy | 78.3241% | 73.7390% |
| CC30 swap, SS | 78.2518% | 74.2812% |
| Adapted FBLight double, FBL2 | 78.7073% | 74.2272% |
| Joint, JS | 78.6745% | 74.4367% |
| SS minus stock, pp [paired 80%] | −0.0724 [−0.4019, +0.3430] | +0.5422 [+0.1270, +1.0105] |
| FBL2 minus stock | +0.3832 [+0.1205, +0.7214] | +0.4882 [+0.1901, +0.8779] |
| JS minus stock | +0.3504 [+0.0894, +0.6082] | +0.6978 [+0.5329, +0.8836] |
| JS minus SS | +0.4227 [+0.0549, +0.6771] | +0.1555 [−0.2673, +0.5379] |
| JS minus FBL2 | −0.0328 [−0.3059, +0.2023] | +0.2096 [−0.1374, +0.4749] |
| Interaction | +0.0396 [−0.4842, +0.4135] | −0.3327 [−0.9222, +0.1017] |
| Gold-only stock/JS ceiling minus stock | +1.2908 [+0.9142, +1.6878] | +1.9773 [+1.6028, +2.2995] |

The intervals resample people 4,000 times over 28 official cells per order. The label-aware ceiling uses gold to choose between stock and JS per window. It bounds those branches and gives no estimate for a deployable rule.

| Person | Within0 SS / FBL2 / JS, pp | Full0 SS / FBL2 / JS, pp |
|---|---:|---:|
| P01 | −0.0168 / +1.5625 / +1.0249 | +3.1418 / +2.6210 / +2.0833 |
| P02 | −0.5040 / +0.0515 / −0.5040 | +1.0417 / 0 / +1.0417 |
| P03 | +0.5208 / −0.5208 / −0.5376 | +0.4861 / 0 / −0.0347 |
| P04 | −0.0007 / +0.0052 / +0.7284 | −0.5264 / +0.5264 / +0.7064 |
| P05 | +1.5625 / +2.6546 / +1.6129 | −0.0168 / 0 / +0.5544 |
| P06 | 0 / 0 / −0.5556 | +0.5376 / +1.6140 / +0.5376 |
| P07 | 0 / 0 / +0.5724 | +1.0417 / +1.0237 / −0.0347 |
| P08 | +1.0417 / +1.1111 / +1.0764 | +0.5556 / +0.5556 / +1.0417 |
| P09 | 0 / 0 / 0 | −1.0955 / +0.5208 / +0.5208 |
| P10 | −1.2051 / +0.1634 / +0.0102 | +1.1595 / −0.3600 / +0.6465 |

FBL2 rescued 30 and 44 mistakes, with 20 and 31 regressions. JS rescued 34 and 53 mistakes, with 25 and 34 regressions. SS had 28 and 51 fixes, with 30 and 36 regressions. The historical §321 joint returning result remains adverse at −0.2879 pp on two people under conditional auxiliary exposure. That study did not test FBL2 alone. [Historical ledger](../../../neural-interfaces-2026/docs/LEDGER.md)

## Actual mix and tail work

All sixty §304 held-person fold files reproduced the six order-specific seed streams bit for bit. Each fold authenticated nine source people and active class-conditional covariance penalty. SS and JS removed the full outer FB family including its established warm delta. FBL2 and JS doubled the actual post-warm FBLight values, which differed from its frozen counterpart on 2,375 windows in both orders. Maximum FBLight warm movement was 0.8381 within0 and 0.7421 full0. FBL2's maximum actual mix movement was 0.03749 and 0.03953. Joint movement was 0.05534 and 0.06288.

P0r and its inner targets stayed exact, with the original four capped warm banks unchanged. Full stock and identity reproduced raw, count-filter and terminal outputs bitwise. All five arms reran actual count and strict-past CSP. Count changed 2,005 rows within0 and 308 full0 in every arm. Each complete arm performed 126 CSP refits and 1,395 fusions. The scorer waited for both sealed banks. The 7m04–7m09 job times measure the CPU replay and five paths, outside any deployment timing qualification.

The first smoke `3134062` stopped before predictions because Python resolved an adjacent experiment's same-named worker. The immutable retry bound the actual authenticated tail module by path. Code `3713c20c`, import correction `0015dc09`. Remote release `ni26_friday_c21_corrected_v2_20261119`.

| Artifact | Job | SHA-256 |
|---|---|---|
| `experiment.sha256` | — | `5b15295e48b605f87ab4855d1557393da09d0df56e67b92d1b033865de518747` |
| `scoring.sha256` | — | `df353daad29c574bc4d7ca4071d60caa6c9dcfe6728ea267f86995c1ba68eb8a` |
| `smoke.json` | 3134070 | `a23bffba1c60ab0c96e9e4ddb44104f8a0084f1e69f322a70a2229f351244ed8` |
| `within0.json` | 3134072 | `c41f6df763d86dee454db348560fce1943620f6ed3a3a1ef8824d4c54ac38472` |
| `within0.npz` | 3134072 | `109a572de0f062a62ddf69d80f847d5a107e33236b45f34d7984af303df6df45` |
| `full0.json` | 3134071 | `442ba868279c332e550bc4cffbd29f372c1e1afdf7fb8d6e86fe1a548fd6f331` |
| `full0.npz` | 3134071 | `5b9da9b5ba865c4b0b6bcdd2251536a472884d561bda85cfb4e624358f3f5650` |
| `score.json` | 3134073 | `cfd9584c29d3e9175228f6f92c633a09d63f91e20d88c6ae722c3e726a58f939` |
