# F-C4 results — Balance neural refit loss weights

Keep the original cap250 weighting for the release. Balancing pre-recognition refit weights increased exact FULL-SYSTEM EXTERNAL accuracy on Scherer from 48.4557% to 48.5747%. The +0.119 pp estimate was positive in both orders, and the paired 80% interval spanned zero. The comparison with anti-balanced weights also remained undecided. The registered condition for independent held-person qualification did not trigger.

| External full-system contrast | Change, pp (paired 80% interval) | Changed decisions |
|---|---:|---:|
| Balanced minus stock | +0.119 [−0.162,+0.417] | 88 |
| Balanced minus anti-balanced | +0.261 [−0.139,+0.655] | 115 |
| Anti-balanced minus stock | −0.142 [−0.509,+0.231] | 94 |

Balanced fixed 37 stock errors and introduced 32 errors. Its gains were +0.053 and +0.185 pp on the two pinned orders. The paired median warmed runtime was 0.9934× stock. Stock reproduced all 4,260 archived canonical labels. Each full arm preserved all stage work counts over nine people, two visits, and both full-session delivery orders. Separate warmed repeats changed one stock label for G and two anti-balanced labels for D. They supply runtime, and the canonical deliveries supply accuracy.

All 356 neural refits had unequal cross-entropy mass across pseudo-classes. Balanced reduced the dominant pseudo-class share by a mean 3.006 pp. Anti-balanced increased it by 2.927 pp. Balanced shifted the mean shares for MI/SUB/WORD by −2.158/+2.058/+0.100 pp. All directions were checked at each actual optimizer call after the package's float32 weight cast. These are loss-coefficient masses, with realized parameter gradients also depending on waveforms and logits. F-C3's consistency change was absent throughout. [Coefficient results](cap250-c4-mass.json)

Galvani smoke 3127413, nine A100 paired producers 3127453 and score 3127454 all completed 0:0. Production took 9m00s–9m21s per person on A100 PCIe 40 GB. Scoring took 10 s, and historical replay printed OK. Manifest `be10f8ec3cb19aeb9faa94616343ab994685ac28198e4f63920274c0fe37f65d`, blind decision receipt `245e2510c7c05cf4fb8ec4ec5ff61c277cb78859b9660287d8f1a062fab71615`. The cohort had appeared in earlier external studies. [All contrasts](cap250-c4-result.json) · [Qualification and queue receipt](cap250-c4-queue.md)
