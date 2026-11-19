# F-C3 results — Track pseudo-label stability during neural refits

Keep the original cap250 loss weights. Downweighting revised pseudo-labels produced 48.2705% FULL-SYSTEM EXTERNAL accuracy versus 48.4557% for exact staged cap250. The change was −0.185 pp [80% −0.463,+0.116]. Both registered directional intervals overlap zero, so the candidate did not advance to independent held-person qualification.

| External full-system contrast | Change, pp (paired 80% interval) | Changed decisions |
|---|---:|---:|
| Primary minus stock | −0.185 [−0.463,+0.116] | 59 |
| Primary minus matched placebo | +0.165 [−0.069,+0.417] | 70 |
| Placebo minus stock | −0.351 [−0.585,−0.116] | 55 |

The primary fixed 19 stock errors and introduced 27 new errors. It changed accuracy by −0.370 pp and 0.000 pp in the two canonical full-session orders. Optimizer weights changed for all nine participants. Each repurposed exact stock refit preserved hard labels, total cross-entropy weight within each current pseudo-class, augmentation, Adam updates and downstream decisions up to the first modified refit. The paired warmed median stream-time ratio was 1.0011× stock.

All exact stock labels and activity counts agreed with the archived stage on 4,260 decisions from nine external participants, two days and two full-session orders. The warmed repeats changed one stock label for G and one placebo label for A. The primary warmed repeat produced all its canonical labels. Warm repeats were used to time replays, while canonical deliveries supply all accuracy measurements. The study uses an exposed external instrument and produces no held-public or private score.

Galvani smoke 3127173, nine paired cells 3127268 and collector 3127269 passed. Producer times ranged from 8m44s to 10m34s. Source manifest `8dd08259cbc95e8042544e3c893d7c53a354b9a851da97590520e16fb90cb92a` and blind decision receipt `5453a86cf856ca844534b5638bb45c69b671af40cc98bad11c746f2ca9a5b7b2`. [Full contrasts by participant and order](cap250-c3-result.json) · [Smoke and queue receipt](cap250-c3-queue.md)
