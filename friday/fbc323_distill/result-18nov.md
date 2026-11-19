# F16 zero-feature S1C distillation

Both registered transfer contrasts remain undecided. Under bounded evaluation, trial-specific distillation gained 0.9702 percentage points against hard CE on seed 1 and lost 0.3359 points on seed 0. Trial-specific targets failed to separate from the matched derangement on either seed. The derangement reached 60.1155% on seed 1, the highest raw-only point estimate here. [Preregistration](../../docs/LEDGER.md#f16--trial-level-distillation-for-zero-feature-s1c)

| Held-new WHOLE DEV balanced accuracy | Seed 0 ordinary | Seed 0 bounded | Seed 1 ordinary | Seed 1 bounded |
|---|---:|---:|---:|---:|
| Hard CE, exact F15 zero-feature control | 58.5458% | 59.1613% | 58.3882% | 58.8654% |
| Trial-specific KD | 58.2576% | 58.8254% | 59.2244% | 59.8356% |
| Within-class/session deranged KD | 58.1007% | 58.6795% | 59.4812% | 60.1155% |
| F15 full-feature dropout teacher | 59.3990% | 59.9374% | 60.1509% | 60.5966% |

| Paired contrast, percentage points [80% interval] | Seed 0 | Seed 1 |
|---|---:|---:|
| Bounded trial minus hard | −0.3359 [−1.0131, +0.2125] | +0.9702 [+0.4365, +1.5182] |
| Bounded trial minus deranged | +0.1458 [−0.5453, +0.8020] | −0.2799 [−0.9023, +0.3411] |
| Bounded deranged minus hard | −0.4818 [−1.1464, +0.1437] | +1.2501 [+0.7053, +1.8548] |
| Ordinary trial minus hard | −0.2883 [−1.0683, +0.3240] | +0.8362 [+0.3849, +1.3149] |
| Ordinary trial minus deranged | +0.1569 [−0.5929, +0.8304] | −0.2569 [−0.7433, +0.3226] |
| Ordinary deranged minus hard | −0.4452 [−1.0308, +0.0755] | +1.0930 [+0.5143, +1.6492] |

Under bounded evaluation, the trial-minus-hard change was +0.1153/−0.1531 pp for P04 and +0.3742/+1.7666 pp for P10. Trial-minus-deranged was −0.6359/−0.8501 pp for P04 and +1.2623/+0.4118 pp for P10. The trial student remained 1.1120/0.7611 pp below its full-feature teacher with the bounded normalizer.

The F15 dropout teacher already fit the same source rows it predicted. Each teacher and all three students excluded the same held person completely. The scorer independently verified the within-class/session donors, preserving the exact teacher-target multiplicities of 1,958,400 padded source exposures. Both soft losses reached the raw body and classifier, and all students had exact zero fixed-feature gradients. Scored inference used the waveform representation and a constant zero vector without computing the fixed features. The hard CE actor reproduced all 20 F15 source state fingerprints and 84,960 held decisions. The held banks passed 672 first-window matches, 1,344 reset checks, and a maximum prefix posterior error of 1.794e−6. People and delivery orders were resampled in paired draws. This is a reused PROTEUS held-new evaluation with three equally weighted delivery schemes.

Ferranti smoke 511137 passed in 11 seconds. Array 511138 completed all 20 source jobs and 60 fitted actors. Galvani smoke 3131488 passed in 53 seconds. All 20 blind replay jobs in 3131489 passed, and collector 3131490 completed 18 November 2026 at 21:32:37 CET.

| Galvani release `ni26_friday_s1c_distill_20261118` | SHA-256 |
|---|---|
| `actor.sha256` | `33d13bad6fe04b0ef14d8fb4f91f5034cf56a8d928ed38b3a09eca82a4c8c0d5` |
| `score.sha256` | `503080a78e0c7b66f51e03451a93fd149613888dfa98328bce7c2795e63e018f` |
| `before_labels.sha256` | `299ba53d78116b2e015654cb90f0d43d1129bf11a0ad083a97a964fe60c41a0e` |
| `whole_results.json` | `e91c377bd9560d7c02f01adc9ccaa5e82b87d347927c19e17fd1d217780a803e` |
