# Training on whole-window phase surrogates (§435)

On genuine EEG from ten excluded people, the static waveform decoder reached **44.781%** after replacing half of each source batch with phase surrogates. The all-real fit scored **44.120%**. The INDIVIDUAL-MODEL gain was **+0.661 pp** (paired cell-weighted 80% interval −1.020 to +2.496). Weighting people equally gave +1.244 pp (−0.295 to +2.716). Both intervals cross zero, so the preregistered verdict is undecided.

| Genuine held-person recall | All-real training | Half-surrogate training | Change, pp |
|---|---:|---:|---:|
| Motor imagery | 39.169% | 44.123% | +4.954 |
| Mental subtraction | 49.396% | 49.389% | −0.007 |
| Word generation | 43.795% | 40.831% | −2.965 |

The fits disagreed on 646 of 2,655 genuine windows. Seven people improved. P05 gained 7.426 pp and P09 lost 5.765 pp. P04 gained 2.599 pp and P10 lost 4.194 pp when each was excluded from training. The scores compare complete static decoders. Their session-independent decisions give the same paired effects in each shuffle. The frozen full stack scored 76.279% and JS scored 76.552% on these public people. Neither received the experimental decoder.

Each matched fit used seed 379, 360 updates and 96 source windows per update. Intervention RNG seed 435 drew fresh phase rotations for 48 windows per update. In the genuine source checks across folds, every electrode pair passed the spectral test (maximum relative error 6.8 × 10⁻⁷). Identity round-trip relative RMS error stayed below 1.9 × 10⁻⁷. Mean absolute change in channel kurtosis ranged from 0.46 to 0.77 on the eight source windows checked per fold. These rotations preserve per-channel spectra and pairwise cross-spectra. The time-domain amplitude distribution can change. The experiment measures generalization after assigning the parent window's class to a surrogate. The surrogate's task label has no independent observation.

Galvani A100 smoke 3125102, ten A100 folds 3125103, and CPU collector 3125104 completed. Code revision `57cb3c6`. The all-real fits reproduced every archived §379 static decision at all eight pinned orders. The CPU collector verified replay OK, both frozen reference endpoints, and the seals on the held-person predictions before reading evaluation labels. One seed, public new-person development only. [Results](phase435_result.json) · [Fold proofs](phase435_before.json) · [Training](../../studies/18nov/phase435/experiment.py)
