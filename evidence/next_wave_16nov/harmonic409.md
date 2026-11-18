# Harmonic phase consistency (§409)

Adding harmonic phase consistency to the complete decoder reduced four-person cell-balanced accuracy by 0.367 percentage points (paired 80% interval −0.709 to −0.047). The standalone decoder benefited relative to both phase controls on these four people, but the random-phase control was stronger in both reserved interface-transfer comparisons. Keep the frozen system.

| Evaluation, cell-balanced accuracy (%) | Stock | Power P | Harmonics X | Shift D | Random phase B |
|---|---:|---:|---:|---:|---:|
| Six source people, leave one out | — | 38.358 | 39.498 | 36.866 | 38.947 |
| Train BrainHero, test Graz in a held source person | — | 38.252 | 38.947 | 36.690 | 39.294 |
| Train Graz, test BrainHero in a held source person | — | 39.504 | 37.490 | 36.979 | 39.860 |
| Four development people, standalone | — | 40.625 | 41.266 | 39.900 | 39.804 |
| Four development people, complete decoder | 81.216 | 80.803 | 80.849 | 80.926 | 81.012 |

The standalone X−D and X−B differences were +1.367 [+0.456,+2.277] and +1.462 [+0.273,+2.651] percentage points. After 10% all-class insertion before the count filter and CP, these differences became −0.077 [−0.309,+0.146] and −0.164 [−0.313,−0.011]. X−P was +0.045 [−0.152,+0.256] in the complete decoder. The X−stock and X−B upper limits satisfy the preregistered refutation rule. In the six source holdouts X−D was +2.633 [+0.480,+4.806], while X−B was +0.552 [−1.436,+2.615]. The shift control alone would have overstated the evidence.

P uses twelve bipolar derivations’ mean spectral power at 4–36 Hz. X appends the absolute normalized bispectrum between each 4–18 Hz fundamental and its second harmonic. D cyclically shifts harmonic phase by three of seven frames. B independently rotates phase per frame and bin with seed 409. Both preserve the amplitudes. None of these tests localizes the underlying signal to a neural source.

All phases used source revision 53ed18f, manifest b2408941646e5ec47735579a1867b2389290b6b8c210ce66b08a83a6876ac982, and source snapshot ni26_harm409_v1_20261118. Slurm real-data smoke 3124284 passed. Extraction, fit, six source holdouts, reserved interface transfer, full smoke, eight orders and scorer 3124290–3124296 completed. Replay printed OK. Stock probability reproduction, independently pinned delivery, actor prefix/stop and active interventions passed. Intervals use paired people and orders (seed 381) or paired source people (10,000 draws, seed 409). Training people overlap across source folds. Development participants P01/P02/P03/P09 were previously used in other studies. Sealed transfer and returning-person routing remain unmeasured. [Per-person results and seals](harmonic409.json)
