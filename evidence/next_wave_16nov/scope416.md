# Run-block delivery on Scherer

Cap250 had a positive point estimate against the uploaded v5.1b package under all three schedules. All three paired intervals included zero. Neither measured interaction met the preregistered ±1 percentage point equivalence criterion.

| Delivery schedule | v5.1b (%) | cap250 (%) | Difference (pp, 80% interval) |
|---|---:|---:|---:|
| Full session (§411b) | 48.677 | 48.717 | +0.040 [−0.857, +0.926] |
| Fixed run order | 49.021 | 49.484 | +0.463 [−0.265, +1.214] |
| Permuted run order | 48.757 | 48.899 | +0.142 [−0.701, +0.969] |

The mean blocked versus full change in the package difference was +0.263 pp [−0.645, +1.204]. Moving versus fixing the run order changed that difference by −0.321 pp [−1.402, +0.694]. Person D's blocked versus full change was +3.542 pp, and person C's was −1.458 pp. Two days and two orders per person limit the resolution of the conditional intervals. The [complete paired results](scope416.json) retain each person's change, each order and activity totals.

## Delivery and controls

Each of nine people contributed two days. Every day used one interface. The §351 independently reconstructed raw-trial manifest identified 142 runs containing 15 retained windows each across 2,130 rows. All five original task classes occurred five times in each raw run. Retaining HAND, SUB and WORD gives five windows of each class per run. Grouping those windows therefore changes both temporal locality and the class composition of short prefixes. The exported NPZ run field contains only zeroes. Source checks matched row identity, person, day and cue onset. [Raw-trial audit](external_semantics351.md)

Seeds 411 and 1411 used the same row permutations as §411b. Both blocked schedules inherited each run's relative row order from the corresponding full permutation. The second schedule randomized the sequence of these same blocks. For every person-day the three schedules were distinct bijections. No class labels entered the schedules. Both archives ran complete sessions one window at a time, retaining model state across runs and starting with an official fresh model at every day/order boundary. Each package repeated its first 17 smoke labels after reset. Cap250 reproduced the archived 31-label public reference. All blocked producer rows matched their pinned deliveries before scoring. Project replay printed OK, and every package/order matched the project implementation of the official cell-balanced metric.

Each schedule produced 36 cap250 streams, zero recognitions, 356 neural refits, 7,120 optimizer steps and 1,020 fused predictions. No stage reported an error. The bootstrap jointly resampled people, days within people and orders within days across schedules and packages, using 10,000 draws. External results use 19 native and 22 reconstructed channels and two orders on the same nine previously studied people. The competition has two interfaces and an unspecified shuffle scope. These results cannot distinguish a per-interface shuffle from a full-session shuffle on Scherer or determine a sealed score.

Smoke 3124578 and array 3124579 completed in 3,714 allocated A100-seconds combined. CPU scorers 3124580 and 3124583 completed. Source revision d68c61a; per-person report e3dc758. Galvani retains row predictions and activity in ni26_scope416_20261118. The combined 72-member manifest, frozen schedules and source digests are in the JSON report. [Exact archive identities](lunch_qualification385.md) · [Worker and analysis](../../studies/18nov/scope416)
