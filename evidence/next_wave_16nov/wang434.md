# Frozen subtraction responses on four new Wang participants

Both exact archives met the registered external selectivity criterion. Both emitted SUB more frequently during serial subtraction than during autobiographical recall or silent singing. This held against the comparator mean in both task schedules and in all four participants. These are external responses during sustained eyes-closed tasks. The experiment measures neither PROTEUS accuracy nor a sealed-score gain.

| System | Subtraction minus recall, pp | Subtraction minus singing, pp | Subtraction minus mean, pp |
|---|---:|---:|---:|
| Exact v5.1b | +18.06 [13.89, 22.22] | +6.25 [2.08, 10.42] | +12.15 [10.42, 13.89] |
| Exact cap250 | +20.14 [16.67, 23.61] | +8.33 [4.17, 11.81] | +14.24 [11.81, 16.67] |
| Explicit A6 | +11.81 [7.64, 16.67] | +9.72 [2.78, 16.67] | +10.76 [5.21, 16.67] |
| Same A6 with population normalization | +1.39 [−2.78, 5.56] | −1.39 [−6.94, 4.17] | 0.00 [−2.78, 2.78] |

Cap250 exceeded v5.1b by +2.08 pp [0.69, 3.82] against the comparator mean. On the previous six people the corresponding difference was −1.39 pp [−4.63, 1.85]. Explicit A6 exceeded its population-normalized control by +10.76 pp [7.29, 14.58] on the four new people. Each interval resamples people within the two fixed schedules and orders within people. Two people per schedule give limited coverage of individual variability. [Prior six](subtraction429.md) · [Full new participant and session measures](wang434_result.json)

## What the second schedule identifies

The release roster lists 57 people with subtraction last/middle/first over the three recording sessions. Two (22/23) have first/last/middle. One (01) has last/last/first. We took the first two standard-schedule IDs following the six scored IDs 02–07 and both rare-schedule IDs. We locked IDs 08, 09, 22 and 23 before retrieving EEG. Sessions 1 and 2 took place about 90 minutes apart on the same laboratory visit. Session 3 followed about one month later. [Pinned release roster](https://raw.githubusercontent.com/OpenNeuroDatasets/ds004148/1.0.1/participants.tsv) · [Wang et al., 2022](https://www.nature.com/articles/s41597-022-01607-9)

| Cap250 measure, pp | 08/09, last/middle/first | 22/23, first/last/middle |
|---|---:|---:|
| Subtraction minus mean comparator | +18.75 [13.89, 23.61] | +9.72 [8.33, 11.11] |
| Same response first minus last | −12.50 [−25.00, 0.00] | +8.33 [−16.67, 33.33] |

The reserve difference between these first-minus-last estimates was +20.83 pp [−12.50, 50.00]. The interval crosses both limits of the ±10 pp equivalence margin. Six of nine session-by-position combinations occur. An additive model with four participant intercepts has full rank (8 columns, 12 observations). For cap250 it estimates last minus first at +5.56 pp [−9.72, 23.61] and session 3 minus session 1 at −6.94 pp [−20.83, 8.33]. Neither contrast resolves an effect. Subtraction position still determines the complete ordering of all three tasks. Sequence assignment differs between people, and the complete session-by-position interaction cannot be estimated.

## Waveform and execution receipts

We fetched 36 immutable S3 byte ranges, totaling 175,680,000 bytes. Each was checked against its release pointer, full-file size and MD5 ETag, range headers, channel identity, units and exclusions. The filter-margin RMS difference was 0.40–0.59% per participant, below the 1% requirement. Forty observed channel names and the restored FCz potential were referenced to the common average and delivered as 41 × 480 samples. Each A100 archive ran with fresh state on 18 windows per genuine session, at seeds 429/1429. The original archive accepted all observations. Cap250 did zero fine-tuning refits. Repeated windows from one block are dependent.

Study implementation `6fe2825` uses the §429 session loader, A6 and exact archive worker, the §429 filter, and the §399 versioned request helper. Galvani directory `ni26_wang434_v1_20261118` contains the retained ranges, metadata and predictions. Worker/preparation manifest SHA-256 `b31f0e1c94e70ff6430e48d0b3e8f68433c6ff6f48620f5ad4d52043b0dafa2a`; scorer/CPU launcher manifest `5bc1dc059e6f93d9e2a3e2afc8f781b265bf83528fdbbc71bd35384f4e1fed92`. Roster SHA-256 `fefc3a5b0c4a28d3cb5370bb1d0ac9be2565ad70dbe8f94c82ff619460b01521`. The first input smoke stopped before transferring EEG because the BrainVision header omitted the fourth unit field. The corrected reader accepts this representation while requiring BIDS microvolt units and unit scale. Smoke 3125089 passed. Archive and CPU smokes 3125119–20 matched the full first session/order for ID 08. Arrays 3125121–23 completed and collector 3125127 verified pinned orders, completeness, classes, v5.1b activity, and the canonical replay's OK. No package change or upload.
