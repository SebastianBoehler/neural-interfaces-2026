# Frozen subtraction response on the Wang recordings

Both frozen archives emitted SUB more often during Wang subtraction than during the average of the two comparison tasks with a common-average reference. Neither showed a resolved difference against silent lyrics under that reference. A6's interval spans zero, and the cap250 result did not recur at the opposite block position. Six people and one task block per visit limit inference.

## Design and source checks

Six adults (02–07) performed serial subtraction, autobiographical recall and silent singing with eyes closed. Each task occupied a five-minute block. Subtraction appeared last, middle and first in visits 1, 2 and 3 respectively. Visit 2 followed visit 1 by about 90 minutes. Visit 3 followed about a month later. This design balances block position in the pooled contrast. A visit-specific difference still combines position, time and other visit effects. [Source protocol and retained Wang data](wang399.md)

The study used 54 version-pinned HTTP ranges (263,520,000 bytes). Each contained 40 seconds of continuous EEG. The four seconds starting at 150 s matched the prior §399 range digest and retained multi-channel samples in every block. Forty released channel names matched PROTEUS. The forty-first, FCz, was restored as zero in the acquisition reference. The released caps were harmonized. We cannot verify that individual apparatus physically measured every matching lead. No channel was reconstructed by scalp interpolation for this experiment.

We filtered each continuous segment at 0.5–45 Hz and resampled from 500 to 120 Hz. We retained six disjoint four-second windows from 142–166 s, leaving eight seconds of genuine EEG on either side. Reprocessing the same windows with four-second margins changed their participant-wide RMS by 0.47–0.73%. All six met the registered <1% requirement. This check concerns crop-boundary sensitivity for the specified filter. The competition uses different passband limits.

Each predictor received eighteen windows per actual visit, in two fixed shuffled orders. Each archive used the official fresh model at every visit. For original A6, the causal normalization statistics were computed separately in each visit. Its population-only control used the same weights. Intervals use paired deliveries, resampling six participants and then two delivery orders per sampled participant. Repeated windows inside a single sustained block do not constitute independent trials. Both comparator tasks differ from PROTEUS word generation. These data measure subtraction responses to a different experimental protocol.

## Acquisition-reference results

| Frozen system | Subtraction minus memory | Subtraction minus lyrics | Subtraction minus mean of both |
|---|---:|---:|---:|
| Original explicit A6 | −0.93 [−8.33, 6.94] | +0.46 [−6.02, 6.48] | −0.23 [−5.79, 5.79] |
| Same-weights population normalization | +0.93 [−3.70, 5.56] | −4.63 [−10.19, 0.93] | −1.85 [−5.56, 2.31] |
| Exact v5.1b | +6.48 [3.24, 9.26] | +9.26 [0.93, 17.13] | +7.87 [3.24, 12.50] |
| Exact cap250 | +3.70 [−0.46, 7.41] | +6.02 [−1.85, 14.35] | +4.86 [−0.69, 10.19] |

Entries are percentage-point differences in the fraction of windows decoded as SUB, with paired 80% intervals. Cap250 minus v5.1b was −3.01 pp [−5.56, −0.46]. Explicit A6 minus population-only was +1.62 [−4.63, 8.33]. For visits 1/2/3, cap250 gave +14.58/+4.17/−4.17 pp against the mean comparator. A6 gave +2.78/+6.25/−9.72. Its visit 3 interval [−15.28, −4.17] refutes a positive effect in that visit. Neither shared selectivity nor recurrence across opposite block positions met the prespecified rule. [Predictions summary and person-level results](subtraction429_result.json)

**The acquisition reference disabled one observation path.** The zero FCz coordinate triggered v5.1b’s flat-channel guard on all 648 delivered windows (324 unique × two orders). The package still emitted class predictions. Cap250 had no recognition, refitting or CP fusion within the eighteen-window visits. We therefore registered a paired electrical-reference control after collecting the primary results. We subtracted the common average of all 41 target potentials from each channel. Recovering the original potentials relative to FCz gave a maximum absolute error of 0.000016 µV.

## Electrical-reference qualification

Changing to a common average restored all 648 v5.1b observations. The paired change in the pooled SUB-call contrast was −1.39 pp [−3.24, +0.46] for v5.1b and +0.23 [−1.16, +1.62] for cap250. Both intervals fit the registered ±4 pp equivalence margin. Cap250 minus v5.1b was −1.39 [−4.63, +1.85] under the common average. This reference changed absolute A6 selectivity by +3.24 [0.00, 6.48]. [Paired reference changes](subtraction429b_reference.json)

| Common-average reference | Subtraction minus memory | Subtraction minus lyrics | Subtraction minus mean of both |
|---|---:|---:|---:|
| Original explicit A6 | +3.70 [−5.56, 13.43] | +2.31 [−4.17, 8.80] | +3.01 [−2.78, 9.03] |
| Same-weights population normalization | +5.56 [0.00, 11.11] | −3.70 [−7.41, 0.00] | +0.93 [−1.85, 3.70] |
| Exact v5.1b | +5.56 [1.39, 9.26] | +7.41 [−1.39, 16.20] | +6.48 [0.93, 12.04] |
| Exact cap250 | +4.17 [0.93, 7.41] | +6.02 [−1.85, 13.89] | +5.09 [0.23, 9.95] |

Subtraction appeared last at visit 1 and first at visit 3. Cap250's response against the mean comparator changed from +15.28 [7.64, 22.92] to −4.17 [−10.42, 2.08]. A6 changed from +4.86 [−0.69, 11.11] to −0.69 [−5.56, 4.17]. The reserve criterion and the joint A6/cap250 two-comparator criterion remain unmet. There were zero cap250 fine-tuning refits in either reference. [Common-average results and participant-level effects](subtraction429b_result.json)

## Reproducibility

Implementation revision `ff5ac58`. The Galvani directory is `ni26_wang429_v1_20261118`. SHA-256 manifests for the producer/worker/scorer, range provenance and processed participant banks are `0ae60b51ee92848dd361ddf8f941250fecbf96d6b2aa0d98db9eba6177b1396b`, `c10700362057148bd5e1592db4eec0779cea37956fe76249f280fde90c0b6e6e` and `49422dc5e231ecc469b22a57e98807bb16e54019299fb0041edd66553dd26a05`. Package SHA-256 values are `4a0bc427f2c9c862ee779db5766badeae93f783446561f4dc8abb43589933bef` (v5.1b) and `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed` (cap250).

Reference control revision `03a8834`, Galvani `ni26_wang429b_v1_20261118`, source manifest `bb8d703cd5c2a7d3292a0bdb509bc988d221c0d73931e3899ae65114dabc024d`. Both replay checks printed OK. The smokes and full predictions matched on the shared first visit and order. The first A6 smoke stopped at import because a helper shadowed an existing module. The renamed helper passed before full inference. Jobs 3124943, 3124953–54 and 3124960–63 covered the original reference. Jobs 3124985–89 and 3124995–96 covered the common average.
