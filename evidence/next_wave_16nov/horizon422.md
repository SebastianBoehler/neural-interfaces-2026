# §422 Which part of cap250's longer history changes word selectivity?

Allowing optimizer steps increased the 60-versus-20-window horizon contrast by **1.603 percentage points** on 26 Shin2018 participants. The paired 80% interval across people was **−0.064 to +3.269 pp**. Both orders improved, by +0.641 and +2.564 pp. Fifteen people improved, seven declined and four tied. The direction and size met the point prediction. The interval leaves the population effect undecided.

| Runtime intervention in immutable cap250 | 60-window selectivity | Three 20-window blocks | Horizon gain relative to v5.1b |
|---|---:|---:|---:|
| Stock | 20.128 | 15.833 | +3.141 |
| Zero optimizer steps, scheduled prefix normalization retained | 18.526 | 15.833 | +1.538 |
| Entire neural fine-tuning branch disabled | 18.526 | 15.833 | +1.538 |
| Uploaded v5.1b reference | 14.808 | 13.654 | 0 |

All values are percentage points. Each acquisition run contributes equally. Selectivity measures the increase during word generation versus fixation in WORD response rate minus the corresponding increase in SUB response rate. The data reuse the participants, windows and delivery orders from [§415/415b](selectivity415.md). This external task has no three-class PROTEUS ground truth.

The stock optimizer changed 121 of 3,120 natural-session decisions compared with the zero-step intervention, 57 at deliveries 21–40 and 64 at 41–60. All 3,120 outputs agreed between the zero-step and branch-off interventions. The remaining relative horizon effect with the branch off was **+1.538 pp** [80% people 0.000,+3.077, rounded]. Its source among the other session states remains unresolved. Covariance personalization fitted 104 times per natural-horizon arm and emitted zero fusions. The stock count filter latched in two of 52 natural streams. Downstream processing used each intervention's actual posteriors, so the optimizer contrast includes its downstream interactions.

The stock and zero-step arms each performed 208 scheduled refits with prefix relabeling and normalization. Stock performed 4,160 optimizer steps, with per-stream maximum absolute weight change 0.0194–0.0199. Zero-step performed none and its measured weight change was zero in every stream. Each arm replaced the family member on 40 of 60 deliveries. The branch-off arm made zero replacements. All three short-horizon arms agreed on every output. All stock labels and counters matched the archived natural and shortened controls. Independent P01 and P26 repeats matched the producers. Local and cluster replay printed OK.

The paired bootstrap resampled 26 people and retained both orders for each person. Joint resampling of people and orders gives **−0.321 to +3.462 pp** for the optimizer contrast. The prior §415b point and all 26 per-person effects reproduced. The primary bootstrap seed was 422. Individual intervals resample each person's three acquisition runs with both orders paired and condition on the recorded inference histories. With three runs they are descriptive. [Paired results, decisions and counters](horizon422_result.json) · [Individual paired order estimates and run intervals](horizon422_people.json).

The §422b fresh-order reserve required at least +1.50 pp and a positive person lower bound. The lower bound was −0.064 pp, so the scheduler canceled its smoke, producers and collector before execution. There is no fresh-order result.

Worker revision faaa18a. Slurm smoke 3124676, producers 3124677, independent P26 repeat 3124678, person intervals 3124719, checksum receipt 3124705. Collector 3124680 stopped on an 8.9×10⁻¹⁶ pp rounding difference in the final additive identity. Collector 3124834 passed with absolute tolerance 10⁻¹⁰ pp. Label and activity checks remained exact. Reserve gate 3124695 exited with the registered no-release code 42. Conditional jobs 3124696–3124698 were canceled. The verified image and immutable package manifests match [§385](lunch_qualification385.md). Shin2018 input SHA-256 was 47ba7da2ea3ab5d70f3f9cd00de3998d7e27d1066dfb03711215ee48ba48c6ea. No archive edits or submissions.
