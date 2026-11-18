# Electrode coverage and cross-study transfer

More scalp coverage improved WORD/SUB ranking when fit and test came from the same instrument. Wider direct supervision failed to improve either transfer direction. In contrast, allowing each Shin source to use the electrodes it actually shared with PROTEUS improved its combined semantic direction. No returning cohort or official three-class score enters these results.

| Source → test | Posterior four | Mean random four | Native 19, EDF 18 | Change vs posterior four (80% interval) |
|---|---:|---:|---:|---:|
| Scherer → other Scherer people | 55.44 | 51.95 | 60.31 | +4.86 [2.17, 7.62] |
| PROTEUS → other PROTEUS people | 60.13 | 58.69 | 63.93 | +3.80 [0.02, 7.49] |
| PROTEUS → Scherer | 54.45 | 50.96 | 54.04 | −0.41 [−3.30, 2.35] |
| Scherer → PROTEUS | 55.21 | 49.47 | 50.97 | −4.24 [−6.39, −2.14] |

Values are equal-person within-run WORD-above-SUB ranking AUC, percent. Matched arms have exactly 18 effective ridge degrees of freedom, computed solely from source eigenvalues. A 19-channel control using only 36 leading source components at the same effective dimension reached 59.58/64.56/52.93/52.01% in table order. Using the original four-channel ridge parameter directly gave 57.69/65.32/55.00/50.50%. An eight-draw random subset average is a control for electrode count; the draws do not represent independent subjects. The four native posterior leads were Cz, P3, P4 and Pz. At the source decision midpoint, wider within-corpus fitting increased WORD recall by 6.69/7.97 pp on Scherer/PROTEUS and reduced SUB recall by 1.88/6.36 pp. Across studies these operating-point effects reversed. [Complete paired results and all eight subsets](coverage397_result.json)

## Source-specific calibration

The Shin2017B serial-subtraction task shares six native channels with either target (the original four plus F3/F4). Shin2018 verbal fluency shares six with Scherer (the four plus C3/C4) and 16 with PROTEUS. Source person-held arithmetic/baseline AUC grew from 59.10 to 64.48%. Word/baseline grew from 64.50 to 67.31% using six and 70.32% using 16. Each active-minus-baseline direction was calibrated by the standard deviation of its own source trial-pair differences. The original uncalibrated four-channel predictions separately reproduce §393.

| Shin-derived WORD-minus-SUB AUC | Scherer, 9 people | PROTEUS, 10 people |
|---|---:|---:|
| Same four native electrodes, source calibration | 59.19 | 50.97 |
| Expand arithmetic alone | 55.74 | 52.09 |
| Expand verbal fluency alone | 58.24 | 53.00 |
| Expand each to its actual native overlap | 56.52 [54.30, 58.75] | 54.97 [51.77, 57.94] |
| Both expanded minus four | −2.67 [−4.11, −1.14] | +4.00 [0.51, 7.61] |

The arithmetic expansion accounts for much of the Scherer loss (−3.45 pp [−4.68,−2.18] in the combined score). The verbal expansion gave the clearest PROTEUS gain (+2.04 [0.34,3.73]); the arithmetic gain alone was +1.12 [−1.58,3.81]. Both targets saw exactly the same trials in all comparisons. The earlier inferior PROTEUS result reflected limited electrode overlap in part. Performance also depends on source task and acquisition protocol. These same participants have been inspected in related studies, and intervals condition on the chosen source fit.

## EEGDash v0 metadata audit

The catalog has 77 named studies. I scanned the title and paradigm manifest and checked primary protocols of plausible cognitive candidates. I found no additional verified match for both covert letter-cued word generation and trial-cued serial subtraction on a wide measured montage. The owner's table is public. The sample store requires separate access. This review downloaded metadata only. [EEGDash v0 catalog](https://eegdash.org/corpus/v0.html)

- **Wang ds004148** is a strong partial match. Sixty people performed silent serial subtraction by sevens from 5,000. The first subtraction recording's BIDS channel file has 61 native EEG channels at 500 Hz and 40 of 41 PROTEUS leads. FCz served as acquisition reference. Each task occupies a continuous block, with no repeated competition-like visual cue. Its memory task is autobiographical recall. It could independently test a block-level arithmetic direction if probe and eye-state conditions are controlled. The project had catalogued it on 14 November, and I found no scored use in the ledger. [Primary paper](https://www.nature.com/articles/s41597-022-01607-9) · [Original channel and event files](https://github.com/OpenNeuroDatasets/ds004148/tree/master/sub-01/ses-session1/eeg)
- **PEERS ds004395** has arithmetic distractors and verbal retrieval within sessions. The participant types sums of three displayed digits, then recalls studied words aloud. Typed responses, external memory demands and vocal muscle activity undermine a covert WORD/SUB interpretation. **ArEEG ds005262** uses five fixed Arabic inner-speech cues on eight electrodes (seven overlap PROTEUS, Oz is absent there). There is no serial arithmetic condition. [PEERS primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12161444/) · [ArEEG primary paper](https://www.nature.com/articles/s41597-025-05387-w)
- **ds005697** uses six seconds viewing and six seconds imagining a displayed image. **ds003969** alternates meditation with generic active thinking. **ds005121**, whose v0 task field is unspecified, records sleep during targeted memory reactivation. These datasets provide neither target task pair. [PerceiveImagine owner description](https://github.com/OpenNeuroDatasets/ds005697) · [Meditation protocol](https://github.com/OpenNeuroDatasets/ds003969) · [Sleep study](https://github.com/OpenNeuroDatasets/ds005121)

Galvani 3123855/3123856/3123857/3123858 completed, covering 19 within-instrument person exclusions and two frozen cross-instrument fits. Source manifest SHA-256 de882d62f9585b4cfb83b7b5ef3caebbc0abb30eed7b410ad8df735fa86c4565.

## Direction specificity

On Scherer the wider W−A direction at 56.52% exceeded the 53.31% 90th percentile of 128 matched source-pair sign permutations. None of the 128 null scores reached the observed value. W+A generic activation reached 49.17%, paired difference +7.35 pp [3.79,10.88]. On PROTEUS W−A reached 54.97% against 49.32% for W+A, paired difference +5.65 [0.93,10.09]. However, 26 of 128 sign permutations reached or exceeded the actual PROTEUS value and the null 90th percentile was 56.91%. A target-person interval above 50% and a source-label permutation test answer different questions. Wider overlap improved PROTEUS ranking against the real four-channel comparator, but source semantic specificity across both instruments remains unconfirmed. [Full null distributions](coverage397b_result.json)

The frozen true scores reproduced §397 before interpreting any control. Galvani 3123916/3123917/3123918 passed. The §397b source manifest SHA-256 is a1d05b834d3fa5ff45c7fa94758bfb55cb3c642dc2afc290d1d59fc60fa7b74e.
