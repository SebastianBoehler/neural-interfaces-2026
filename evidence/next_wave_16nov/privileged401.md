# §401 — What can the official EEG window learn from the rest of its trial?

The late source EEG carried repeatable information about the same trial. A classifier trained on the official window gained no clear advantage from the trial-specific late class probabilities compared with targets from other trials in the same recording and class.

## Class teacher and complete decoder

The registered trial-specific teacher used only the interval 5–7.5s after the cue, outside the official 1–5s input. The 1.5s overlap in the original raw export matched official waveforms within 0.00000763 µV and was excluded. Teachers trained on five people and predicted late probabilities for the sixth. Their mean cell-balanced accuracy was 40.06% over the six source people, ranging from 36.65% to 45.57%. This differed from §312's baseline-rest pretraining and §347's overlapping extra supervised crops. The held student consumed only the official window. [Source qualification and teacher outcomes](privileged401_source.json)

| Official cell-balanced difference in percentage points | Whole student, paired 80% interval | Complete B2_pool, paired 80% interval |
|---|---:|---:|
| Trial-specific − ordinary labels | +0.295 [−0.096,+0.686] | +0.134 [−0.054,+0.332] |
| Trial-specific − recording/class mean teacher | −0.928 [−1.327,−0.529] | +0.022 [−0.022,+0.067] |
| Trial-specific − recording/class donor 401 | −1.185 [−2.109,−0.260] | +0.013 [−0.031,+0.058] |
| Trial-specific − recording/class donor 1401 | +0.014 [−0.789,+0.808] | −0.022 [−0.087,+0.035] |
| Trial-specific − frozen stock | — | −0.059 [−0.270,+0.141] |

Whole trial-specific student accuracy was 45.058%. For the complete decoder, the 10% cognitive-pair mix preserved motor-imagery mass upstream, then reran the original count filter, personalization fits and fusion. It scored 81.157% versus 81.216% stock. Across all eight orders, SUB↔WORD errors increased from 420 to 430 while MI↔cognitive errors fell from 698 to 694. Mean SUB/WORD recall changed by +0.163 pp on Graz and −0.641 pp on BrainHero versus stock. Versus ordinary student training, those changes were +0.098 and +0.169 pp. The run/class donor controls preserve labels and interface. They cannot identify the origin of within-trial EEG differences. Jobs 3123972/73 and 3123982/83/84 passed the exact stock count-filter, CP score, native label and row-order guards. Four source-excluded people comprise the held cohort. [Paired results and confusion matrices](privileged401_result.json)

## §401b temporal pairing

Measured source feedback onset spanned cue+2.626–3.110s. A separate source-only diagnostic used equal-length EEG segments at cue+1.00–2.50s, +3.25–4.75s and +5.75–7.25s. It removed the physical recording/class mean spectrum and compared the early-to-late cosine similarity with 256 derangements within those same groups.

| Area | Before feedback − donor | After feedback − donor | After − before |
|---|---:|---:|---:|
| Frontal | +0.048 [+0.036,+0.061] | +0.029 [+0.004,+0.056] | −0.019 [−0.032,−0.006] |
| Central | +0.077 [+0.065,+0.088] | +0.063 [+0.040,+0.085] | −0.014 [−0.034,+0.003] |

Same-trial spectral topography persisted before and after feedback. The prediction that post-feedback signals would show stronger persistence failed. The six source people and overlapping physiological and artifact signals limit causal interpretation. CPU jobs 3124028/29 passed. [ROI, participant and permutation measurements](privileged401b_timing.json)

## §401c remove or amplify shared source variation

The top eight source cross-window directions had squared singular values summing to 2.450 for genuine trial pairings, versus 1.358 and 1.299 under the two recording/class-matched permutations. Each removal discarded exactly 8/96 of the variance in the source-only whitened official features. The student could therefore use the estimated directions at inference without receiving late EEG. [Source spectra](privileged401c_source.json)

| Removing the genuine shared directions versus | Whole classifier, pp | Full decoder, pp |
|---|---:|---:|
| Original feature metric | +0.707 [−1.514,+2.928] | +0.078 [−0.097,+0.242] |
| Amplifying those directions | +0.729 [−1.748,+3.205] | +0.132 [−0.022,+0.273] |
| Removing donor directions, seed 401 | −0.232 [−1.707,+1.244] | +0.187 [−0.003,+0.384] |
| Removing donor directions, seed 1401 | −2.026 [−3.461,−0.850] | +0.109 [−0.016,+0.231] |
| Frozen stock | — | −0.126 [−0.431,+0.131] |

The complete decoder scored 81.090% compared with 81.216% stock. Its cognitive-class recall changed by +0.358 pp on Graz and −0.187 pp on BrainHero versus the original feature metric. All four matched full contrasts favored removal on average, with intervals spanning zero and opposite interface directions against the original metric. The four exposed people provide too little evidence for a new package. Jobs 3124046/47 and 3124049/50/51 completed with the original score and order guards intact. [Paired results and interface outcomes](privileged401c_result.json)
