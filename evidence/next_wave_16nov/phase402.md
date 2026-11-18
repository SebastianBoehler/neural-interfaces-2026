# §402 Within-window signed phase slope

A consistent delay between two sensors rotates their cross-spectrum as frequency rises. For each oriented electrode pair, the decoder averages the imaginary part of the conjugated coherency at frequency f times the coherency at f+1 Hz. This signed statistic does not identify the neural origin of a delay.

The per-window inputs are common-average-referenced official EEG only. P uses band log power. I adds signed band-mean imaginary coherency. X adds signed slope to I. D replaces only the slope with a surrogate, rotating each electrode-frequency coefficient independently across electrodes and frequencies while keeping the same rotations across the seven frames. The surrogate preserves individual powers and cross-spectral magnitudes. The same input always receives the same surrogate. The rank and geographic coverage of the twelve pairs are fixed.

The same RBF ridge estimator is fitted on six source people for each three-class member. Held people are P01/P02/P03/P09. Each member contributes 10% of the conditional subtraction/word probability before the B2_pool count filter and causal personalization. The incoming motor-imagery probability stays fixed at this insertion. The four-person cohort provides no returning-person estimate.

[Preregistration](../../docs/LEDGER.md) · [Experiment](../../studies/18nov/phase402/experiment402.py) · [Source and member hashes](phase402_source.json)

Galvani jobs 3124092–3124096 cover the feature smoke, source fit, replay and actor smoke, eight order actors, and scorer. Source manifest SHA-256 9298835a36bcbcfc5df5ca22bc09cddb23335170f7a70eb8096db9ce3657a388. The local real-window smoke verified sign reversal under reversed electrode pairs, surrogate magnitude preservation, and active phase destruction (maximum change 0.623).

All jobs completed. The v3 reference printed OK. The unmodified actor recreated the archived count-filter and causal-personalization outputs, including their exact probabilities. Eight independent pinned order checks passed. The scorer sealed the complete order grid before reading query labels.

| Four held people, balanced accuracy (%) | Power P | Imaginary I | Signed slope X | Phase-destroyed D | Frozen stock |
|---|---:|---:|---:|---:|---:|
| Individual model | 39.912 | 41.218 | 42.134 | 41.371 | — |
| Full-system dev | 80.844 | 81.005 | 80.973 | 81.126 | 81.216 |

Individual X−I was +0.916 pp (paired 80% interval −0.012 to +1.845). X−D was +0.763 (−0.444 to +1.970). X beat each matched control on three of the four people, with different dissenting people. Both intervals include zero. The broader X−P comparison was +2.222 (+0.483 to +3.962); it includes contributions from both imaginary coherency and slope.

Full-system X−I was −0.033 pp (−0.111 to +0.049). X−D was −0.153 (−0.263 to −0.050). X−stock was −0.243 (−0.618 to +0.077). The full system favored the phase-destroyed control over genuine slope for this estimator and insertion. This rejects the positive full-system X−D prediction for the four-person cohort. The independent-model slope comparisons and full-system stock contrast remain undecided. No submission prepared. [Complete results, including person and interface contrasts](phase402_result.json).
