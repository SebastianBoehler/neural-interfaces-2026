# §384 · Using J/K waveform predictions to select personalization examples

The learned selector changed 42 and 40 emitted labels across the eight delivery orders. Its B2_pool accuracy change was **+0.097 pp** on seed 382 and **−0.024 pp** on seed 1382. Both 80% paired people/order intervals span zero. This four-person comparison supplies no replicated improvement over the original complete cap250 system. [All arms and people](selection384_result.json)

We changed the training set used by the deployed causal spatial-pattern and linear discriminant models. Within each incumbent pseudo-class, the original stage keeps the most confident previous windows. The alternative ranked each past window by 75% incumbent confidence and 25% J/K probability assigned to that incumbent class. Every refit retained its original class quotas. At each window the downstream classifier used its newly fitted state and the unchanged count-filter prior. The test re-ran every trajectory from session reset. [Actual selector and replay](../../studies/18nov/selection384/pilot.py)

J and K were trained on six people. For each of the four others, P01, P02, P03 and P09, we trained a calibration model on the other three. It saw only their labeled waveform probabilities, with equal cell/class and shuffle-scheme weights. It saw no incumbent predictions. For each held session, fixed upstream inputs and the native and frozen zero controls matched the original outputs. The stock endpoint was 81.2156%. P04 and P10 belonged to waveform training; this comparison yields no held scores for them. Both seeds used the same four historically exposed people. [Calibration model](../../studies/18nov/selection384/calibrate.py) · [Frozen score inputs](selection384_before_labels.json)

| Full-system B2_pool difference, pp [80%] | Galvani 382 | Ferranti 1382 |
|---|---:|---:|
| Learned S − stock | +0.097 [−0.098,+0.282] | −0.024 [−0.198,+0.149] |
| Learned − shuffled waveform evidence | −0.134 [−0.470,+0.172] | −0.233 [−0.582,+0.127] |
| Learned − scalar temperature | +0.068 [−0.054,+0.191] | −0.054 [−0.195,+0.087] |
| Learned − same-capacity K-only head | +0.022 [0.000,+0.056] | −0.010 [−0.064,+0.045] |
| Shuffled evidence − stock | +0.231 [+0.014,+0.440] | +0.209 [−0.031,+0.440] |

Across 288 refit events per seed, S replaced 1,370/1,355 selected row memberships. The shuffled control replaced 1,944/1,932. It used identical class quotas and waveform evidence multisets, but seeded its within-class shuffle again at each refit. The candidates therefore differ in selection turnover. The positive shuffled-control estimates warrant separating that effect from waveform semantics before attributing a gain.

## Did the training examples become more accurate?

Stock's selected pseudo-labels were already **91.708% correct**. At the refits that contributed to downstream emissions, the learned selector reduced that fraction by 0.434 pp [−0.797,−0.075] and 0.485 pp [−0.868,−0.105]. Against the shuffled comparator, the learned selector improved purity by 0.625 pp [+0.227,+1.032] and 0.469 pp [−0.039,+0.993]. This diagnostic averages pseudo-class precision within each active refit and is separate from the official system metric.

A label-assisted reference kept correct past pseudo-labels first, with stock confidence breaking ties. It achieved 100% purity at the same quotas. Its full-system gain was only **+0.163 pp [−0.007,+0.350]**. This diagnoses the effect of maximizing selected-label purity. Other sample choices, including those optimized for covariance estimation, can produce different system scores. The label-assisted rule requires withheld labels and has no deployment role. [Diagnostic results](selection384b_result.json)

On Galvani the primary snapshot was `ni26_selection384_v1_20261118`, source manifest `d58a4d9b75aa6cca648f59126eb593e51c42f4cc242346a65e0d0d55fe840500`. Replay/smoke 3123397, all eight workers 3123398 and scorer 3123399 passed. The matched-quota diagnostic ran in `ni26_purity384_v1_20261118`, manifest `bf55f5d27f06e35d25bdd891c536fc8b39d6d865b2b2e0ac7c3e2ce540e21140`; jobs 3123409, 3123410 and 3123411 passed. All work used CPU. No new packaging or sealed score. [Preregistration](../../docs/LEDGER.md)
