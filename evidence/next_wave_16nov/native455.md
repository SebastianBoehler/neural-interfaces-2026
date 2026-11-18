# §455 Cap250 personalization on held new people

The frozen historical B2_pool cap250 scored 76.2793% FULL-SYSTEM DEV on ten held new people. Correcting the implicit class prior in its session-trained classifier changed ten of 21,240 delivered decisions and gained 0.009 pp. The two larger changes that used agreement between the classifier and the chain both lost accuracy on average. Every primary interval spans zero.

| Native full-system contrast (percentage points) | Change | Paired 80% interval | Changed decisions | Held-new P04 / P10 |
|---|---:|---:|---:|---:|
| Half prior correction − stock | +0.009 | −0.014 to +0.035 | 10 | +0.042 / −0.029 |
| Past forecast reliability − stock | −0.042 | −0.099 to +0.013 | 62 | −0.074 / −0.002 |
| Past forecast reliability − cyclic control | −0.076 | −0.151 to +0.001 | 102 | −0.192 / −0.017 |
| Reliability plus prior correction − prior correction | −0.042 | −0.114 to +0.022 | 62 | −0.117 / +0.085 |

At each 20-window refit, the prior arm estimated the empirical proportions of the original LDA pseudo-labels from class-stratified selection counts. It removed half their smoothed log prior from its late log evidence. The reliability arm measured the earlier, unfiltered chain probability assigned to the personal classifier's predicted class. It multiplied the current personal contribution by a bounded function of this previous agreement, beginning with 20 neutral observations. All arms used the original count filter, fusion ramp, cross-session resets and terminal cap250 scores. The scorer used 28 equally weighted official evaluation cells and gave each of three delivery schemes equal weight. Paired intervals resampled people and orders within scheme.

The readout becomes active at zero-based age 90. Every active recovered personal probability vector met the package's clipped-probability sum bound, and every early count probability reproduced the full stock terminal vector. The real within0 smoke confirmed session-prefix invariance and the source selector. Replay printed OK. Galvani CPU smoke 3126852, eight-order array 3126853 and both collectors 3126854/3126855 completed. [Prior results by order and person](native455_prior.json) · [Reliability results by order and person](native455_reliability.json)

## Separate two-seed learned-memory control

On the same historical stock, requiring the two §443 waveform seeds to agree on each correction component scored 76.2478%, −0.032 pp [−0.138,+0.069]. The two-seed average scored 76.2823%. Subtracting a causal session-average offset before computing agreement scored 76.3250%, +0.046 [−0.012,+0.100] against stock and +0.094 [+0.015,+0.185] against matched donor memory. The prior-only arm scored 76.3440%, leaving residual minus prior-only at −0.019 [−0.101,+0.067]. The donor contrast supports a contribution from the held session compared with substituted history. A positive gain over unchanged stock remains undecided. Both original single-seed controls, zero correction, order and stock decision checks passed. CPU smoke and both waves completed as 3126809/3126839/3126840. [Agreement results](consensus455_agreement.json) · [Residual results](consensus455_residual.json)

The historical comparison uses the original §443 auxiliary seed counts. The §456 study separately qualifies these edits against the count-matched three-eye/three-rest historical baseline. These experiments reuse public development people and produce no sealed score.
