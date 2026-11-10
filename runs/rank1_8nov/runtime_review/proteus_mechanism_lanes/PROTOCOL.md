# Source-only mechanism screens for PROTEUS

All lanes use the same seven source people and participant-disjoint
folds, 0/100/200 matched bootstrap draws, exact 10% channel-power
baseline parity, fixed 10% auxiliary fusion and official equal-cell
balanced accuracy. None reads P04/P09/P10 EEG or labels.

| Lane | Question | Matched check |
|---|---|---|
| Copula covariance | Does electrode dependence carry robust signal when sensor amplitudes are distorted? Replace each channel's within-window ranks with normal scores before band correlation and tangent projection. | Ordinary standardized correlation, same spectral bands and geometry. Soft compression stresses the auxiliary head only. Strictly monotone compression leaves the rank view unchanged. |
| Interface nuisance | Does suppressing directions that vary by BrainHero/Graz within person and command preserve class evidence across people? Remove the three strongest directions of the training-context mean differences. | Same rank removed using within-interface random split-half directions, plus the unprojected head. |
| Temporal order | Do broad trajectories within a command window add evidence beyond average power? Append the first two nonconstant cosine coefficients across seven one-second tiles. | Same features after independent label-free tile shuffling, plus a mean-only head. |

Each primary arm must gain +0.8 points overall and +0.5 on BrainHero,
preserve Graz, improve two seeds, and beat its matched control by +0.3.
These thresholds screen source evidence only. The seven people provide
only session 1. The task reports all results without opening the
reserved bank or submitting to Codabench.

[Gaussian rank normalization](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23471)
