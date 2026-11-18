# §410 Does training-set size explain the burst-duration reversal?

The standalone effect of preserving EEG burst durations still reversed when both query cohorts used the exact same fitted models. On six source people, intact durations lost 2.014 points to durations computed after sample shuffling. On four development people, they gained 1.358. The difference was +3.372 pp [80% +1.695, +5.054]. This cohort difference persisted for both interfaces and when every person contributed only their first session.

| Intact duration minus sample shuffle, pp | Six source queries, five training people | Four development queries, same five | Four development queries, original six | Matched cohort gap [80%] |
|---|---:|---:|---:|---:|
| Whole classifier | −2.014 | +1.358 | +1.040 | +3.372 [+1.695,+5.054] |
| Whole, session 1 only | −1.880 | +1.358 | +1.040 | +3.238 [+1.535,+4.953] |
| Whole, Graz | −1.273 | +1.042 | +1.302 | +2.315 [+0.231,+4.427] |
| Whole, BrainHero | −2.756 | +1.674 | +0.778 | +4.430 [+2.414,+6.327] |
| Full frozen stack, motor-only insertion | −0.035 | +0.076 | +0.016 | +0.110 [−0.071,+0.282] |

The original whole-classifier gap of +3.054 points equals the matched cohort gap plus the effect of adding the sixth training person, −0.318 [−1.598,+0.994]. The latter point estimate was inside the predicted one-point range. Its interval extends past −1. Motor imagery and word generation had the largest matched recall gaps, +5.366 and +4.181 pp respectively. Subtraction was +0.570 [−4.615,+6.016]. P02 contributed the largest development gain (+4.784); P04/P10 changed −0.330/+1.052. The identities, repeated public exposure, and four-person development sample limit any inference about population differences or physiological causes.

**Full-system development.** Across all ten people and all 28 official participant/session/interface cells, frozen B2_pool scored 76.279%, amplitude 76.186%, intact duration 76.144%, sample shuffle 76.129%, and block shuffle 76.183%. Intact duration changed stock by −0.136 pp. The corresponding whole classifiers scored 35.789%, 36.075%, 36.447%, and 35.920%. Within a fold, each member was an actual model run through the complete system. For the four development people we averaged the six models' scores; there was no probability ensemble. These are new-person replays. The people-weighted gap in the table measures transport, whereas the aggregate above uses official cell weights.

**Design and checks.** Each of six training panels excluded one original source person and all four development people. The unchanged §407 predictors ran on all five excluded people. A separate array ran the original motor-only insertion, count filter and causal personalization on eight pinned orders; each order family had equal weight. Frozen zero and prefix checks, six real actor smokes, v3 replay OK and original per-person results all matched. The matched interval resamples source people with their training panels, development people shared across panels, and orders within family (5,000 draws, seed 410). The full matched gap remains undecided. No held returning-person or sealed estimate.

Galvani jobs 3124285/86/87/88, 3124302 and reporting supplement 3124388 completed. The supplement reproduced every primary contrast exactly and verified the all-cell stock anchor. Implementation began at c72ee93; the evidence retains the scorer and output manifest SHA-256. [All scores and per-person contrasts](transport410_result.json).
