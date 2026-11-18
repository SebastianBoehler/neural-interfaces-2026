# Can the same task contrast transfer between instruments?

A direction estimated exclusively from two independent Shin studies ranked WORD above SUB on Scherer. Its direction on PROTEUS remains uncertain. The registered dual-instrument condition failed, so these four-channel features do not qualify a package change. Both target instruments had been examined previously in this project, while their EEG and labels remained outside the source fits in this study.

| Within-run ranking AUC | Scherer | PROTEUS |
|---|---:|---:|
| Semantic direction W−A, equal people | 59.16% [56.35, 62.00] | 50.78% [46.38, 55.08] |
| Generic active direction W+A | 47.48% | 47.64% |
| Semantic minus generic | +11.67 pp [8.13, 15.23] | +3.14 pp [−2.15, 8.38] |
| W alone / −A alone | 52.60% / 56.26% | 48.70% / 51.05% |
| 90th percentile, 128 source-pair sign permutations | 54.11% | 54.25% |
| People above 50% | 8/9 | 6/10 |

AUC is the probability that a WORD trial outranks a SUB trial from the same run, with ties split evenly. Scherer provided 1,420 rows in 142 runs. PROTEUS provided 1,770 rows in 84 runs. All runs had both conditions. Intervals are 80% resamples of target people; the direction is fixed from the two source cohorts. Source-parameter uncertainty is outside these intervals. Target task labels were opened only after recording all 132 prediction columns, source/asset hashes and their row counts. No target threshold or target feature standardization enters predictions. [All results](semantics393_result.json)

Shin2017B supplies 29 people performing serial subtraction and a randomized fixation baseline. Shin2018 supplies 26 people performing word generation and a randomized baseline. Their respective within-source person-held AUCs were 59.10% [55.74,62.51] and 64.50% [62.10,66.79]. Only Cz/P3/P4/Pz are native across all four instruments. Six bipolar differences on those electrodes remove the common reference, and band fractions remove their overall scale. Each source direction compares the first four-second crop of matched within-run active and baseline trials, giving equal weight to every source person. Both directions use exactly the same spectral features. The previous two-source cog2 proposal lost its Shin2018 input before fitting (§148). §§317/352/356 used different source/target arrangements. [Arithmetic source protocol](https://moabb.neurotechx.com/docs/generated/moabb.datasets.Shin2017B.html) · [Word generation source paper](https://www.nature.com/articles/sdata20183)

The Scherer clinical cohort, language, cue timing and arithmetic ranges differ from both target competition tasks and source studies. A stable spectral direction across Scherer participants identifies cross-corpus discrimination on that instrument. Its interpretation as a general neural signature or an official three-class score would require additional evidence. [Scherer acquisition and provenance](external_semantics351.md)

Galvani 3123800/3123801/3123802 completed. Source manifest SHA-256 bd94d6e45a0b294d22fd1607c1bad9b55a2b8d9ce3d0b0a1e6aa31376d4c8d19; before-label manifest 9116e73d0e203978c90ab8f65b5f7761ffd0973d4cc12fbfc4fc841b308e3da0. [Source](../../studies/18nov/semantics393/produce.py)
