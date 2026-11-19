# F-C29 · Source-class covariance in the native session CSP fit

F-C29 fitted source-class covariances in the original eight native bands, averaged source participants equally, and whitened each prior using its source pooled covariance. Each genuine session CSP refit recolored the class priors from its strictly past unlabeled covariance. It used 12 prior rows against the actual selected class or rest count and preserved the native covariance traces. The original confidence-ranked rows, eigensolver, readout and fusion remained in the path. [Registered design](../../docs/LEDGER.md#f-c29--transport-source-class-covariance-into-the-real-session-csp-fit)

The reflection control preserved the transported priors' eigenvalues, rank and traces. It altered only their orientation in the same original EEG coordinates. The pooled control used the same shrinkage strength without class labels in the prior. Native identity used the same hook and returned the original covariance matrices unchanged.

## Corrected held-new PROTEUS

**FULL-SYSTEM DEV** at 09:43:41 CET on 19 November. Ten reused held-new people and 28 official balanced cells in each corrected delivery. On full0, the transported prior beat exact stock, reflection and pooled controls with positive paired lower bounds. Within0 reversed direction. Adding the prior to the exact matched hard parent also reversed by order. The fixed intervention missed its cross-order advancement rule.

| Complete system | Within0 | Full0 |
|---|---:|---:|
| Exact stock and native identity | 78.32415% | 73.73898% |
| Transported class prior | 78.13723% | 74.06929% |
| Reflected class prior | 78.39666% | 73.72553% |
| Pooled prior | 78.20795% | 73.87570% |
| Exact hard parent | 78.62892% | 74.07033% |
| Exact hard path with transported prior | 78.44071% | 74.32978% |

| Contrast, pp [paired 80%] | Within0 | Full0 |
|---|---:|---:|
| Transported minus stock | −0.187 [−0.526, +0.103] | +0.330 [+0.053, +0.562] |
| Transported minus reflection | −0.259 [−0.531, +0.002] | +0.344 [+0.094, +0.580] |
| Transported minus pooled | −0.071 [−0.332, +0.179] | +0.194 [+0.007, +0.347] |
| Transported minus hard | −0.492 [−0.832, −0.244] | −0.001 [−0.274, +0.214] |
| Hard with transported prior minus exact hard | −0.188 [−0.496, +0.075] | +0.259 [−0.099, +0.504] |
| Label-aware stock/transported ceiling | +0.378 [+0.149, +0.571] | +1.010 [+0.707, +1.239] |

Transport on stock rescued 10/27 decisions with 15/18 regressions within0/full0. Its point effects against stock on held-new P04 were +0.712/+0.532 pp, and on P10 were −0.357/+0.496 pp. The same transport on hard rescued 8/21 and lost 13/14 decisions versus the exact hard parent. The label-aware ceiling describes the rescues available between these outputs and stays outside candidate selection.

The source-only bank materialized ten leave-person-out artifacts and one public-to-external artifact. Each public fold excluded the held person from its class and pooled priors. The exact corrected stock, native identity and separate F-C20 hard path reproduced all signed arrays. Seven private full-system paths each ran on 2,655 rows, with 126 real selected-row CSP fits and 1,395 fusions per arm. Truncated 31- and 105-window prefixes were exact, including an interleaved independent session. Jobs took 7m10 and 7m13.

| Public receipt | Job | SHA-256 |
|---|---:|---|
| Within0 bank | 3135202 | `94e29d2666dd8cb89d5dd665fedb9352dc87301555e41aa03415d0a75da46240` |
| Full0 bank | 3135203 | `0b5e0da0845e75f03de52473f32aa2b0258133f442b8126c026df50bd2d88e45` |
| Seal before scores | 3135204 | `b1bb4de16e3209bd350046bb2dbe0f6a9462432af12bf1b1b1d2095766543956` |
| Score | 3135204 | `65202e4587481330c9cb5265d4ce9e0bdfd440f371f4b02b4f985eb1b63d9a9a` |

## Native Scherer replay

**FULL-SYSTEM EXTERNAL DEV** at 09:49:43 CET. The pinned RTX Solver supplied the same complete live stock upstream to all five personalizer branches. The prior was trained on public PROTEUS and saw no Scherer data. Both fresh deliveries reuse the same nine Scherer participants. All primary external intervals cross zero, and both stock comparison point estimates were negative.

| Complete system or contrast | Pooled | Paired 80% | Orders 24411 / 25411, pp |
|---|---:|---:|---:|
| Exact stock and native identity | 48.76653% | — | — |
| Transported prior | 48.67394% | — | — |
| Reflection | 48.76653% | — | — |
| Pooled prior | 48.72024% | — | — |
| Transported minus stock | −0.09259 pp | [−0.20833, +0.02315] | −0.09259 / −0.09259 |
| Transported minus reflection | −0.09259 pp | [−0.20833, +0.02315] | −0.09259 / −0.09259 |
| Transported minus pooled | −0.04630 pp | [−0.13889, +0.04630] | −0.13889 / +0.04630 |
| Label-aware stock/transported ceiling | +0.04630 pp | [0.00000, +0.11574] | +0.04630 / +0.04630 |

The transported prior changed 15 decisions, with two rescues and six regressions. Motor imagery was flat, subtraction changed −0.20833 pp and word generation −0.06944 pp. Effects for A/C/D/E/F/G/H/J/L were 0/−0.208/−0.208/0/+0.208/0/0/−0.208/−0.417 pp. The exact two-decision ceiling fell below the forecast of +0.65 pp.

The real RTX G smoke preserved the five CSP fits, 30 fusions and 96-column LDA. It recorded a 0.20929 relative change in an actual class or rest covariance passed to the native eigensolver. The prior differed from its reflection by 0.07446 relative while its spectrum and trace matched. The transported terminal score moved by up to 0.01714. All 120 stock and identity scores were bitwise. Native P10 stopped CSP at recognition age 10 and emitted exact stock across all 31 frames in every arm. The 31- and 105-window interleaved prefixes were exact.

All nine prediction jobs sealed 4,260 decisions per arm. They verified the same actual selected 5,549 prior-row exposures, 180 CSP fits and 1,020 active fusions. The real native solver used 4,320 transported class/rest covariance pairs. The initial scorer `3135205` stopped during shell authentication because it looked for the new source-fit manifest inside the older C20e directory. The retry corrected only that manifest location in a separate release and reused all nine closed decision files. The initial scorer opened no gold.

| Artifact | Galvani job | SHA-256 |
|---|---:|---|
| Native source feature smoke | 3135140 | `59883b6d0e17f45438ac23932535f2fb6eed67015ff3d5ce4b188e2806f8a889` |
| Source-only fit bank | 3135141 | `68ab2b53087f39aa8d89e482bb90d54960d76660c499b3ccb33787ea9eef9263` |
| Native G/P10 smoke | 3135168 | `bae5fdf19856464dfb15be6d88f02c30433f4fb938fcfbb4eae3b35a0bd18e4d` |
| Native upstream/CP predictions | 3135169 | Source `9871b7568696f7b6dbbae8284522738cef8aed3b94e845a0d32117be7952ad43` |
| External seal before gold | 3135467 | `34394f27efd4f63bb7eb8c5d7b19415cbfd835308d8757b5ec2db19e8a501e8f` |
| External score | 3135467 | `23a38478cdd28443ac1135dbc5089a19dc7c22af28fe0241d1ecf36067d665b7` |

Prediction and public score release `ni26_friday_c29_spatial_prior_v1_20261119`. External score release `ni26_friday_c29_spatial_score_v2_20261119`, scoring manifest `c1f1dbf9095c8862869b39f53fc9216bc071154e12afbccc0f89884ff4fce3fe`. Source code, fixed orders and worker decisions stayed identical for that scorer retry.
