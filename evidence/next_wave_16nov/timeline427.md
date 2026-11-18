# §427 · Where cap250 corrected and introduced errors

On ten held public people, the historical cap250 emulation gained **1.556 percentage points** over the v5.1b emulation. Windows after position 90 contributed 1.116 points, or 71.7% of that gain. Those windows held 52.5% of official-metric weight. The resulting excess late gain was +0.299 pp [80% +0.162, +0.438]. The prediction was +0.4. The excess varied with delivery scheme, reaching +0.493 within interface, +0.025 under full shuffle, and +0.379 within run.

The exact archives scored 48.677% and 48.717% on nine different Scherer people. Late deliveries carried 23.8% of metric weight and contributed −0.060 pp. Late excess was −0.069 [−0.394, +0.222]. The registered test leaves transport undecided. Positions are sampled at different frequencies on the shorter Scherer sessions. §411b previously found +2.110 pp on windows that crossed early and late between two orders. That estimate conditions on matched windows; this analysis accounts for every window and all three position bins.

| Cohort | Motor recall Δ | Subtraction recall Δ | Word recall Δ | Corrections | Regressions | Late excess |
|---|---:|---:|---:|---:|---:|---:|
| Public new, official cells | +1.999 | +1.669 | +0.999 | 4.17 | 2.62 | +0.299 |
| Returning P04 | +1.853 | +1.831 | +1.741 | 3.90 | 2.09 | +0.013 |
| Returning P10 | +0.672 | +0.799 | +0.574 | 1.48 | 0.80 | −0.114 |
| Returning, 20/80 | +0.908 | +1.006 | +0.808 | 1.96 | 1.06 | −0.091 |
| 56.6% public new + 43.4% returning 20/80 | +1.526 | +1.381 | +0.916 | 3.21 | 1.94 | +0.129 |
| External exact archives, full shuffle | +0.694 | −0.417 | −0.159 | 3.83 | 3.79 | −0.069 |

Word recall improved less than the mean of motor and subtraction on both datasets. The public new-person contrast was −0.835 pp [−1.702,+0.141], versus +1.0 predicted. The external contrast was −0.298 [−1.696,+1.023]. Both intervals include zero. The data establish neither a word advantage nor a specific source of physiologic or algorithmic change.

## Accounting and validation

A correction changes a wrong decision to correct. A regression changes a correct decision to wrong. Wrong-to-wrong changes contribute zero. For each official cell and true class, divide each of the three bin counts by the full cell/class count, then equally weight classes and cells. Subtracting summed regressions from summed corrections reproduces both official endpoint differences. Recall changes use percentage points within class. Late excess subtracts the net gain expected at the observed late exposure if gains were uniform by position. Bins are positions 1–50, 51–90 and 91 onward within each person-session. Resample people and shared orders within delivery scheme, seed 427, 6,000 draws. Returning uncertainty describes just two people. The 56.6% mixture is descriptive.

The [scorer](../../studies/18nov/timeline427/score.py) repeats the §419 source, route, permutation, endpoint and zero-arm validation. It checks both archives, both endpoints by public person and the order/person decomposition identities. The external scorer verifies the 18 original §411b JSON digests, source data/order implementation, expected rows, activity, both absolute scores and separate class recalls, and all order/person paired differences. Replay finished OK and a real-data smoke checked both arms on the first external order. No package, data split or model parameters changed. The [numeric result](timeline427_result.json) contains all class/position corrections and regressions by person and scheme; matrix rows are motor/subtraction/word and columns follow the three position bins.

The [§419 audit](lineage419.md) documents the historical emulations' auxiliary seed, gate and order limitations. The external archives used a different task protocol, montage and full-session permutations. This secondary analysis reuses frozen development outputs previously examined in aggregate. It measures where gain occurred and cannot attribute changes to a particular adaptive stage.

## Reproduction

At repository root, copy the §411b producer files `{old,cap}_{0..8}.json` and `result.json` from Galvani `/mnt/lustre/work/gehler/gfh527/ni26_cross411b_20261118/` into `runs/timeline427/bank/`. From that remote folder run `sha256sum -c sources.sha256 --status && sha256sum -c members.sha256 --status && sha256sum cap_[0-8].json old_[0-8].json`; save the checksum list as local `runs/timeline427/bank/output.sha256`. Stage the original §419 dependencies, run `.venv/bin/python replay/check.py > runs/timeline427/replay.txt`, then execute `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python studies/18nov/timeline427/score.py`. The `--smoke` flag checks one external order.
