# F-C21gcv · Share the current raw EEG covariance across native tails

19 November 2026, 07:09:56 CET. On the RTX 2080 Ti, the covariance-sharing selector completed Scherer G at **1.10425× stock** and **0.93819× the original dual-tail selector**. The worst matched ratio to stock was **1.10585×**. All three registered timing criteria passed on this fixture.

| Warm G visit 1, 120 windows | First run | Mirrored run |
|---|---:|---:|
| Exact stock | 26.432 s | 26.364 s |
| Full dual-tail selector | 31.106 s | 31.035 s |
| Covariance-sharing selector | 29.145 s | 29.154 s |

The six fresh sessions ran stock, full, shared, shared, full, stock after one stock warm-up. Covariance sharing avoided 120 alternate extractions per stream and preserved actual classes, stock work and five private CSP refits with 30 fusions. Each branch owns its copy of the matrices and its own pseudo-labels, filters, classifier and count state. The per-row source guard verifies the current raw waveform, expected stock age and actual new covariance row.

The RTX smoke independently re-evaluated and matched 120 G covariance matrices and ten recognized-P10 prefix matrices bit-for-bit. It used the actual live adapted FBLight path and independent stock and alternate tails. The historical same-input, append, prefix and reset checks passed. No task labels were opened. These tests measure one implementation and a single paired unrecognized timing fixture. The separately measured [recognized retirement](../cap250c21_gate/native/README.md) and covariance sharing require a combined live and package check.

| Artifact | Galvani job | SHA-256 |
|---|---:|---|
| Source manifest | — | `40ce7409bfc3f86ba53729b7e84c3b2ae9dd99f27392c51bf419abcfdd1c498d` |
| Smoke | `3134466` | `2416211f376140f5114816dc47c4700fdc248675167eb793fc1fe88589975f2d` |
| Paired timing | `3134467` | `72c2cc9d58dfef88cdb597dd97b23b1df6e7c6d05600846371cf827891e05874` |

Release `ni26_friday_c21gcv_rtx_v1_20261119`. Implementation `da7fe3fc`.
