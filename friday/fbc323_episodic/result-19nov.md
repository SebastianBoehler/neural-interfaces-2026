# F30 · Train with temporary source-person support

19 November 2026, 06:52:39 CET. Both preregistered bounded standalone contrasts remain **undecided**. Cross-person support gained on seed 1 versus exact hard CE. It did not improve consistently over the same-exposure within-person control. Fixed source cross-query loss was 0.9% and 1.9% higher than the within-person training arm. This is DEV on ten fit-excluded people and eight PROTEUS deliveries. [Protocol](prereg-19nov.md)

| Inference | Seed | Exact hard CE | Within-person support | Cross-person support | Cross − hard, pp [80%] | Cross − within, pp [80%] |
|---|---:|---:|---:|---:|---|---|
| Bounded | 0 | 59.1613% | 58.8659% | 58.9597% | −0.202 [−0.783, +0.377] | +0.094 [−0.273, +0.493] |
| Bounded | 1 | 58.8654% | 59.5005% | 59.4207% | +0.555 [+0.072, +1.008] | −0.080 [−0.469, +0.314] |
| Ordinary | 0 | 58.5458% | 58.2222% | 58.4895% | −0.056 [−0.568, +0.446] | +0.267 [−0.045, +0.589] |
| Ordinary | 1 | 58.3882% | 59.0168% | 58.9079% | +0.520 [+0.036, +0.997] | −0.109 [−0.494, +0.292] |

Bounded cross-minus-hard for P04/P10 was +0.636/−0.902 pp and +0.648/+0.903 pp at seeds 0/1. Cross-minus-within was +0.219/−0.477 and −0.274/+0.004 pp. Both people were excluded from their fits.

Both intervention arms shared 156,372/156,060 actual source query exposures, paired with the same number of support exposures. Native hard weights and decisions matched F16 exactly. Source-only residual steps agreed with independent autograd, and the fitted inference checkpoints use the original architecture. There were 672 exact first windows, 1,344 exact resets, and at most 1.482e−6 absolute held prefix drift.

Ferranti `511601/511602` and Galvani `3134377–3134379` passed. Artifact hashes at `ni26_friday_s1c_episodic_20261119`:

| Artifact | SHA-256 |
|---|---|
| Source actor | `d32d01f71b9eef27c383973238f02810486fd5b2988155e9d16039d3dfeb043a` |
| Held scorer | `bfb0f91a49bb951acd432b5936c367564d27efd7d8ff22bc44f5bf29ed8e4d7f` |
| Blind manifest | `50bca24bd78849b3b4c0aebde1b40dbc648bd4b4098b71b1db59a11342d15b3f` |
| Results | `892022a7c6cd5dae4e8888ea596dbbce4dbf6dbf636c7cea5629fcaa93fe847f` |
