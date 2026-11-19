# F34 · Analytic envelope morphology on held-new PROTEUS

19 November 2026, 09:44:53 CET. The analytic-envelope model lost 5.43 and 8.35 percentage points to exact hard CE. This refutes the registered whole-accuracy gain. Its difference from the same-capacity real-power branch remains **undecided** across seeds. The scores are standalone A100-trained DEV on ten whole-person exclusions across 28 official cells and eight deliveries. [Protocol](prereg-19nov.md)

| Inference | Seed | Exact hard CE | Analytic envelope | Real power | Analytic − hard, pp [80%] | Analytic − real power, pp [80%] |
|---|---:|---:|---:|---:|---|---|
| Bounded | 0 | 58.6790% | 53.2491% | 52.9392% | −5.430 [−7.361, −3.967] | +0.310 [−0.319, +1.178] |
| Bounded | 1 | 59.1089% | 50.7541% | 49.9909% | −8.355 [−10.505, −6.395] | +0.763 [+0.233, +1.496] |
| Ordinary | 0 | 58.1123% | 53.1400% | 53.1358% | −4.972 [−6.738, −3.617] | +0.004 [−0.620, +0.870] |
| Ordinary | 1 | 58.7062% | 50.7450% | 49.9255% | −7.961 [−9.999, −6.128] | +0.820 [+0.295, +1.536] |

Bounded cognitive mean recall analytic minus real power was +0.817 [+0.172, +1.717] and +1.007 [+0.509, +1.539] pp. Against hard it fell 6.618 and 8.375 pp. Bounded P04/P10 analytic minus hard was −3.388/−2.342 and −10.903/−3.145 pp. Average last source minibatch cross-entropy was 0.349/0.266 for hard, 0.011/0.009 for analytic, and 0.011/0.009 for real power. Those source minibatches measure optimizer behavior. The independent held results show a sharp transfer cost in both larger models.

Float32 phase smoke `3135174` failed before held inference. V2 used identical float64 spectral and power arithmetic for both arms before their common float32 convolutions and passed the unchanged limits. A100 `3135190` and CPU `3135208` smokes passed. All twenty `3135209/3135210` elements and independent scorer `3135211` passed. Every hard checkpoint and 84,960 hard decisions reproduced F33a. All 672 first windows and 1,344 interleaved resets passed with maximum prefix difference 2.311e−6. The complete prediction bank was sealed before labels. [Precision repair record](precision-19nov.md)

| Artifact in `ni26_friday_s1c_envelope_a100_v2_20261119` | SHA-256 |
|---|---|
| Source actor | `56ca60c2d190b3b51005f09cd53d66243af92ba2a319e58544594d0e9a94e60c` |
| Held scorer | `899e0abb8c25227f8d0dbbb2061b53e49b8d9c5880da423ef119efb760fb6db5` |
| Sealed decisions | `5b9a8e2fc8fb0de1019608334f642adbdce40bee28313ce84ef52f185437d15c` |
| Whole results | `6090173a9a8690e1467ee703c6d0752dc1e59e982bdef8ed24cae9792f834858` |
