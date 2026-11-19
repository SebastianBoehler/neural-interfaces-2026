# F24 · Source calibration-context robustness

19 November 2026, 04:36:40 CET. The cross-person auxiliary reduced clean source cross-context disagreement by 91.4% and 84.7% relative to exact jitter. Held accuracy fell at both seeds. Both registered paired contrasts remain **undecided** across seeds because the seed 1 intervals include zero. The observed results provide no qualification of a full-system gain. [Protocol](prereg-19nov.md)

| Held-new whole balanced accuracy | Hard CE | Exact jitter | Within-session moments | Cross-person moments |
|---|---:|---:|---:|---:|
| Bounded seed 0 | 59.1613% | 59.4005% | 59.0398% | 58.2650% |
| Bounded seed 1 | 58.8654% | 59.9529% | 59.6043% | 59.0880% |
| Ordinary seed 0 | 58.5458% | 58.7817% | 58.7727% | 57.8198% |
| Ordinary seed 1 | 58.3882% | 59.2789% | 58.9502% | 58.6257% |

| Paired contrast, pp [80% interval] | Seed 0 | Seed 1 |
|---|---:|---:|
| Bounded cross vs. jitter | −1.136 [−1.865, −0.594] | −0.865 [−2.070, +0.214] |
| Bounded cross vs. within | −0.775 [−1.852, −0.030] | −0.516 [−1.580, +0.426] |
| Bounded within vs. jitter | −0.361 [−0.809, +0.245] | −0.349 [−0.779, +0.114] |
| Ordinary cross vs. jitter | −0.962 [−1.714, −0.366] | −0.653 [−1.840, +0.406] |
| Ordinary cross vs. within | −0.953 [−1.948, −0.186] | −0.325 [−1.247, +0.508] |

Bounded cross-minus-jitter effects for P04/P10 were −0.409/+0.322 pp at seed 0 and −2.032/+1.823 pp at seed 1. Cross-minus-within was +0.786/+1.220 and −1.327/+1.898 pp. These are held-new roles, with all ten public people fit-excluded in turn and eight order deliveries.

Both producer arms received matched disjoint query/donor exposures. They had zero donor gradients, active native body and affine gradients, and preserved native running-state writes. Both hard and jitter anchors matched every parent checkpoint and held decision. The held audit checked 896 exact first windows, 1,792 resets and prefix drift at most 3.054e−6.

Ferranti `511412` and held smoke `3133805` passed. Nineteen elements of `511413` completed 0:0. Seed 1/P07 element `511413_16` printed PASS after closing and hashing all four complete 40-epoch models, then Slurm recorded TIMEOUT. All eight transferred files matched their source digests. Independent held task `3133806_16` authenticated the payload and completed 0:0. The sealed collector `3133807` likewise completed 0:0.

| Artifact in `ni26_friday_s1c_calibration_v2_20261119` | SHA-256 |
|---|---|
| Source actor | `107d380fe6cb5e55532c3194986d7f0a5eb0705f48666fc8383fbedabfbaa3d2` |
| Held scorer | `0acf64a3303718a1dcfe874e962e0cb36cbe71b4db22794d0b4cbc1df2e935cf` |
| Blind decision manifest | `f05a315560c4fc4b0908105eafb66d4be2f482fcda0c1d922074f0b9ab0a16e1` |
| Results | `2f3748d9c72f94b72c95906eeca2fa582f12222c3e6767b8367607807acbbceb` |
