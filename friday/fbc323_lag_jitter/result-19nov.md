# F23 · Joint 50 ms lag pooling and two-sample source jitter

19 November 2026, 03:27:01 CET. The joint model reached 59.536% and 59.891% bounded whole-decoder balanced accuracy. Its margins over exact jitter and lag and the factorial interaction remain **undecided** across the two seeds. On seed 1 the interaction was negative with its paired interval below zero. This is whole-model DEV on ten fit-excluded PROTEUS participants and eight delivery orders. [Registered test](prereg-19nov.md)

| Inference | Seed | Exact hard CE | Exact jitter | Exact lag | Joint |
|---|---:|---:|---:|---:|---:|
| Bounded | 0 | 59.1613% | 59.4005% | 59.1599% | 59.5357% |
| Bounded | 1 | 58.8654% | 59.9529% | 59.4260% | 59.8909% |
| Ordinary | 0 | 58.5458% | 58.7817% | 58.6641% | 59.0250% |
| Ordinary | 1 | 58.3882% | 59.2789% | 58.9947% | 59.4326% |

| Bounded paired difference, pp [80%] | Seed 0 | Seed 1 |
|---|---:|---:|
| Joint vs. jitter | +0.135 [−0.210, +0.523] | −0.062 [−0.487, +0.464] |
| Joint vs. lag | +0.376 [−0.062, +0.886] | +0.465 [−0.126, +1.018] |
| Joint vs. hard CE | +0.374 [−0.177, +0.849] | +1.026 [+0.507, +1.561] |
| Factorial interaction | +0.137 [−0.458, +0.880] | −0.623 [−1.245, −0.019] |

The bounded joint-minus-jitter change for P04/P10 was −0.652/+0.200 pp on seed 0 and −0.965/+0.023 pp on seed 1. The P04 interaction was −1.773/−1.453 pp. P10's was +0.216/+0.087 pp. Both people were fit-excluded.

All 60 parent checkpoints matched their pinned bytes and every parent held decision matched the original bank. Both joint fits' seed groups updated all 288 lag coefficients on the exact jitter row tape. The collector checked 896 first-window identities, 1,792 interleaved resets and maximum prefix drift 1.95e−6. Ferranti `511387/511388` and Galvani `3133377–3133379` succeeded.

| Artifact in `ni26_friday_s1c_lag_jitter_20261119` | SHA-256 |
|---|---|
| Source actor | `0ae228bed1dc7dd9d653dbe66ac51ca5d5a486e0054cd95e98eb3ecc431d276d` |
| Held scorer | `6e8d6a1b69ec22dcbb3db51c1a1206424060b05d7b7883dd9b57e0e8bfb68910` |
| Blind decision manifest | `8f8e9bdefab28ffefe6b3c93c17b7427afadfae31a2b13896fb05e46d23823fc` |
| Results | `363de60158d284da9979286fba1b08f04d8a247d4a9f18c1b19f3163e2ffff49` |
