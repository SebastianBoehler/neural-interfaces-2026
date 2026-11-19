# F27 · Nonlinear electrode differences

19 November 2026, 05:48:55 CET. Physical-neighbor mixing improved over exact hard CE on seed 1 and lost to scrambled neighbors on seed 0. Both preregistered bounded effects remain **undecided** across seeds. This is standalone whole-decoder DEV on ten fit-excluded PROTEUS people and eight delivery orders. [Protocol](prereg-19nov.md)

| Inference | Seed | Exact hard CE | Physical | Scrambled | Physical − hard, pp [80%] | Physical − scrambled, pp [80%] |
|---|---:|---:|---:|---:|---|---|
| Bounded | 0 | 59.1613% | 58.9095% | 59.3747% | −0.252 [−0.767, +0.151] | −0.465 [−0.907, −0.145] |
| Bounded | 1 | 58.8654% | 59.3181% | 59.1193% | +0.453 [+0.100, +0.812] | +0.199 [−0.129, +0.517] |
| Ordinary | 0 | 58.5458% | 58.3405% | 58.9394% | −0.205 [−0.739, +0.201] | −0.599 [−1.149, −0.194] |
| Ordinary | 1 | 58.3882% | 58.8721% | 58.5566% | +0.484 [+0.088, +0.865] | +0.316 [−0.028, +0.641] |

Bounded physical minus hard changed +0.159/+0.288 pp on P04/P10 for seed 0 and +0.534/+0.438 for seed 1. Against scrambled it changed +0.098/−0.117 and +0.199/+0.377 pp. These are held-new roles.

The scramble preserved every sensor's directed degree and replaced 90.24% of edges. Median neighbor distances were 37.12 and 112.74 mm. All 1,476 extra coefficients were active in every learned fit. Mean maximum source movement under a fixed graph swap was 0.0129/0.0111 at the physical arm, and 0.0118/0.0133 in scrambled. Original raw hard checkpoint bytes and held decisions matched F16 exactly. Held verified 672 first-window identities, 1,344 interleaved resets and worst prefix drift 1.632e−6.

Ferranti `511476/511477` and Galvani `3134042–3134044` passed. Artifact hashes at `ni26_friday_s1c_electrodes_20261119`:

| Artifact | SHA-256 |
|---|---|
| Source actor | `4629f3dfa3b0d4d443de6599b7b2812e6e76cb0a479a60a0a15220c2cfae1332` |
| Held scorer | `dc0645c6d9c8f51245899943bef712c72f616fbcb2110b224100751a0aef9f10` |
| Blind decision manifest | `f58c95199d048734b9256f6a39ce47ff6ec3461adf1f807bb6ddd858f3c0297f` |
| Results | `5eb8c7e938a1b14a90562253b5aaa1b867c0e79ddcbb5f61439b49d210698e54` |
