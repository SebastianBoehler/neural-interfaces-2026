# F22 · Within-block lag in the raw S1C decoder

19 November 2026, 02:20:54 CET. Both preregistered bounded contrasts remain **undecided**. The physical lag improved seed 1 versus exact hard CE with a positive paired interval. Seed 0 was flat, and both matched scrambled-pair intervals cross zero. These are whole-decoder DEV results on ten fit-excluded PROTEUS people, eight deliveries and two seeds. [Registered protocol](prereg-19nov.md)

| Inference | Seed | Hard CE | 50 ms lag | Scrambled pairs | Lag − hard, pp [80%] | Lag − scrambled, pp [80%] |
|---|---:|---:|---:|---:|---|---|
| Bounded | 0 | 59.1613% | 59.1599% | 59.1083% | −0.001 [−0.553, +0.434] | +0.052 [−0.462, +0.546] |
| Bounded | 1 | 58.8654% | 59.4260% | 58.9636% | +0.561 [+0.230, +0.967] | +0.462 [−0.015, +1.012] |
| Ordinary | 0 | 58.5458% | 58.6641% | 58.8438% | +0.118 [−0.378, +0.510] | −0.180 [−0.659, +0.308] |
| Ordinary | 1 | 58.3882% | 58.9947% | 58.5358% | +0.606 [+0.260, +0.971] | +0.459 [−0.044, +1.030] |

Bounded lag-minus-hard for P04/P10 was +1.121/−0.017 pp on seed 0 and +0.487/−0.063 pp on seed 1. Against scrambled pairs it was +0.767/−0.473 pp and +0.963/−0.857 pp. Both people were held out of their fits.

All 40 modified models updated all 288 coefficients. The mean fold-level maximum pooled output change when swapping the pairing was 0.106/0.102 on the lag arms and 0.129/0.139 on scrambled. The real source analytic variance error was at most 2.61e−18. Warm 32-window H100 forward medians were 1.790 ms hard, 1.872 ms lag and 1.865 ms scrambled.

Ferranti 511347/511348 and Galvani 3133120/3133121/3133122 completed. All 20 hard checkpoints matched F16 byte-for-byte, and all hard decisions and aggregate scores matched. Held checks covered 672 exact first windows, 1,344 exact interleaved resets, 863,851 bounded clip entries, and maximum absolute prefix drift 1.950e−6.

- Actor SHA-256 `8a7d1784adbd88fd7381e9f3628a613fa1b7768c4732ca2551ae37cb9687f2c1`
- Held scorer SHA-256 `979f5a2152b6e521fbeaac6108978817be6ae8f0ddddc4af4629855291b949bd`
- Blind decision manifest SHA-256 `9c9dff817b519229033339f402e735191cd01b4a6e913c70df1db956093229e5`
- Results SHA-256 `ef8d307bb460f65886bcd03de70ffdf31ec8a6dffc525d0b5152c918c8596931`

Authenticated results and blind banks are on Galvani under `/mnt/lustre/work/gehler/gfh527/ni26_friday_s1c_temporal_v3_20261119`.
