# Lane TB-grow · Team B on the hard-only base

Leonardo run `cap250/runs/laneTB-grow-21nov`. [Preregistration](../../docs/LEDGER.md#lane-tb-grow--dose-insertion-point-and-independent-sources-registered-21-nov-2026-before-scoring) · [Slurm dispatch](DISPATCH_21NOV.md)

## Instrument

Hard-only fold fixtures use `production-20261120-32a94ab7`. The ten held-new folds contribute 28 official cells per order. P04 and P10 contribute 8 returning cells per order, from visits 2–3 with visit 1 enrolled. Every arm uses the live hard Solver, with one EEG window at a time and a fresh copy per session. Local v3 replay printed OK. The PROTEUS export SHA-256 is `2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327`.

Orders 71140–71147 select by J, the equal mean of the held-new and returning cell deltas. The collector freezes one runtime-eligible arm. The primary confirmation uses orders 71148–71155. The additional block uses 71172–71179. Both replay the chosen arm, hard alone, and hard nine25. The pooled 16-order estimate is secondary. Paired 80% intervals resample people with their official cells, then orders within each sampled person. The 57/43 column assumes 57% new and 43% returning, with returning P04/P10 at 20/80. The sealed proportions remain unverified.

Timing pairs use the full Solver on the same warm A100, with order reversed on the second P01 stream. The projection starts from the previous 2,943 s A100 estimate for hard nine25, then adds the measured difference over 20,083 windows and any positive load and fresh-session setup differences. Official runtime on RTX PRO 4500 is unmeasured. Team B cost can grow with session length.

## Selection on orders 71140–71147

The frozen choice is **`ba_terminal`**. It keeps Team B nine25 after the hard mixer, then assigns 10% to the independent Team A MANAS A1 B6 posterior after the count and CSP stages. Its selection J was +1.45 pp. Nine25's J was +0.98 pp.

| Arm vs hard | Held-new Δ pp [80%] | Returning Δ pp [80%] | P04 / P10 Δ pp | J pp | 57/43* pp | Warm A100 ms/window | Projected s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ba_terminal` | +0.81 [+0.34, +1.18] | +2.09 [+1.55, +2.68] | +2.58 / +1.60 | +1.45 | +1.23 | 274.5 | 3,285 |
| `terminal25` | +0.85 [+0.43, +1.18] | +1.35 [+1.07, +1.60] | +1.17 / +1.53 | +1.10 | +1.11 | 259.7 | 2,903 |
| `nine35` | +0.93 [+0.31, +1.43] | +1.25 [+0.97, +1.52] | +1.26 / +1.24 | +1.09 | +1.06 | 260.7 | 2,922 |
| `nine50` | +0.56 [−0.20, +1.23] | +1.50 [+1.23, +1.77] | +1.59 / +1.40 | +1.03 | +0.94 | 262.8 | 3,017 |
| `native25` | +0.71 [+0.23, +1.15] | +1.33 [+0.86, +1.82] | +1.76 / +0.90 | +1.02 | +0.86 | 260.9 | 2,944 |
| `nine25` | +0.71 [+0.20, +1.11] | +1.25 [+0.92, +1.61] | +1.44 / +1.06 | +0.98 | +0.89 | 260.2 | 2,943 |
| `twelve25` | +0.78 [+0.28, +1.18] | +1.07 [+0.80, +1.36] | +1.16 / +0.99 | +0.93 | +0.89 | 261.8 | 2,959 |
| `eighteen25` | +0.68 [+0.17, +1.08] | +1.09 [+0.68, +1.50] | +1.40 / +0.77 | +0.88 | +0.77 | 263.7 | 2,932 |
| `bc_a1` | +0.26 [−0.05, +0.60] | +0.63 [+0.35, +0.95] | +0.86 / +0.41 | +0.45 | +0.36 | 271.5 | 3,263 |
| `bc_light` | +0.23 [−0.05, +0.55] | +0.59 [+0.32, +0.88] | +0.80 / +0.37 | +0.41 | +0.33 | 281.5 | 3,311 |
| `bc_a2` | +0.18 [−0.10, +0.48] | +0.45 [+0.12, +0.80] | +0.70 / +0.21 | +0.32 | +0.23 | 314.8 | 4,038 |

Hard alone scored 73.54% held-new and 79.12% returning on these orders. Full four-step Team C (`bc_a2`) exceeded the 3,600 s projection limit. The projection for each other arm was below the limit.

Freeze job 65003091 validated 576 prediction files and wrote `select.json` (SHA-256 `ecb5c265f3fb094c59af7a4b130fa098ebe0a83753480a77e592a9d970818157`). The hard and nine25 P01 smoke matched 380 pinned labels. Each candidate passed the prefix and fresh-state checks. Both returning Team C fits used the registered six-member roster and passed separate live smokes after the initial three-member default was rejected. Those initial tasks stopped before running a Team C arm. The retry completed every missing prediction.

## Primary confirmation on orders 71148–71155

**`ba_terminal` improved both cohorts versus hard.** Its incremental gain over nine25 was supported on the two returning people. The held-new increment remained undecided.

| Contrast | Held-new Δ pp [80%] | Returning Δ pp [80%] | P04 / P10 Δ pp | J pp | 57/43* pp |
|---|---:|---:|---:|---:|---:|
| nine25 − hard | +0.64 [+0.22, +1.06] | +1.20 [+1.02, +1.37] | +1.16 / +1.24 | +0.92 | +0.89 |
| ba_terminal − hard | +0.84 [+0.41, +1.26] | +1.62 [+1.25, +2.01] | +1.81 / +1.43 | +1.23 | +1.13 |
| ba_terminal − nine25 | +0.20 [−0.08, +0.48] | +0.43 [+0.06, +0.80] | +0.65 / +0.20 | +0.31 | +0.24 |

Absolute official cell BA for hard was 73.17% held-new and 79.36% returning. Ba_terminal reached 74.01% and 80.98%. The returning estimates cover P04 and P10 at visits 2–3, with visit 1 enrolled.

Runtime projects to **3,285 s** on the 2,943 s nine25 anchor. On A100-SXM-64GB the two warm P01 pairs added 15.91 and 13.31 ms/window. The projection includes load and first-window overhead. It leaves about 315 s against the 3,600 s limit. Runtime on the official RTX PRO 4500 worker remains unmeasured.

All 48 confirmation tasks and collector 65003093 completed. The collector matched selection SHA and all shared fold receipts, and validated 144 prediction files. `confirm.json` SHA-256 `640885e950f3e91841c1b6951bd64529e1389a608c5cbd135b6e82b437b0bf2d`.

## Additional confirmation on orders 71172–71179

Replay 65003770 was active at the most recent Leonardo readback. The additional result is pending.
