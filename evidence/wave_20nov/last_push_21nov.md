# Team B weight and early adaptation · 21 November 2026

[Preregistration](../../docs/LEDGER.md#final-weight-and-early-adaptation--21-november-registered-before-new-scores) ·
[Dispatch](DISPATCH_21NOV.md) · [ba_six fixtures and qualification](leonardo_fill_21nov.md)

The organiser-confirmed sealed proxy is 50% held-new and 50% returning. The dev instrument has ten held-new people and two returning people, P04 and P10. It uses official cell-balanced accuracy and 20,000 paired bootstrap draws that resample people and then orders. The fit and held-out halves reuse the same people.

## Covariance prior already ran

Leonardo ran the registered population covariance prior inside the live fused nine25 encoder on 71624–71631. The two live smokes (65014924/27), 48 fold/order chunks (65014928), and report (65014933) completed. Source `9bf1dc24`. Report SHA-256 `5c8db8b7db2408baab42c26192d578cb3fe5ed25c45954f1faee1d52b5899a84`.

| Variant | Held-new pp [80% CI] | Returning pp [80% CI] | P04 / P10 pp | 50/50 pp [80% CI] |
|---|---:|---:|---:|---:|
| Population covariance prior − fused nine25 | −0.05 [−0.24, +0.13] | −0.14 [−0.30, +0.02] | −0.24 / −0.03 | −0.09 [−0.22, +0.02] |

The held-new 1–40 delta was −0.38 pp. The registered screen stopped and left the confirmation orders unopened. This result resolves the registered nine25 implementation. [Original mechanism](breakthrough_21nov.md)

## New weights and early-route variants

Team B mixes before RouteB, Rebalance, SessionEM, TSEM, count filtering and CSP. The ba_six package then applies its 10% MANAS term. The scorer captures the hard mixer output, Team B posterior, RouteB inputs and MANAS posterior, then reruns all affected stages. The signed hard path has `v6r=false`, so its upstream fine-tune and recognition labels come from the independent shadow path. Every stream checks exact 0.25 route scores, terminal scores and labels against the live Solver and checks all reference labels against Leonardo fill's stored shuffled delivery.

The fit uses 71800–71807. The held-out half uses 71808–71815. Each roster selects a weight on the fit half from 0.25, 0.30, 0.35, 0.40 and 0.50. The early variants halve Rebalance alpha, quadruple Rebalance prior mass, or delay SessionEM to delivery 41 while preserving its history. All three restore the signed parameters at delivery 41. The primary early candidate for each roster is chosen on fit orders.

### Dispatch receipts

- Local v3 printed `OK`. Leonardo run `cap250/runs/last-push-21nov`, source `62c0aa4d`: pinned P01/P04/P10 × two roster smokes 65025856–65025861. Fit 65025911 waits for these and the collaborator's reference arm. Freeze 65025913 → held 65025914 → report 65025915. Each dependency is `afterok`.
- Galvani run `ni26_last_push_20261121`: production archive smokes 3147652/53 reached v3 `OK`, then stopped on an omitted source dependency before loading either candidate. The complete archive from the same revision supplied the dependency, and retries 3147655/56 were submitted.
- The smoke guard found `em_nmin=20` in both signed archives. The initial 10-window assumption came from a source default. Amended the ledger before scoring and fixed the replay at `78b5d73c`. The native updates affected by the delay are deliveries 21–40.
- The capped Mac replay passed the signed nine25 source identity, active variants, and causal-prefix checks on 45 actual P01 windows. Galvani jobs 3147669/70 both passed exact 0.25 tape parity on 190 windows and live labels and route scores on the 72-window prefix for 40% and all three early variants. A100 fold checks use the actual shuffled reference tapes.
- Leonardo replacement chain uses the same orders and fitted assets. Smokes 65026147, 65026148, 65026154, 65026181, 65026182 and 65026183 gate fit 65026184. Freeze 65026185 → held 65026186 → report 65026187.

## Fit on 71800–71807

Fable Round 3's frozen oracle gained +8.50 pp [80% +4.95, +12.57] on the first ten held-new deliveries and about +1 pp after delivery 40. The oracle changed multiple adaptation stages. Fable Round 2's selected position ramp gained +0.08 pp [−0.07, +0.20] at 50/50 across its own held-out orders. [Round 3](fable_round3.md) · [Round 2 P3](fable_round2_tables.md#p3--frozen-confirmation-on-orders-7118871195)

The 96 fit tasks completed. Collector 65026185 froze 40% and delayed SessionEM for both rosters. `fit.json` SHA-256 `96652b793c28352d1d5b09b6db6d8ff0fc9011f58289b66dc2cdfca1de38458d`. Each row is relative to that roster's unchanged 25% Solver.

| Roster | Arm | Held-new pp [80% CI] | Returning pp [80% CI] | P04 / P10 pp | 50/50 pp [80% CI] |
|---|---|---:|---:|---:|---:|
| nine25 | 30% | −0.18 [−0.34, +0.01] | −0.11 [−0.36, +0.13] | −0.33 / +0.10 | −0.14 [−0.29, +0.01] |
| nine25 | 35% | −0.12 [−0.31, +0.14] | −0.09 [−0.59, +0.37] | −0.52 / +0.34 | −0.10 [−0.35, +0.16] |
| nine25 | **40%** | −0.11 [−0.34, +0.18] | +0.16 [−0.17, +0.45] | −0.06 / +0.37 | **+0.02 [−0.17, +0.23]** |
| nine25 | 50% | −0.48 [−0.81, −0.12] | +0.13 [−0.37, +0.58] | −0.26 / +0.51 | −0.18 [−0.47, +0.11] |
| nine25 | Rebalance alpha | −0.11 [−0.43, +0.25] | −0.14 [−0.43, +0.13] | −0.34 / +0.06 | −0.13 [−0.35, +0.11] |
| nine25 | Rebalance prior mass | −0.18 [−0.41, +0.10] | −0.21 [−0.43, +0.03] | −0.34 / −0.07 | −0.19 [−0.35, −0.01] |
| nine25 | **SessionEM delay** | −0.05 [−0.29, +0.14] | +0.10 [−0.15, +0.38] | +0.30 / −0.09 | **+0.02 [−0.14, +0.18]** |
| ba_six | 30% | −0.17 [−0.32, −0.01] | +0.20 [+0.03, +0.40] | +0.30 / +0.11 | +0.02 [−0.09, +0.14] |
| ba_six | 35% | −0.35 [−0.54, −0.16] | +0.52 [+0.08, +1.03] | +0.96 / +0.08 | +0.09 [−0.16, +0.34] |
| ba_six | **40%** | −0.30 [−0.58, −0.05] | +0.54 [+0.16, +0.93] | +0.86 / +0.21 | **+0.12 [−0.11, +0.35]** |
| ba_six | 50% | −0.66 [−1.10, −0.28] | +0.60 [+0.30, +0.89] | +0.60 / +0.60 | −0.03 [−0.31, +0.22] |
| ba_six | Rebalance alpha | −0.21 [−0.50, +0.10] | −0.10 [−0.30, +0.08] | −0.15 / −0.06 | −0.16 [−0.33, +0.02] |
| ba_six | Rebalance prior mass | −0.16 [−0.44, +0.14] | −0.05 [−0.23, +0.13] | −0.07 / −0.03 | −0.11 [−0.25, +0.05] |
| ba_six | **SessionEM delay** | −0.13 [−0.36, +0.07] | +0.12 [−0.03, +0.30] | +0.23 / +0.01 | **0.00 [−0.15, +0.12]** |

## Held-out result on 71808–71815

Both 96-task arrays and both collectors completed. The held-out collector 65026187 finished by 14:12 CET. `held.json` SHA-256 `1ebee841444d03991ff1be3d2e662e67e439c2722747087ff12f1ca3e6bec5fd`. First-40 effects use the same full-session class weights restricted to the prefix.

| Roster | Arm | Held-new pp [80% CI] | Returning pp [80% CI] | P04 / P10 pp | 50/50 pp [80% CI] | First 40 new / returning pp |
|---|---|---:|---:|---:|---:|---:|
| nine25 | **40%** | +0.10 [−0.49, +0.66] | −0.06 [−0.35, +0.25] | −0.15 / +0.04 | **+0.02 [−0.32, +0.35]** | −0.20 / −0.54 |
| nine25 | Rebalance alpha | +0.21 [−0.10, +0.53] | +0.10 [−0.13, +0.34] | +0.23 / −0.03 | +0.16 [−0.04, +0.36] | +0.59 / +0.25 |
| nine25 | Rebalance prior mass | +0.17 [−0.13, +0.50] | +0.10 [−0.06, +0.28] | +0.20 / 0.00 | +0.13 [−0.02, +0.31] | +0.68 / +0.09 |
| nine25 | **SessionEM delay** | +0.18 [−0.13, +0.45] | −0.13 [−0.46, +0.19] | −0.42 / +0.16 | **+0.02 [−0.19, +0.22]** | −0.02 / 0.00 |
| ba_six | **40%** | −0.24 [−0.62, +0.12] | +0.07 [−0.42, +0.52] | −0.35 / +0.49 | **−0.09 [−0.39, +0.20]** | −0.43 / +0.38 |
| ba_six | Rebalance alpha | −0.12 [−0.35, +0.17] | +0.02 [−0.25, +0.27] | +0.01 / +0.03 | −0.05 [−0.23, +0.16] | −0.26 / +0.01 |
| ba_six | Rebalance prior mass | −0.13 [−0.39, +0.18] | +0.07 [−0.14, +0.26] | +0.07 / +0.06 | −0.03 [−0.20, +0.17] | −0.22 / +0.32 |
| ba_six | **SessionEM delay** | −0.11 [−0.34, +0.10] | +0.25 [+0.03, +0.42] | +0.14 / +0.35 | **+0.07 [−0.11, +0.23]** | −0.42 / +0.62 |

Retain the 25% blend and original early-stage settings for this freeze. The largest cold point was nine25 Rebalance alpha at +0.155 pp, with an 80% lower bound of −0.039 pp and a negative fit result. Treat small effects as undecided. No variant met the accuracy rule for a new ZIP, official warm-up pair, or paired cold runtime test.
