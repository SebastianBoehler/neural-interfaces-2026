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
