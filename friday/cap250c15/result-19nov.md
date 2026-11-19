# F-C15 · Full-system robust outer rebalancing

19 November 2026 · Exact cap250 worker · FULL-SYSTEM EXTERNAL

The robust rebalancer lost **0.4927 percentage points** against stock and **0.4200 pp** against the same-magnitude scalar control. Both paired intervals excluded zero and both fresh orders lost. The beneficial direction prediction is refuted for this estimate on these people. Scalar remains undecided against stock. [Preregistration](../../docs/LEDGER.md#f-c15--bound-the-past-class-estimate-in-the-actual-outer-rebalancer) · [Worker and smoke](README.md)

| Arm | Official balanced accuracy | Change vs stock, pp [paired 80%] | 4411, pp | 5411, pp | Rescues / losses |
|---|---:|---:|---:|---:|---:|
| Stock cap250 | 48.0919% | — | — | — | — |
| Stock-direction scalar | 48.0192% | −0.0728 [−0.3009, +0.1620] | −0.0397 | −0.1058 | 20 / 23 |
| Robust | 47.5992% | −0.4927 [−0.9491, −0.0661] | −0.4563 | −0.5291 | 38 / 59 |

Robust minus scalar was −0.4200 pp [−0.8102, −0.0496], with −0.4167 and −0.4233 pp by order and 32 rescues against 50 losses. The control used the same history-dependent norm. The loss is therefore specific to replacing the native correction direction in this comparison. The experiment covers nine reused Scherer people, both visits and 4,260 matched decisions per arm. Paired intervals resample people, visits and delivery orders.

| Contrast | Motor imagery, pp | Subtraction, pp | Word generation, pp | Warm A100 median / worst vs stock |
|---|---:|---:|---:|---:|
| Scalar vs stock | +0.0694 | −0.9226 | +0.6349 | 0.9999× / 1.0367× |
| Robust vs stock | +0.7639 | −2.3115 | +0.0694 | 0.9929× / 1.0392× |
| Robust vs scalar | +0.6944 | −1.3889 | −0.5655 | — |

Both timed arms met the A100 threshold. The worst robust session was Scherer H visit 2 on order 5411. The worst scalar session was E visit 1 on the same order. The accuracy gate withheld the RTX follow-up.

Each treatment reached the outer rebalancer on every window. Its correction direction differed from the scalar control by more than 0.01 on 2,778 windows. The largest norm difference was 1.12e-16. The robust rebalancer's maximum posterior change was 0.07334 against stock and 0.06193 against scalar. All work counters matched on all 36 sessions. Protected inner-stage posterior drift was at most 5.57e-6, exact protected decisions and refit EEG hashes matched, and independently repeated native warm actors drifted by at most 8.90e-5.

Galvani `3132752` finished its final producer at 01:01:07 CET. Collector `3132753` authenticated all nine receipts, sealed the decisions before reading gold, and completed at 01:01:24 CET.

| Artifact in `ni26_friday_c15_rebalance_v2_20261119` | SHA-256 |
|---|---|
| `before_labels.json` | `5f2133d4f0ac3921a2ebd36037eca6c8af8c36d4dfbf00e50a4231282bb220e0` |
| `result.json` | `e79d2a804a0ca75891d19feca03902146ed1ea271a72e1147e99a70040c060cd` |
