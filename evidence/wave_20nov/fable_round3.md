# Fable Round 3 · 21 November 2026

[Preregistration](../../docs/LEDGER.md#fable-round-3--21-november-registered-before-scoring) · [Dispatch log](DISPATCH_21NOV.md) · [Scorers](../../scripts/fable_round3/)

Signed reference `final_stack_hard_safe_st_team_b_nine25_tbgraph.zip`, SHA-256 `6e09953709b2338507deaac1ec5f6afa4a21eef52de7814f7c2ae77f42e86ee9`. The protocol uses 28 held-new cells across ten people and eight returning cells across P04/P10. Fit orders 71300–71307 and held-out orders 71308–71315 shuffle the full session. The order blocks reuse the same people.

## Verification before scoring

- The local `replay/check.py` reproduced all six v3 reference scores and printed `OK`. The scorer matched the official balanced accuracy for each public test cell to 1e−12 and returned an exact zero on a paired diagnostic.
- Galvani A100 smoke 3146351 completed in 86 s. It exercised the signed lead on 72 public windows, verified native and cached Team B label parity, tapped pre-EM and terminal scores, applied the live rebalance and verified prefix and frozen query-order parity. The public smoke did not measure accuracy.
- Exact fold, Team B weight and tape inputs are on Leonardo. Batch SSH still returned `Permission denied` at 06:59 CET. Decisive scoring awaits the pinned-fold smoke.
- Galvani jobs 3140818 and 3140819 are `CANCELLED`. A full queue scan found the latter was the only child of the former and found no other dependents.
- Public smoke 3146352 completed in 86 s from `cabc1e92`. Bet 2's zero and unit weights reproduced the hard-base and Team B labels exactly.
- State-only smoke 3146353 completed in 53 s on Galvani. The signed lead processed 100 unrecognized Scherer visit-1 windows with eight fine-tuning refits and four CSP refits. The frozen checkpoint reproduced its labels when the query order was reversed. The worker read waveform tensors and session identifiers for this smoke.
