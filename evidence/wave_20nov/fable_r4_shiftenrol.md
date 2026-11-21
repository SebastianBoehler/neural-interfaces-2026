# Fable Round 4 Bet 2 · shift-robust enrolment

[Preregistration](../../docs/LEDGER.md#fable-45-bet-2--shift-robust-enrolment--21-november) · [Dispatch](DISPATCH_21NOV.md)

The clean returning folds are `production-20261120-32a94ab7/hardcb/{ret_P04,ret_P10}`. The runtime
is the signed hard-only fused-conv lead with the clean fold actors and the verified Team B fold banks.
The same-day diagnostic replaces one person's enrolment arrays and replays disjoint test windows.
Population weights and the 25% Team B blend stay fixed. This test measures recoverable day gap on
two people.

## Inputs and smoke · 09:27 CET

- Design: `PROPOSALS_4.md` at `b0f46ff` and `PROPOSALS_5.md` at `afbfe3b` in the cleanroom.
  The user retained the 3 pp day-gap threshold and set the later keep rule to mean > 0 pp with
  neither person below −0.5 pp.
- Lead ZIP SHA-256 `7017bd5c8a707e7233fe6b1e2df4376c28d49f6613174ec82c0fdbf700bc3f07`.
  Public EEG SHA-256 `2be4187fe33a5d476145cbb6d46164093b9b5c337209482503761042343e1327`.
  Fold files matched their Leonardo `pkg.sha256` manifests. Each nine-member Team B bank
  matched the pinned 71124 tape's source hash and roster.
- Local `python replay/check.py` printed OK. Galvani A100 smokes 3146904 and 3146905 matched every
  pinned 71124 label (375 total), matched the weights-only zero control exactly, and exercised
  the same-day personal route, own-space heads and both Team B CUDA graph shapes on 72 test
  windows per person. Neither smoke read an accuracy score.
- Pre-check jobs 3146906 and 3146907 use source `b9026ab4`. The fixed train/test counts were
  P04 94/95 and 93/95, P10 92/94 and 99/100. Each test half starts with an empty solver state.
  The same-day arm knows the labels of its disjoint enrolment half. The other arm uses the
  signed visit-1 fit. The personal galleries can affect recognition time as part of the test.

## Interim and decision · 09:31 CET

**Stop Bet 2 at the pre-check.** Same-day enrolment gained **+1.54 pp** over visit-1 enrolment
(80% paired people-then-orders interval **+0.19 to +2.88 pp**). The registered threshold was +3 pp.
Both Galvani A100 jobs completed in 3m20s or less. All eight paired replays used the full Solver
and the 25% Team B blend. The four cells for each person and order have equal weight.

| Person | Visit | Visit-1 enrolment BA | Same-day enrolment BA | Change (pp) |
|---|---:|---:|---:|---:|
| P04 | 2 | 69.03% | 67.33% | −1.70 |
| P04 | 3 | 71.15% | 73.23% | +2.08 |
| P04 | Mean | 70.09% | 70.28% | +0.19 |
| P10 | 2 | 85.76% | 91.53% | +5.76 |
| P10 | 3 | 91.94% | 91.94% | 0.00 |
| P10 | Mean | 88.85% | 91.73% | +2.88 |
| Both | Mean | 79.47% | 81.01% | +1.54 |

The two order estimates were +1.27 and +1.81 pp. The unverified 20/80 P04/P10 mix gives +2.34 pp
on this same-day diagnostic. Validation covers two people and two orders. The same-day enrolment
half is smaller than visit 1. Each test half is shorter than a full later visit, which affects
the length of causal history. P10 visit 2 accounts for most of the gain.

The augmentation implementation generates eight transformed copies with 20–40 reference windows
per copy. A local transform smoke checked this on both returning people’s visit-1 tensors. The day-gap rule stopped augmented
training, fresh-order evaluation and packaging. Keep the signed fused lead for this bet.

## Receipts

Galvani: `/mnt/lustre/work/gehler/gfh527/ni26_fable4_shiftenrol_20261121`. The collector verified the
registered source/test split, delivery rows, decision classes and official cell metric for each case.

| File | SHA-256 |
|---|---|
| `day-P04.json` | `6ac8d3aa4be17e4c9d44d9d5bdd6845ee1128a455ebf4afeafa8821088c3e352` |
| `day-P10.json` | `f26cf1abda0652ee28b0ccb64277518d1f7df692cdfe2bd56d117371a112122c` |
| `day-receipt.json` | `2b18ed88be61e9b361987e5a710978d5528b0aa5470f47b29614d46f050d282c` |
