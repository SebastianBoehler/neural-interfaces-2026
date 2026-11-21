# New-cohort final review · 21 November 2026

## Protocol

Base `final_stack_hard_safe_st_team_b_ba_six_fc_fp16_turing_recognised_em_half_diet_quarter.zip`, SHA-256 `42fb63cf1ae9f54cf48315b00894b3788bc65400e75e06bd208ffe3753c43b07`. The base already skips CountFilter once the recognition latch is set. The [ledger registration](../../docs/LEDGER.md#last-new-cohort-review--causal-neighbours-and-count-bypass-21-november-registered-1750-cet-before-scoring) precedes scoring.

- **A, neighbour vote.** At delivery 30 and later, while the native recognition latch is unset, find earlier neighbours by cosine similarity. Subtract the earlier feature mean from the current feature and all past features. The feature concatenates the hard mixer and pooled Team B class probabilities. Add the log of the neighbours' mean terminal probability at weight 0.3 or 0.6. The fit grid includes 7 and 15 neighbours. The history stores the original terminal probabilities.
- **B, count bypass.** While recognition is unset, replace CountFilter's output with RouteB's output. Reapply the recorded CSP log-evidence and native 10% MANAS mix. CSP computes its adaptive state from the existing RouteB posterior, independent of the count argument. Both arms keep the quarter diet output from the actual latch onward.
- **Instrument.** Use ba_six fold captures and canonical quarter predictions on ten held-new people and returning P04/P10 visits 2 and 3. Fit 71800–71807, freeze A, then score that setting and B on 71808–71815. The order split reuses people. The metric gives each person × session × interface cell equal weight within its cohort, then averages the two cohort scores equally. Paired 80% intervals use 20,000 draws of people and orders nested within each sampled person.
- **Promotion.** Held 50/50 gain ≥ +0.15 percentage points (pp), lower 80% endpoint > 0, and returning gain ≥ −0.10 pp. A promoted ZIP also needs 379 recognised-window label identity, the official warm-up pair and unprofiled end-to-end timing ≤ 3,400 s on the worst observed normal RTX 2080 Ti node by 20:10 CET.

## Capture and checks

The [Bottleneck Review](bottleneck_review_21nov.md) supplies 71800–71803 and 71808–71811. Leonardo Slurm smoke 65046279 and dependent array 65046285 capture the remaining eight orders in `cap250/runs/new-cohort-last-21nov`. They reuse the same unchanged ba_six fold Solver and fixture hashes. The [Last Push](last_push_21nov.md) provides pinned native predictions and exact same-tape replay. The [quarter diet](fable_r6_diet.md) provides canonical post-latch predictions.

The local and Leonardo v3 guards printed `OK`. A local smoke on two archived fit orders verified regenerated delivery rows, captured native labels, matching fixture receipts, recognition ages, zero controls, 67-window prefixes and active effects before computing accuracy.
