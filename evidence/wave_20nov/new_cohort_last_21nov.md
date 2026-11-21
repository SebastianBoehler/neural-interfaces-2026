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

The label-blind B control uses the P01/1 fold on 71800. Its stored count posterior differs from RouteB beginning at window 41, with 100 affected windows also using CSP. Compare the reconstructed B posterior directly with the Last Push tail with CountFilter bypassed. Require exact labels, maximum probability error ≤ 1e−7 and at least one active decision change. No accuracy is computed by this control.

Control 65046842 loaded the older Bottleneck snapshot of Last Push, which lacks the bypass hook. Retry 65047298 uses the explicit Last Push tail staged for the quarter-diet lane. The capture array keeps the original snapshot for stock replay.

Retry 65047298 matched the direct bypass on all 190 labels and changed four decisions versus native. The maximum posterior difference was 2.22e−16.

## Fit freeze

The quarter fold reference scored 74.551% new, 81.513% returning and 78.032% at 50/50. A froze `k=7, w=0.3` from 71800–71807 at +0.034 pp at 50/50. The other settings `(15, 0.3)`, `(7, 0.6)`, and `(15, 0.6)` gave −0.026, −0.114, and −0.141 pp. Fixed B gave +0.010 pp. Fit receipt SHA-256 `73a3e3ed5d683577230d16ca13b5bcfa9bde065d50c517205fee8996356dc0ff`.

## Held orders 71808–71815

**Keep the signed quarter diet.** Neither arm met the +0.15 pp 50/50 threshold or a positive lower bound. The quarter fold reference scored 74.397% new, 81.981% returning and 78.189% at 50/50, exactly matching the earlier quarter-diet receipt.

| Arm | New Δ pp [80%] | Returning Δ pp [80%] | 50/50 Δ pp [80%] | P04 / P10 Δ pp | New / returning label changes |
| --- | ---: | ---: | ---: | ---: | ---: |
| A · Neighbour vote, `k=7, w=0.3` | +0.0008 [−0.279, +0.247] | 0.000 [0, 0] | +0.0004 [−0.140, +0.123] | 0.000 / 0.000 | 763 / 0 |
| B · Count bypass before recognition | +0.0210 [−0.056, +0.105] | 0.000 [0, 0] | +0.0105 [−0.028, +0.052] | 0.000 / 0.000 | 72 / 0 |

Each split covered 21,240 held-new and 6,096 returning decisions. All 32 held returning streams recognised. Both arms kept every returning label, including 374 decisions before the latch. The historical full bypass gained +0.27 pp on returning in the old ba_six control. Quarter already bypasses CountFilter from the latch onward, and the selective test added +0.021 pp new points. These are paired order comparisons on reused public people.

Slurm smoke 65046279, all 47 array tasks 65046285 and bypass control 65047298 completed. Scorer source `c6dad1fb` and live control fix `2170723d` precede the held result. The fit and held scorers both passed authenticated delivery, zero-control and prefix checks. The held receipt SHA-256 is `593a731d767b21cebe8797e9763045641d67a3345b086a26e5be9f411be3093b`. The 48-file fit and 48-file held capture manifests, each a sorted list of file digests and basenames, hash to `a29938db51cffcade848fc4508a980905fc86516fa86b6e0bc7e757c9307737b` and `564937cdfc1ed92622ffdcfd337254ffabc2d413d56be8bb9abb47f1858d3ae2`.

The promotion rule triggered no derivative ZIP or production qualification.
