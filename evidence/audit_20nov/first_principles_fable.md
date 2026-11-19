# Fundamental audit, hypothesis 3 · first principles (Fable, 19 Nov 2026, 21:00 CET)

Question: is our understanding of the problem wrong, and would a simpler or different design beat the stack?

**Answer.** No hidden structural lever exists in the places named in the brief. Class-balance transduction is worth at most
+0.6 pp even non-causally with the whole session in hand. The per-window streaming constraint is real. A simple EA + tangent-space
baseline reaches 50 % where the stack reaches 74 %. The 3 pp to 0.86 sits in one place: the cross-person representation for the
half of new people our ten source people do not cover (75 % of the sealed error mass). The hundreds of ±0.3 pp experiments
mostly swapped sibling members that share 90 % of their errors.

Everything below is DEV on the ten public people. Instruments: cap250 stock from the corrected Friday replay
(`ni26_friday_c25d_held_v1_20261119/full0.npz` and `within0.npz`, held-new, 2,655 windows, 28 cells, one order each) and Lane A's
returning stock (`ni26_friday_return_fbl2_score_v2_20261119/matched_c21r.npz`, P04/P10 visits 2–3 with visit 1 enrolled, 762
windows, 8 cells). Intervals resample people (10,000 draws, 80 %). Two returning people cannot carry an interval. Scripts were
run locally on the banks (not committed, per the task scope); the Galvani scripts are pinned by hash in section 5.

## 1. Error decomposition

| View (full0, stock terminal 73.74) | Numbers |
|---|---|
| Person, all cells | P01 72.2 · P02 88.3 · P03 73.5 · P04 50.0 (6 cells) · P05 70.6 · P06 81.3 · P07 87.9 · P08 87.3 · P09 84.7 · P10 78.8 (6 cells) |
| First session only, person mean | **77.35**. Sorted: 55.2, 70.6, 72.2, 72.4, 73.5 then 81.3, 84.7, 87.3, 87.9, 88.3 |
| Interface | BrainHero 73.97, Graz 73.51 (14 cells each): no interface effect |
| Confusion, true → predicted (cell-weighted %) | MI 80.2 / 8.5 / 11.3 · SUB 11.0 / 72.8 / 16.2 · WORD 16.4 / 15.3 / 68.3 |
| Delivery position, plain accuracy | windows 0–9 70.0 · 10–19 77.1 · 20–39 72.1 · 40–79 71.4 · 80–119 73.2 · 120–159 76.1 · 160+ 75.4 |
| Stages | raw 73.94 → count filter 73.86 → terminal 73.74 (within0: 77.61 → 78.32) |
| Returning, visit 1 enrolled | P04 85.4, P10 94.5, eight cells 80.2–98.1, mean **89.98**. Same cells held-new: 64.72 |

- **The error is person-level.** The lower five people average 68.8 and carry 69 % of the first-session error; the upper five
  average 85.9. P04 is at chance-plus on all three of his sessions (45–59 %) and at 85 % once one labelled session of his is
  enrolled. His classes are separable and stable across days, but differ from the population pattern.
- **The 28-cell dev mean over-weights P04**: his six cells hold 41 % of its error mass. In a first-session view he holds 20 %.
- **Not SUB vs WORD alone.** SUB↔WORD is 40 % of the off-diagonal mass; MI is involved in the other 60 %. WORD is the worst class.
- **Warm-up is small.** The last two fifths of a session score 75.8 against 73.7 overall. If every window were decided with
  late-session knowledge, held-new would gain about +2.1 pp. Labels from one earlier session are worth +25 pp on the same cells.
- **The count filter and tail do nothing under full shuffle** (−0.2 pp) and add +0.7 pp under run-contiguous delivery.

**Sealed composition check.** 0.57 × 77.35 (new, first sessions) + 0.43 × 89.98 (returning) = **82.78**; sealed measured 82.99.
The dev model of the sealed set is consistent, and it implies the sealed error splits **12.9 pp on new people and 4.3 pp on
returning people**. The same sum with within0 (first-session mean 80.94) gives at least 84.8, which is 1.8 above the sealed
score: the sealed delivery behaves like a full-session shuffle, not like run-contiguous blocks. This rests on the repo's 57 %
share and on P04/P10 standing in for all ten returning people; both are assumptions.

## 2. What a 0.86 system must do

Sealed error must fall from 17.0 to 14.0.

| Path to 0.860 | New people | Returning people |
|---|---|---|
| New only | 77.4 → 82.7 (lower five people from 68.8 to about 79.4) | 90.0 |
| Returning only | 77.4 | 90.0 → 97.1 (not credible: P10 is at 94.5) |
| Mixed | 77.4 → 80.9 | 90.0 → 92.5 |
| Third place, 0.847 | 77.4 → 80.4, or 79.4 with returning 91.4 | |

So first place needs roughly a quarter of the new-person error removed. Three candidate mechanisms from the brief, measured:

**Class balance / optimal transport / Hungarian (bank test, labels used only for scoring).** Class counts are equal per cell
(32/32/32 Graz, 29–35 BrainHero), and stock predictions are unbalanced (P01 Graz predicts 50/33/13). Even so:

| Arm on stock terminal posteriors | full0 | within0 | returning |
|---|---|---|---|
| Whole-session Hungarian, equal counts (non-causal) | +0.43 [−0.03, +0.79] | −0.01 [−0.63, +0.59] | +0.92 (2 people) |
| Per-cell Hungarian, interface given (non-causal oracle) | +0.04 [−0.34, +0.44] | −0.41 [−0.85, +0.18] | +0.42 |
| Whole-session Sinkhorn (non-causal) | +0.59 [+0.21, +0.87] | −0.63 [−1.09, −0.23] | +0.52 |
| Causal prefix Sinkhorn from row 20 (Lane D-c form) | +0.37 [+0.04, +0.67] | −0.09 [−0.33, +0.12] | −0.26 |
| Causal prefix Hungarian | +0.06 [−0.47, +0.64] | −0.34 [−0.64, −0.08] | −1.04 |
| **Label-fitted** affine map of log posteriors per session, leave one run out | **−0.84** | **−2.22** | n/a |

The last row is the decisive one: a per-session remap of the three posteriors does not help even when fitted on the session's
own labels. No posterior-space operation (balance, prior shift, confusion correction, temperature) can supply the gap. The
errors are in the features. This agrees with Lane D (D-c +0.52 / −0.21 / Scherer −0.18; whole-session Gaussian +0.29).

**Is the causal constraint self-imposed?** No. FAQ, "What reaches predict": "the model receives no future window or label".
"What must the model return": one integer per window, validated per call, input `(1, 41, 480)`. A decision cannot be deferred
or revised. What is free: unlimited label-free computation on the delivered prefix (re-clustering every earlier window at each
step is allowed), bounded only by the time limit. The FAQ sentence "no shuffling within a stream" is superseded by the
organisers' 13 Nov statement.

**Session-level transduction.** The non-causal bounds above and Lane D's whole-session Gaussian fit put it at ≤ +0.6 pp in the
stack's posterior and PC spaces. Whether a learned embedding holds more is test T-A (section 5).

**Which component would have to change.** Not the tail, the count rule, the mixer weights or class balancing. The source
members themselves: across the seven arms in the bank (stock, FBL2, hard, motor, joint, joint+motor) **89.8 % of stock's
errors are errors in every arm**, and a label oracle choosing among all arms reaches only 76.40 (+2.66). Past oracle ceilings
realised about 25 %. The search has been inside one family whose members fail on the same windows of the same people.

## 3. Simple baseline against the stack

Five bands (4–8, 8–13, 13–20, 20–30, 30–45 Hz), covariance per window, Euclidean alignment, tangent space, standardised L2
logistic regression (C = 0.01), leave one person out, official cell metric, full0 delivery. Run locally, 45 s.

| System | Held-new, 28 cells |
|---|---|
| EA per session (non-causal) + TS + LR | 50.13 |
| EA over the delivered prefix (causal) + TS + LR | 49.71 |
| Same, whole-session balanced self-training, 3 rounds (non-causal) | 51.8 / 52.2 / 50.7 |
| Same, EA per run with run identity given (oracle) | 52.02 |
| Same features, **own session's labels**, leave one run out | 72.37 |
| Other people + own session's labels ×10, leave one run out | 62.72 |
| cap250 stock, label-free | **73.74** |

The stack adds **+23.6 pp** over the simple design and beats that design even when the simple one is given the session's own
labels. The simple-design hypothesis is refuted in this form. Limits: my baseline has no content below 4 Hz and no learned
filters; a tuned classical pipeline may reach 55–60, still far below. Run identity could not be recovered without labels
(Ward clustering of log power, ARI 0.05 against runs, 0.09 against interface), so latent-run alignment is not a cheap lever.

## 4. Three most likely fundamental reasons, ranked

**R1. Population coverage of the source model (about 75 % of the sealed error).** Ten source people; half of new people fall
outside what they span, and nothing label-free inside one session recovers them. The earlier learning curve was still steep
(old ledger N12: +3.86 pp from 7 to 9 people, standalone core1, recording order). Our foundation-model members stand at 55 %
standalone, below our own from-scratch members, so "large pretraining" has not been tested in a working form; a leader who
has it working would gain exactly here. Test T-B re-measures the slope under full-shuffle delivery with causal alignment.

**R2. Label-free in-session adaptation is close to inert.** Late minus early windows is about +3 pp, a labelled session is
worth +25 pp, and whole-session look-ahead in posterior space is ≤ +0.6 pp. Either the session's class structure is not
discoverable without labels (capped, stop spending here) or it is discoverable only in a learned feature space we never
adapted in (open). Test T-A separates the two.

**R3. The search ran below the resolution of its instrument, in a regime the sealed set does not use.** On the seven-arm bank,
choosing the best arm on five people and scoring it on the other five gives +0.52 pp against +0.44 for an arm picked at random
(within0: +0.38 against +0.36); per-person sd of one arm's gain is 0.85 pp, so se ≈ 0.27 over ten people. Arm gains are a
common +0.4 pp plus noise, and selection among them adds nothing. Dev increments below 1 pp transferred at 0.46. Six of eight
pinned orders were run-contiguous, where the stack is 4.6 pp better and the tail stages help; the sealed arithmetic matches
full shuffle. Test (done here, no compute): the split-half selection check and the composition sum above. A follow-up worth
one CPU job is raw against terminal on eight full orders once fresh full-order banks exist; expected −0.2 pp, not decisive.

Not ranked but still the largest lever with a known sign: returning people are 43 % of cells at 90 %. Lane A owns it.

## 5. Pre-registration of the two Galvani tests (written before any full run was scored)

Release `/mnt/lustre/work/gehler/gfh527/ni26_audit_fp_v1_20261120`; `audit_fp.py` SHA-256
`8180feb8014dbeef6a6577403ec31edc42e77ae2e228c3a1e35b5043ad3c90b4`, `audit.sbatch`
`1fec060360cc4256030db592492e1859ecdb4a6cabc22db949110819bb96b99a`. One compact net (temporal conv 40 × 31, spatial conv,
BatchNorm, square, average pool, log, linear; 40 epochs AdamW), 1–45 Hz, causal Euclidean alignment over the delivered prefix
in the pinned `full|0|all_loso` order, leave one person out, official cell metric, 28 cells, seeds 0 and 1, people-resampled
paired 80 % intervals. **Scope: standalone diagnostics of a ceiling on one small net. They cannot close a direction or promote
a stage; the stack was not in the loop.** A supported outcome only justifies a full-system form.

**T-A, learned-embedding transduction against the label ceiling.** Mechanism: if a new person's three classes form separable
groups in a learned embedding, balanced self-training in that embedding moves predictions toward the label ceiling; if the
groups exist only with labels, it cannot. Arms against the source net: (a) label ceilings, leave one run out: linear probe on
the session's own embedding, and source rows plus own rows ×10; (b) non-causal whole-session: balanced Hungarian on
posteriors, balanced LDA self-training in 24 source PCs (primary), prototype re-estimation (T3A); (c) the same two as causal
prefix arms from row 30, refit every 10 rows.
- Predictions: source net 58 % [50, 66]. Source + own labels: +10 pp [+4, +16]. Own-label probe: +6 [0, +14]. Whole-session
  LDA self-training: **+1.0 [−1.0, +3.0]**. Causal prefix LDA: +0.5 [−1.0, +2.0]. T3A: +0.5 [−1, +2] in both forms.
- Rule: "open" if whole-session LDA self-training minus source has a people-80 % lower bound above +1.5 pp on both seeds and
  realises at least 30 % of the source + own label gain. "Capped at this level" if its upper bound is below +1.5 pp on both
  seeds while the label gain is at least +5 pp. Anything else is undecided. If the label gain is below +5 pp the net is too
  weak to carry the question and the test is void.

**T-B, source-people learning curve.** Train on k = 3, 5, 7 (two random subsets each) and 9 of the other people, score the
held person. Mechanism: if new-person error is coverage, accuracy still rises steeply between 7 and 9 source people.
- Predictions: cell means 46 / 51 / 55 / 58 % for k = 3 / 5 / 7 / 9. Gain 7 → 9: **+1.8 pp [+0.3, +3.5]**; 5 → 9: +4.5 [+2, +8].
- Rule: "still steep" if the 7 → 9 gain has a people-80 % lower bound above +0.5 pp on both seeds; "saturating" if its upper
  bound is below +0.5 pp on both; otherwise undecided. A steep curve supports R1 and says the remaining budget belongs to
  more same-task people or a working pretrained member, not to stages. Random subsets add variance at k < 9.

Smoke jobs (2 folds, 3 epochs, code path only): 3138367 (T-A), 3138368 (T-B); the first pair 3138365/6 stopped at a tensor-layout error before any score. Full job ids and results are appended below.
