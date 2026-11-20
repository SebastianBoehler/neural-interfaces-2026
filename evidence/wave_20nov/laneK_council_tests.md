# Lane K · Council decisive tests (Codex sections 2, 3, 4)

Registered Fri 20 Nov 2026 ~10:30 CET, before any arm of this lane was scored. Source: [codex.md](../council_20nov/codex.md) sections 2-4. Instrument: the ten reused held-new
people, corrected Friday chain (`submissions/final_stack/chain` consumer), paired people-resampled 80 % intervals (as `score_final.py`), full0 only (the B5 banks have no within0). Weights of
the sealed mixture are unverified and not applied. Not covered by any test here: returning people, within0, the exact-ZIP path, timing.

## Test 3 · different source subsets per seed (code `laneK/final22k.py`, `laneK/score_k.py`)
Banks: Galvani `ni26_laneB5_20261120/banks` (each `bank_N5s{s}` = 3 seeds trained on the same 5-person subset s per held person).
- n9: 3-seed N9 ensemble. n5s: N5 subset 0, seeds 0-2 (the shared-subset control). n5d: seed s of subset s, s = 0,1,2 (diverse subsets), probabilities averaged before the chain.
- Consumer arms, all MASS 5/6 of the raw slot: n9, n5s, n5d alone (B5 swap form) and 0.8*X + 0.2*CBraMod (hardcb form: the four hard shares replaced by X, CBraMod share kept). Controls in the same run: stock, identity (bitwise equal to stock), hard, hardcb.
- Entropy match: n5dT = n5d with posteriors power-scaled so that the mean entropy equals the N9 ensemble's; arm n5dTcb. Disagreement diagnostic (raw banks): accuracy, entropy, member disagreement and any-member-right restricted to stock terminal misses.
- Predictions (pp, paired): n5dcb - hardcb +0.05 [-0.45,+0.55] (codex); n5scb - hardcb -0.15 [-0.6,+0.3]; n5d - n5s +0.15 [-0.2,+0.5]; n5d - n9 +0.25 [-0.1,+0.6] (B5 gave n5 - n9 swap +0.39 with shared subset: if diversity adds, n5d >= n5s); n5dTcb - n5dcb -0.1 [-0.4,+0.2].
- Zero-change: identity equals stock bitwise; hard and hardcb reproduce the 19 Nov receipt (hardcb - stock +0.514, hard +0.331 full0); n9 alone reproduces B5 N9 swap (-0.15 vs stock, 73.59).
- Rules: refuted if the paired upper bound of n5dcb - hardcb <= 0 or of n5dcb - n5scb <= 0. Qualified for follow-up (Leonardo fits, within0, external) if both lower bounds > 0 and n5dcb - hardcb >= +0.3. Else undecided. If n5dTcb retains less than half of n5dcb - n9cb, the gain is calibration, not diversity. No upload effect before Sat 08:00 unless qualified and timing-neutral (same member count: yes, time-neutral).

## Test 2 · within-window segment dispersion
Mechanism, forecast and kill rule exactly as codex section 2: dispersion of the four native FBC segment logit contributions of one member, shrinkage diagonal class densities (shared variance, 20 prior windows per class) on frozen past shadow pseudo-labels, fixed 0.10 bounded log-likelihood correction after RouteB, start at 8 effective observations per class; arms identity / whole-window-only / dispersion, genuine vs age-matched donor history. Forecast +0.12 [-0.35,+0.60] pp over hardcb. Refuted if the paired upper bound <= 0 vs either matched control; qualify if > +0.3 against both. Check first that segment dot products plus bias recover the logits and that class buffers fill and change eligible labels before gold.

## Test 4 · alpha-peak landmark
Step 1 (no labels): peak (aperiodic-removed, 7-13 Hz, posterior sensors) agreement between disjoint earlier-window halves at ages 40 and 80, genuine vs matched-age donor history. If genuine does not beat donor, stop (refuted form). Step 2 only then: compact source-only band-power heads at fixed and peak-relative (+-1 Hz bounded) bands, 1/6 raw-family share in the complete consumer, fixed / genuine / donor. Forecast +0.08 [-0.40,+0.55] pp vs fixed insertion; nonpositive paired upper bound vs fixed or donor refutes.

### Test 4 step 1, exact form (`laneK/alpha_peak.py`, registered before it ran)
Per held-new person (P01-P10, full0 delivery order `full|0|all_loso`), history of the first a = 40 and 80 delivered windows split into two disjoint halves (interleaved: even/odd positions; sensitivity: first/second half in time). Per half: Welch PSD (240-sample segments, 50 % overlap, 0.5 Hz bins) per window, mean over windows and over posterior channels (names starting P or O), log10; aperiodic line fitted on 3-6 and 14-30 Hz bins; peak = argmax of the residual on 7-13 Hz (parabolic refinement); no labels, no gold. Replication error = |peak(half A) - peak(half B)| of the same person (genuine) against |peak(half A of person) - peak(half B of donor)| averaged over the nine other people as donors at the same age (donor). Reliable = residual height >= 0.1 log10 units in both halves; also report the share of reliable cases.
Prediction: genuine median error 0.6 Hz (80 % 0.3-1.2) at age 80, 0.8 at age 40; donor 1.3 Hz at both; donor minus genuine +0.5 Hz [+0.1,+1.0] at 80, +0.4 at 40. Rule: the form is dead (stop, step 2 not run) unless donor minus genuine has a people-resampled 80 % lower bound > 0 at both ages (cases where both are reliable); a lower bound above 0 at only one age is undecided.

## Results (Fri 20 Nov ~12:30 CET; ten held-new people, full0 only, paired people-resampled 80 % intervals, pp; receipts on Galvani `ni26_laneK_20261120`, `ni26_laneK4_20261120`, `ni26_laneK2_20261120`)
Guards: identity equals stock bitwise in every chain; test 3 hard +0.331 [+0.214,+0.448], hardcb +0.514 [+0.336,+0.697] reproduce the 19 Nov receipt, n9 swap -0.151 reproduces B5; the seg export posterior equals B5 N9 seed 0 exactly (max diff 0); segment contributions plus bias recover the logits (assert 1e-4).

### Test 3 (smoke 3140648; chain 3140668, score 3140863)
BA full0: stock 73.74, hardcb 74.25, hard 74.07, n9 73.59, n5s 74.08, n5d 73.96, n9cb 73.84, n5scb 73.96, n5dcb 74.03, n5dTcb 74.07.
| contrast | pp | 80 % |
|---|---:|---|
| n5dcb - hardcb (registered +0.05) | -0.218 | [-0.407, +0.004] |
| n5dcb - n5scb (shared subset) | +0.076 | [-0.154, +0.390] |
| n5dcb - n9cb | +0.191 | [+0.039, +0.389] |
| n5d - n5s (registered +0.15) | -0.113 | [-0.392, +0.127] |
| n5d - n9 | +0.374 | [+0.001, +0.719] |
| n5s - n9 | +0.487 | [+0.132, +0.828] |
| n5dTcb - n5dcb (entropy matched to n9) | +0.035 | [-0.174, +0.256] |
| n5scb - hardcb / n9cb - hardcb | -0.295 / -0.409 | [-0.616,-0.061] / [-0.622,-0.217] |
Diagnostic on the 698 stock terminal misses: member disagreement 0.42 (n5d) vs 0.15 (n5s, n9); any-member-right 0.54 vs 0.39/0.40; ensemble accuracy on misses 0.320/0.318/0.322 (n5d/n5s/n9), entropy 0.865/0.828/0.797. Subset diversity raises member disagreement but not ensemble accuracy on the incumbent's misses.
Reading: no arm beats hardcb; the raw-family replacements lose 0.2 to 0.4 pp to hardcb. n5d vs shared n5s is a point loss, so the diversity form adds nothing measurable; the N5-over-N9 advantage is reproduced (+0.49 shared, +0.37 diverse), and entropy matching does not remove the n5dcb advantage over n9cb (+0.226 for n5dT). By the rule the form is not refuted to the letter (n5dcb - hardcb upper +0.004, n5dcb - n5scb upper +0.39) and not qualified (need lower bounds > 0 and +0.3): undecided, leaning negative; no follow-up fits warranted.

### Test 4 (step 1 job 3140653; head 3140671; chain smoke 3140677, complete 3140862, score 3141266)
Step 1, interleaved halves: genuine median error 0.20 Hz at 40, 0.04 at 80 (reliable n = 7, 5); donor 1.84, 0.97; donor minus genuine +1.83 [+1.44,+2.23] and +1.14 [+0.85,+1.44]; first/second halves: +1.11 [+0.78,+1.44], +1.26 [+0.93,+1.59]. Between-person SD of the peak is 1.9 Hz. Prediction (genuine 0.6, donor minus genuine +0.5) was conservative: the peak is strongly person-specific and replicates label-free. Step 1 passes (rule: lower bound > 0 at both ages).
Step 2 (compact log-band-power logistic head, 5 bands 4-30 Hz, 41 channels, causal expanding standardisation, MASS 1/6 on stock): standalone accuracy about 0.49 and not better with peak-relative bands (gen 0.47). Full chain BA: stock 73.74, fixed 73.77, gen 73.66, don 73.85. gen - fixed -0.109 [-0.397,+0.223]; gen - don -0.185 [-0.302,-0.073]; fixed - stock +0.033 [-0.23,+0.26]. Rule: a nonpositive upper bound against donor refutes: **refuted** (genuine peak shift is not better than another person's shift). Registered forecast was +0.08 vs fixed: missed by -0.19. Not covered: other band layouts, a stronger head (this one is 49 % raw accuracy), share other than 1/6, other orders. Part of the null is the head, not the landmark.

### Test 2 (seg export 3140678; chain smoke 3140963, complete 3140970; score 3141270)
Stand-in member: one raw FBCNet SessionBN seed (N9 B5 recipe), not the package's native FBC (the chain consumes banks, no live models); segment contributions exported through the final Linear (4 segments). Densities on past stock-raw argmax pseudo-labels (frozen shadow), start at 8 per class, W 0.10, clip 5, applied to the hardcb raw posteriors before count/CSP (2365 of 2655 windows active, 3-5 % of argmax labels changed). Donor = next person's features and pseudo-labels at the same age.
BA full0: hardcb 74.25, whole_g 74.48, disp_g 74.59, whole_d 73.10, disp_d 73.22.
| contrast | pp | 80 % |
|---|---:|---|
| disp_g - hardcb (registered +0.12) | +0.334 | [-0.205, +0.773] |
| whole_g - hardcb | +0.227 | [-0.008, +0.430] |
| disp_g - whole_g | +0.107 | [-0.312, +0.460] |
| disp_g - disp_d | +1.371 | [+0.309, +2.328] |
Reading by the rule: upper bounds against both matched controls are positive, so not refuted; disp_g - whole_g is +0.107 < +0.3, so not qualified: **undecided**. The whole-window Gaussian arm itself gains +0.23 over hardcb (lower bound -0.01) and the donor arms lose 1.0 to 1.2 pp, so the correction is history-specific and active; the dispersion features add +0.11 over it, inside noise. Not covered: native member, within0, returning people, Scherer, timing (<2 ms/window not measured), one member seed, correction constants not tuned (fixed before scoring). The +0.23 whole arm is a post-hoc reading of a control, not a registered effect.
