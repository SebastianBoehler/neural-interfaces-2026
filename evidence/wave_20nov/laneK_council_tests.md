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
