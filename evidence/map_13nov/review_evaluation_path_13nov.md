## Review of the evaluation path (replay/, experiments/, studies/11–13nov)

**Verdict:** the core replay is sound for new people: I found no leakage, causality or row-alignment error in `replay/`, and `replay/check.py` prints OK. The defects are in what the studies do around it. Three of them move "seen" or J by 0.3 to 0.8 points and reverse verdicts in §58b, §119 and §130b.

Nothing in the repository was written. The other session changed `studies/13nov/` while I read (`stacklib.py` and `pkg.py` appeared); line numbers are as of my read. My check scripts are in `/private/tmp/claude-501/-Users-sebastianboehler-Documents-GitHub-neural-interfaces-2026/ca8951cd-9c0d-4449-8fa2-f0de43144db9/scratchpad/` (file names given per finding).

### Defects, ranked by impact on decisions

**1. "Seen" and J from `A.system` are routing v3 with a gate refitted on the arm's own outputs; they do not hold on v4.1. CONFIRMED.**
- **Where:** `replay/alib.py:88-96` (`route(..., sh["all_loso"], ...)`), `replay/env.py:3-6`. `A.system` is routing v3, and it re-derives the gate's debias from the arm's own all_loso output.
- **Claimed otherwise:** §73 (`studies/12nov/netview2.py:27`) and §75 (`studies/12nov/lams.py:12`) say "full routed v4.1 replay" and call `A.system`.
- **New people are unaffected:** no all_loso session is routed in v3 or v4.1, so "new" is identical in both.
- **Checks (six-base means):**

| Lever | Seen as recorded (v3) | Seen on v4.1 (`route4` max, t10) |
|---|---|---|
| OCU on the shipped chain (§58b) | +0.51 (P04 +0.97); dJ +0.28, reproduced exactly | −0.33 (P04 −0.49, P10 −0.18); dJ −0.14 (`t9.py`) |
| LAMS λ 0.20 / 0.25 (§75) | −0.12 / −0.91, reproduced | +0.01 / −0.73 (`t8.py`); the verdict on new is unchanged |
| STACK FT alone (§130) | +0.10 | 0.00 exactly on every base once the gate is fixed, so the seen change is pure gate refit (`t10.py`) |
| STACK OCU alone | +0.47 | −0.03, gate fixed |
| STACK FT+OCU (§130b) | +0.18 | −0.03, gate fixed |
| STACK FT+REST+OCU (§130b) | −0.49 | +0.31, gate fixed |

- **Consequence:** the §130/§130b seen-bound verdicts reverse. "REST costs seen" and "FT+OCU is the only subset within the bound" are v3 artefacts, and `pkg.py` has no REST arm.
- **Also affected (seen or J only):** §54, §61, §87 (`focal_swap.py`), §99, §132, §133, and the seen columns of `ttt_reduce_score.py` and `ttt3_score.py`.
- **Noise:** per-base seen deltas of one arm range from −0.68 to +0.39, so a ±0.15 bound cannot be resolved.

**2. Personal nets are given to whole returning sessions from window 0, in both tails. CONFIRMED.**
- **Where:** `studies/12nov/pnet.py:10-35`, `studies/13nov/pkg.py:42-59`, and the PFT arms of `add.py` and `dose.py`.
- **What the code does:** the pfh60 outputs replace the SessionBN slots for every row of a seen session, including the shadow tail that feeds the gate.
- **What a package can do:** swap them in only after the gate routes (§102). That is window 10 for three sessions and window 60-63 for P04 session 2.
- **Check (`t11.py`, `t11b.py`):** as coded I reproduce §119 exactly, +0.55 (P04 +0.81, P10 +0.28). With personal nets only from the first routed window, in the routed tail, and the gate from the population shadow, it is −0.05 (P04 −0.10, P10 0.00).
- **Detail (B2_pool):** P10's +0.28 is one window, inside windows 0-9, before any gate can act.
- **Affects:** §78, §119, §138 arms P1-P4, and the PFT rows of §40/§43.

**3. "Six bases" are near-copies whenever the SessionBN families are swapped. CONFIRMED.**
- **Where:** `stacklib.py:70-74`, `ttt_reduce_score.py:32-39`, `ttt3_score.py`, `focal_swap.py`, `pnet.py`, `pkg.py`, `weak.py`, `teach2.py`.
- **Why:** slots 5-8 are replaced by one stored set, so the bases differ only in slot 9, which has weight 1 of 15.
- **Check (`t2.py`):** final labels agree 99.0-99.8 % between bases, against 97.3-99.0 % for the shipped chain. Per-person FT deltas correlate 0.92-0.99 across bases.
- **Still noisy:** FT ranges from +0.16 to +0.72 across these near-copies, so that spread is perturbation sensitivity, not seed replication.
- **Consequence:** "6/6 bases" in §106, §108, §130, §130b and §138 is not replication. The queued fresh-seed runs are the right test.

**4. The six bases in general are about two independent ones. CONFIRMED.**
- **Structure:** B2_pool is exactly the mean of B2_s012 and B2_s345 in slots 5-9, and B2_s012 is exactly the mean of s0, s1, s2. Slots 0-4 and 10, two thirds of the weight, are identical in all six.
- **Not fresh:** B2_pool contains the "fresh" seeds at half weight, so only B2_s345 is independent of the fresh set.
- **Null check (`t2.py`):** 16 random-view second-TSEM arms give a single-base SD of 0.16-0.28 on new and a six-base-mean SD of 0.16 (0.09 if independent). The correlation between exploration mean and fresh mean is 0.76. The best of 16 nulls reached +0.61 on the exploration bases.
- **Bootstrap:** the person bootstrap is on the right unit but ignores this noise.
- **Resolution:** one window is 0.07 J on the seen side and about 0.03 J on the new side.

**5. `seqstage.py` (§70) pools fold-specific PCA coordinates across people. CONFIRMED defect, effect on §70 not measured.**
- **Where:** `studies/12nov/seqstage.py:14, 22-25, 46-51`. It trains one projection on `pc128|all_loso` of nine people and applies it to the tenth.
- **Problem:** each person's rows are in their own fold's basis. Between folds 52-54 of 128 components flip sign, and beyond the first 8-16 the same-numbered component has median |corr| of 0.06-0.42 (`t5.py`).
- **Consequence:** the −11 to −16 points and the lesson "a similarity learned on nine people does not carry to a tenth", which §98 cites, are confounded.
- **Same fault:** `studies/11nov/clus/ceiling.py:29-34`, the "lda2" row of §30.
- **Units:** the script also printed values ×100; the ledger divided correctly.

**6. The MIX lever's seen column is computed with an inconsistent slot substitution. CONFIRMED.**
- **Where:** `stacklib.py:47-48, 82` with `replay/rlib3.py:61-66`. The personal head is substituted into the already powered mix.
- **Check (`t6.py`):** seen as coded is +0.13 / +0.53 / +0.81 on three bases; with the substitution done before the power it is −0.01 / +0.26 / −0.38. New is identical.
- **Affects:** the seen numbers of every §130 arm containing MIX.

**7. The KAPPA template prior is not a shrinkage. LIKELY.**
- **Where:** `studies/11nov/rt4/route4.py:54` adds `kp * PM` to the numerator but not `kp` to the denominator; `replay/pl.py:109` does add it.
- **Effect:** the template mean is over-scaled by (n + m0 + κ)/(n + m0).
- **Affects:** §19 (+0.19) and §57b.

**8. The grid chain has no gated early boost. CONFIRMED; disclosed in the docstrings.**
- **Size (`t12.py`):** it sits 0.6-1.0 points below the shipped chain on new.
- **Affects:** §69, §71, §90, §92, §103b, §109, §117, §118 measure against that weaker baseline. Early-window levers such as POPPRIOR overlap with the missing stage.

**9. Small ones.**
- **`segfp3.py:12-36`:** after re-ordering, the gate's `oof` is indexed by global row but passed in permuted order. CONFIRMED; the effect is ≤ 0.07 J (`t7.py`).
- **`netaug_score.py:12-21`:** arms with fewer seeds are compared with the 3-seed twin (sinc cmix4 with 2, sinc swa50 with 1). CONFIRMED in the log.
- **`replay/score.py:32-35`:** JG1 is new only and JG2 is half new plus half seen, so the two columns are on different scales. CONFIRMED by reading.
- **`weak.py:11-15`:** the gate threshold is an in-sample quantile of the dev sessions, and the gate uses the B2_pool mix for all bases. Minor.

### Checked and fine
- **Folds:** no all_loso fold trains on its test person. Seen folds train on session 1 only of the test person, plus all sessions of the other returning person.
- **Alignment:** `idx`, `sess`, `y`, `cell` and `person` are identical in `inputs.npz` and `bases.npz`. Sessions are contiguous and in acquisition order. Every stored-output loader asserts its index, except the ttt2 all_loso loads in `add.py`, `dose.py`, `pft_score.py` and `pnet.py`.
- **Metric:** every cell holds all three classes, and the person-equal weighting is as documented.
- **Causality:** rebalancing, gate, SessionEM, TSEM, both routing gates, the own-space members and the ttt fine-tuning all use only windows up to t.
- **Pseudo-labels:** `chain_final.npz` equals the shipped B2_pool v3 output exactly; no oracle labels. It is one base's chain used for all bases, by design.
- **Module state:** the shipped chain is bit-identical before and after EXTRA, `A.mix`, GE and LAM patches. `_HEADS`, `_H` and `_BIO` omit settings from their keys, but these are constant per process in every current caller.
- **Others:** core1 is identical across bases; the hindsight per-session picks in §76, §110 and §117 are declared as bounds; own2 calibration, PFT rows and ocunet are fold-clean.

### Check scripts (same folder as this file)
- `t1.py`: identifies `chain_final.npz` (equals the shipped B2_pool v3 routed output).
- `t2.py`: near-copy bases in the swapped-family configuration; 16 null arms across the six bases; state carry-over check.
- `t4.py`: fold composition, row alignment, session order, member identity across bases and protocols.
- `t5.py`: PCA basis consistency across folds.
- `t6.py`: MIX lever with the slot substitution as coded and consistent; rows of the in-population proxy.
- `t7.py`: `segfp3.py` with the gate's `oof` as coded and re-aligned.
- `t8.py`: LAMS deltas under v3 and v4.1 routing.
- `t9.py`: OCU view on the shipped chain under v3 and v4.1 routing.
- `t10.py` (output `t10.out`): stack arms under v3 as coded, v3 with the gate fixed, v4.1 with the gate fixed.
- `t11.py`, `t11b.py`: personal nets on whole sessions against from the first routed window; per-window location of the gain.
- `t12.py`: grid chain against the gated-boost chain and the routed chain on new people.
- `insp1.py`, `insp2.py`: array inventory of the replay data; base structure and index checks.
