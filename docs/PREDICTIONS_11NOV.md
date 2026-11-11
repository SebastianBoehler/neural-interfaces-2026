# Predictions ledger · written 11 Nov 2026, 01:50 UTC, before any of these results

Each row states the mechanism we believe in, a numeric prediction with an ~80 % range, and what would refute it.
After a run finishes, fill in **Observed** and **Verdict** (hit = inside range; miss = outside; mechanism refuted /
supported). Units: dev J and its pieces in balanced-accuracy points (person-equal, 0.5·new + 0.5·seen) unless stated.
Sealed points are 100 × the Codabench cell mean.

Reference values after the 10 Nov rescoring:
- cand_nxif 77.23, TSEM package 80.80, route3geb 81.35 sealed.
- The TSEM step is +3.57 sealed and the routing+GEB step is +0.55.
- Before the fix these were +4.02 and +0.43.

## 1. Fast solver package (Galvani 3085817–3085820, then ABBA timing)

| Check | Mechanism | Prediction (80 % range) | Refuted if | Observed | Verdict |
|---|---|---|---|---|---|
| Parity on 2,655 windows, worker image and venv | Shared FIR bank is the same float32 maths; the overlap only changes scheduling, not arithmetic | Labels identical (P ≈ 0.95); posteriors bitwise or ≤ 1e-6 | Any label flip, or posterior diff > 1e-5 | Bitwise at every stage on all 2,655 windows, in both the worker image and the venv. On CUDA the first (cold) stream differs from warm streams by ≤ 6.1e-8, identically for both codes; labels are unchanged | **Hit** |
| Weights-only variant (lean vs fast) | Post-mix stages are unchanged lean code | Bitwise identical (P ≈ 0.95) | Any difference in stage outputs | Bitwise on all 2,655 windows (job 3085820) | **Hit** |
| A100 solver time vs shipped | Codex measured 58 / 156 s on the shipped-equivalent | Ratio 0.37 (0.32–0.50) | Ratio > 0.6 | Solver 216 / 521 s = 0.41 on the same node; whole run including overhead 0.53 (ABBA 3086004) | **Hit** |
| Estimated sealed duration | Fixed overhead ≤ 60 s + ratio × the rest | 17–22 min against the 2,687 s run (15–28); 22–30 min against the 3,732 s rerun | > 35 min | About 19 / 26 min at the 0.41 ratio; 25 / 35 min at the overhead-inclusive ratio (not yet measured on the sealed worker) | Hit (estimate) |
| Peak memory | Same members, one extra worker thread | Within +10 % of shipped | > +30 % | Host RSS 2.56 vs 2.60 GB; CUDA 55 vs 61 MB | **Hit** |

## 2. Order fingerprint, stage 1 (no tuning; replay of our three packages)

**Hypothesis H1:** since the fix, windows within a session arrive in a fixed mixed order across runs. Every PROTEUS run
holds exactly 10 or 12 windows per class, and our W = 30 rebalancing (≈ one run) exploits that local balance. Mixing
runs removes it. TSEM's confident seeds get worse early, and TSEM amplifies seed quality. Routing gains slightly,
because early windows then represent the whole session.

**Alternatives:**
- **H0:** the order is unchanged and the drop has another cause (data, scoring).
- **H2:** windows are shuffled within each run only.
- **H3:** runs are permuted, chronological inside.

Predicted Δ versus chronological (H0) on dev J for each package:

| Order | cand_nxif | TSEM package | route3geb | TSEM step change | Routing step change |
|---|---|---|---|---|---|
| H1 within-session shuffle | −0.6 (−0.2 to −1.2) | −1.0 (−0.5 to −1.8) | −0.9 (−0.4 to −1.6) | −0.4 (−0.1 to −0.8) | +0.1 (−0.2 to +0.4) |
| H2 within-run shuffle | −0.1 (−0.4 to +0.1) | −0.2 (−0.6 to +0.1) | −0.2 (−0.6 to +0.1) | ≈ 0 | ≈ 0 |
| H3 runs permuted | −0.1 (−0.4 to +0.2) | −0.2 (−0.5 to +0.2) | −0.2 (−0.5 to +0.2) | ≈ 0 | ≈ 0 |

**Sealed pattern to match:** −0.72 / −1.17 / −1.05, with the TSEM step −0.45 and the routing step +0.12.
- **Supports H1:** the H1 row matches that pattern in sign and ordering, and H2/H3 are much smaller.
- **Refutes H1:** H1 changes are small (> −0.2 for all three), or H2/H3 reproduce the sealed pattern better.
- **Cores recomputed under H1:** the causal recentring converges regardless of order. Predicted extra effect ±0.2.

Prior P(H1) = 0.6–0.7 (from the board pattern; see the research report in docs/SEALED_UPLOADS.md).

| Item | Observed | Verdict |
|---|---|---|
| H1 row | −1.96 / −1.95 / −3.51; TSEM step +0.02; routing step −1.56 (RMS vs sealed 1.66) | **Miss.** The drops are 2–3× the predicted range and the step signs are opposite. H1 is refuted as the explanation of the sealed drop. Under H1, GEB + routing collapses on dev (all 10 new people lose), whereas on sealed that step grew. |
| H2 row | −0.42 / −0.03 / −0.92; TSEM step +0.40; routing step −0.90 (RMS 0.69, closest) | **Miss.** The routing-step sign is opposite to sealed. Rejected. |
| H3 row | −1.37 / −2.19 / −3.83; TSEM step −0.82; routing step −1.64 (RMS 1.75) | **Miss.** Far larger than predicted. Rejected. |
| Lesson | Our chain is much more order-sensitive than predicted. Any reshuffle of runs costs 2–3× the observed sealed drops, so the sealed order is very likely still (close to) chronological. The fix must have changed something else (data, scoring or a worker issue). Stage 2 does not trigger. | |
| Core recompute under H1 | | |

## 3. Order-robust re-tune, stage 2 (only if H1 is accepted)

**Mechanism:** without run-level balance, a longer or cumulative rebalancing window with a stronger α captures the
session-level balance (draws without replacement across the session). TSEM and GEB schedules tuned on chronological
early windows may need an earlier start.

**Predictions:**
- **Selected setting:** cumulative or W = 60 rebalancing with α 1.3–1.5. TSEM n0 30. GEB probably kept.
- **ΔJ under H1:** +0.5 (+0.2 to +0.9).
- **ΔJ under H0 (protection clause):** −0.3 (0 to −0.6). This is a real trade-off: chronological order loses the
  run-balance benefit.
- **Sealed, if H1 is true:** +0.4 (+0.1 to +0.7), i.e. 0.8135 → about 0.817 (0.814–0.820).
- **Sealed, if H1 is false:** −0.2 (0 to −0.5).
- **Refuted if** no setting gains ≥ +0.2 under H1 (then the order change is not recoverable by stage settings).

| Item | Observed | Verdict |
|---|---|---|
| Selected setting | | |
| ΔJ H1 / H0 | | |
| Sealed (if uploaded) | | |

## 4. ST-EEGFormer member (run 11 Nov: smoke 3085913, array 3085914)

**Mechanism:** a large pretrained transformer might add a new kind of error diversity, as SessionBN did. Our three
earlier foundation-model members (CBraMod, LaBraM, REVE) were neutral or worse.

**Predictions** (written before any result; the full pre-registration is in the scratch `steeg/prereg.txt`):
- **Standalone, new people:** 60 (55–66), below core1's 69.3.
- **In the chain:** +0.0 (−0.3 to +0.4).
- **Sealed:** +0.0 (−0.2 to +0.3).

**Run:** ST-EEGFormer-small (25.3 M parameters, MIT release checkpoint-300), fine-tuned with the authors' recipe for 50
epochs. The 24 folds are all_loso plus seen_session, with seeds 0 and 1. The member is the mean of the two seeds. Each
fold trained in about 25 min on an RTX 2080 Ti. Inference takes 3.8 ms per window at batch 1, with 3.45 GB peak memory.

| Measure | Predicted | Observed | Verdict |
|---|---|---|---|
| Standalone new (raw argmax, two seeds) | 60 (55–66) | 53.85 (one seed: 53.38 and 52.64; after rebalancing 56.35) | Miss, below the range |
| Standalone seen / J | — | 53.73 / 53.79 | — |
| Chain dJ, nested over people halves | +0.0 (−0.3 to +0.4) | +0.16 (held-out G2 +0.04, G1 +0.27; both halves picked w 0.5) | Hit |
| Chain dJ by weight (mean of B2_pool and B2_s345) | — | w 0.5: +0.08 (new +0.21, seen −0.06); w 1: −0.27 (seen −0.54); w 2: −0.63 | — |
| Sealed | +0.0 (−0.2 to +0.3) | not tested (gate failed, no package) | — |

**Gate:** failed. Standalone new is 53.85 against a bar of ≥ 66, and nested dJ is +0.16 against ≥ +0.5. Both halves
are ≥ 0 and seen at the pick is −0.06 (≥ −0.3), so those two checks passed. The fresh bases (B2_s012, s0, s1, s2) were
not scored.

**Reading:** the pretrained encoder is 15 points below core1 on new people, the same gap as CBraMod, LaBraM and REVE.
P04 (38 and 41) and P05 (46 and 42) are near chance. Its errors are partly new: at w 0.5 it nudges new people by +0.21.
Any weight that lets it vote on seen people hurts them. About 2,400 training windows from 9 people, each 4 s at
128 Hz, are too few to adapt a 25 M-parameter transformer past what the tangent and filter-bank cores already extract.
This closes the foundation-model member line (the fourth model to fail).

## 5. Overnight research runs (11–12 Nov)

Written 11 Nov, 03:17–03:20 CET, before any arm was scored (arm preregs stamped 03:17:22 CET). Scratch: `night/` (mirrored in
`runs/archive_scratch_11nov/night/`). The common protocol and gate are in `night/PROTOCOL.txt`; each arm has its own
`night/<arm>/prereg.txt`. Running summary: `night/SUMMARY.md`.

**Base.** The full route3geb chain replay (lean `alib.system`), J 79.18–80.21 on the 6 stored bases.

**Procedure.**
- Every config runs on the exploration bases (B2_pool, B2_s345).
- The pick (best mean exploration dJ) runs on the 4 fresh bases.
- Its matched control runs on all 6 bases.

**Gate** (fresh means):
- dJ ≥ +0.40, new ≥ +0.20, seen ≥ −0.20, and at least 6 of 10 people ≥ 0;
- pick − control ≥ +0.30, where the arm has a control.

**Verdicts.** SCREEN+ means fresh ≥ +0.20 without passing the gate: it earns a follow-up only. Oracle arms are
ceilings and have no gate.

**Limits.**
- The fresh bases differ only in member seeds, not in people.
- With about 20 arms, 2–3 SCREEN+ are expected by chance.

**Already known when this was written** (D0, descriptive):
- **Error shares:** MI↔WORD 37 %, SUB↔WORD 36 %, MI↔SUB 27 %. WORD has the lowest recall (73.7 %) and the lowest seed
  precision (78 %, against 84 % for MI and SUB).
- **Run-count prior:** with the chain's own counts after the chain, it loses 0.6–8 J.

**Honest prior.**
- No arm has a plausible +0.5 J.
- P(at least one PASS) ≈ 0.10.
- The informative outputs are the oracle ceilings (A01, A09) and the pair-specific error counts.

| Arm | Mechanism | Prediction (80 % range) | Refuted if | Observed | Verdict |
|---|---|---|---|---|---|
| A01 run-count oracle (new; ceiling) | Without-replacement class prior inside true runs (online runs = 10/class), smoothed, before SessionEM / TSEM | Best of 5 configs, all 6 bases: −0.1 (−0.6 to +0.3). Shift-15 control: −0.2 (−0.8 to +0.2) | Best < +0.2, or best − control < +0.2 | Best b0.25_keep −0.76 (all 5 configs −0.76 to −4.24); shift-15 control −0.64 | **Miss** (below the range). **Mechanism refuted:** the run prior with the chain's own soft counts compounds count errors. |
| A02 run-count, protocol runs (new) | Same prior, label-free: acquisition block of o windows, then runs of 30 (mixture over o = 66–72) | Fresh −0.15 (−0.6 to +0.2); P(PASS) 0.02 | Fresh < +0.2, or pick − period-25 control < +0.2 | Fresh −0.75 (new −0.59, seen −0.91); period-25 control −0.60 | **Miss** (below the range). **Refuted.** |
| A03 pair-specific SUB/WORD discriminant (new) | TSEM's SUB:WORD ratio from a pair LDA with SUB/WORD-only covariance | Fresh −0.05 (−0.4 to +0.3); SW errors −2 (−8 to +4) | Pick − shared-covariance control < +0.1, or SW errors do not fall | Pick ps0.9: fresh −0.07. Control = shipped TSEM (shared covariance with exponent 1 is identical), 0.00. SW errors +5.5 | **Hit on dJ, miss on SW** (SW errors rose). **Refuted.** |
| A06 Dawid–Skene before TSEM (new) | Per-session class-conditional confusion matrices of the 8 member families, label-free EM over the prefix | Fresh −0.1 (−0.5 to +0.25). Row-permuted null −0.3 (−0.8 to 0) | Fresh < +0.2, or pick − null < +0.3 | Pick λ.1 κ40: fresh −0.48. Null −1.53. λ ≥ 0.2: −2.6 to −15.9 on exploration | **Hit** on fresh (low end); **miss** on the null (more harmful). **Refuted:** session-specific confusion matrices reinforce the chain's own errors. |
| A07 Dawid–Skene after TSEM (new) | As A06, after TSEM | Fresh −0.05 (−0.3 to +0.2) | Same as A06 | Fresh −0.63; null −1.51 | **Miss** (below the range). **Refuted.** |
| A08 Dawid–Skene, 4 groups (new) | As A06, with 4 less-correlated annotator groups | Fresh −0.05 (−0.4 to +0.25) | Same as A06 | Fresh −0.11; null −1.07 | **Hit.** **Refuted.** |
| A09 view oracles (new; ceiling) | Second template stage seeded by true past labels on window-local raw views. cm is the common mode of the M1-referenced channels (mastoid EMG, glossokinetic potentials); CAR removes it, so TSEM never sees it | All 6 bases: cm λ .15 +0.3 (0 to +0.8); cm λ .3 +0.4 (−0.1 to +1.0); cmhi +0.15 (−0.1 to +0.5); phys +0.4 (0 to +1.0) | Label-free A10/A11/A16 are moot where the oracle < +0.3 | cm λ.15 +0.11; cm λ.3 −0.04; cmhi −0.20; phys −0.03 | **Hit, hit, miss, miss.** Every ceiling is < +0.3, so the label-free view arms are moot: there is no extra session-specific class information in cm or phys, even with labels. |
| A10 common-mode view stage (new) | cm view stage after TSEM, seeded by the TSEM output | Fresh 0.0 (−0.3 to +0.25); WORD errors −1 (−6 to +4); P(PASS) 0.03 | Fresh < +0.2, or pick − row-permuted null < +0.3 | Pick λ.1: exploration +0.15, fresh −0.09; null −0.06. WORD errors MW −5.5, SW −0.2 | **Hit.** **Refuted.** |
| A11 cm slow vs EMG bands (new) | Ablation of A10: 0.5–4 Hz against 20–58 Hz | Fresh 0.0 (−0.3 to +0.2) | Same as A10 | Pick cmlo: exploration +0.17, fresh +0.07; null +0.05 | **Hit.** **Refuted** (pick − null +0.02). Follow-up A28. |
| A16 task-physiology view (new) | 8 CAR features: frontal-midline theta, parietal alpha, beta asymmetries, mu asymmetry, face-vs-hand motor beta, cm EMG | Fresh +0.05 (−0.25 to +0.3) | Same as A10 | Pick λ.1: exploration +0.08, fresh −0.20; null +0.01 | **Hit.** **Refuted.** |
| A21 cm inside TSEM (new) | PC128 + cm (×1 or ×4) as TSEM's joint feature space | Fresh −0.05 (−0.4 to +0.2) | Fresh < +0.2, or pick − permuted-cm control < +0.3 | Pick w1: fresh −0.01; permuted-cm control −0.20 | **Hit.** **Refuted** (pick − control +0.19). |
| A13 recursive seeds (refine) | Seeds from TSEM's own past outputs | Fresh −0.1 (−0.6 to +0.3) | Fresh < +0.2 | Pick rec ρ.7: exploration −0.04, fresh −0.46 | **Hit.** **Refuted.** |
| A12 Sinkhorn seeds (refine) | Seeds from a balanced prefix assignment | Fresh 0.0 (−0.3 to +0.3) | Fresh < +0.2 | Pick τ1: fresh −0.08 | **Hit.** **Refuted.** |
| A20 class-specific seed quota (refine) | Fewer, more confident WORD seeds (ρ_WORD 0.5–0.6) | Fresh +0.05 (−0.25 to +0.3) | Fresh < +0.2 | Pick (.7, .7, .6): fresh −0.14 | **Hit.** **Refuted.** |
| A04 SUB:WORD exponent (refine) | Rescale the TSEM SUB:WORD exponent ×0.5, ×1.5 or ×2 | Fresh −0.05 (−0.3 to +0.2) | Fresh < +0.2 | Pick ×1.5: fresh −0.17 (SW errors +6.2) | **Hit.** **Refuted.** |
| A05 MI-pair covariance (refine) | Pair-specific covariance for MI/SUB or MI/WORD | Fresh −0.05 (−0.4 to +0.25) | Fresh < +0.2 | Pick MI/SUB ps0.9: exploration +0.14, fresh +0.07 (new +0.24, 7/10, seen −0.09) | **Hit.** Below +0.2, so refuted by its criterion. The only arm with the same new-person sign on both base sets. Follow-up A27. |
| A17 PC16 template stage (refine) | Low-dimensional extra stage (template variance), growing or early-only exponent | Fresh +0.05 (−0.2 to +0.3) | Fresh < +0.2 | Pick λ.1 early: fresh −0.08 | **Hit.** **Refuted.** |
| A18 shrinkage schedule (refine) | TSEM shrinkage falling with t | Fresh 0.0 (−0.3 to +0.25) | Fresh < +0.2 | Pick 0.8→0.6: fresh −0.15 | **Hit.** **Refuted.** |
| A19 margin-ranked seeds (refine) | top1 − top2 margin instead of top1 | Fresh 0.0 (−0.25 to +0.2) | Fresh < +0.2 | Pick ρ.7: fresh −0.20 | **Hit** (at the edge). **Refuted.** |
| A24 rebalancing window (refine; checks §3) | W 60 or cumulative, α 1.15 or 1.3, chronological order | Fresh −0.15 (−0.5 to +0.1). §3 said −0.3 (0 to −0.6) | Run-balance account weakened if any config is ≥ +0.2 | All 4 configs negative on exploration (−0.42 to −0.49). Pick cumulative α1.15: fresh −0.44 (new −1.16, seen +0.27) | **Hit; §3 hit.** **Supported:** the W30 run-scale balance is what the chain exploits. Longer windows cost new people about 1 J. |
| A15 combination (refine) | Picks of up to 4 arms with exploration ≥ +0.15, composed | P(run) 0.35. If run: best single + 0.0 (−0.3 to +0.2) | Fresh < best single + 0.2 | Not run: only A11 reached exploration ≥ +0.15 | Consistent with P(run) 0.35. |
| A25 mirrored-MI core1 (new; pre-registered 03:20) | Non-dominant-hand MI: the hemisphere depends on handedness, which the solver does not know. core1 is refitted with left-right mirrored training MI windows, an exact channel permutation of the tangent and slow blocks. D0: P03 is left-lateralised | Fresh +0.05 (−0.3 to +0.3); P03 +1.0 (−1 to +3); P03 core1 MI recall +5 (0 to +15); mirrored-SUB control −0.1 (−0.4 to +0.2); P(PASS) 0.04 | P03 Δ ≤ 0, or aug − control < +0.2 | Fresh −0.25 (new −0.32, seen −0.17, 4/10); all 6 −0.34; P03 −0.28. core1 MI recall: P03 57.1 → 58.7, but P04 65 → 30, P05 64 → 41, P09 84 → 62, P10 55 → 41. Mirrored-SUB control fresh −0.03. Permutation checked against mirrored raw windows (relative error 5e-8) | **Hit** on every number, but **refuted** (P03 Δ < 0; aug − control −0.22). Lateralised MI evidence is useful for the right-lateralised majority, and mirror invariance destroys it. |
| A26 oracle seeds per class (diagnostic; pre-registered 03:23) | TSEM seeds of class k drawn from true past labels, for each class and for all three. This bounds the seed-precision arms | All 6 bases: all three +0.85 (0.5 to 1.3); WORD +0.4 (0.1 to 0.8); SUB +0.2 (−0.1 to 0.5); MI +0.2 (−0.1 to 0.5) | If all three < +1.0, seed precision is worth < 0.3–0.4 J label-free | All three +0.93 (new +0.66, seen +1.19); WORD +0.42; SUB +0.18; MI +0.19 | **Hit** ×4. Perfect seed labels buy < 1 J, and WORD seeds are the largest slice. The seed-precision lever is capped at about 0.3 J label-free. |
| D3 family accuracy on the chain's errors (diagnostic) | How often each member family's argmax is the true class when the chain is wrong | No family > 40 % for any class; at least one family right in 70 % (55–85) | — | Share of chain errors where the family is right:<br>• MI: teacher 42 %, others 15–27 %; any family 81 %<br>• SUB: 13–29 %; any 76 %<br>• WORD: Shallow 42 %, SBN FBCNet 41 %, SBN FBLight 41 %, SBN Sinc 36 %, core1 18 %; any 92 % | **Miss** (two families exceed 40 %; WORD any = 92 % is above the range). The cores are the weakest on WORD errors. Follow-ups A29 / A30. |
| D5 per-session anatomy (diagnostic) | Errors by session, and the margin distribution | 3 weakest new people hold ≥ 45 % (35–60) of new-person errors; ≥ 30 % (20–45) of errors and ≥ 15 % (8–25) of correct windows have margin < 0.1 | — | Error counts: P04 (3 new sessions) 263 of 622 new-person errors; P04 + P10 + P03 67 %. P04 as a new person: SUB recall 46–54 %, WORD 40–56 %. Margin < 0.1: 38.7 % of errors and 13.0 % of correct windows (seen 33 % / 10 %) | **Miss** on concentration (higher). **Hit** on margins. Near-tie correct windows (≈ 264) outnumber near-tie errors (≈ 241), so any small uninformative reweighting is zero-sum, which explains the ≈ 0 arms. |
| A27 MI/SUB pair covariance, neighbourhood (deeper; pre-registered after wave 1) | A05 pick region: shrink 0.85, 0.9 or 0.95; exponent 1 or 1.25 | Fresh +0.0 (−0.25 to +0.25); new +0.1 (−0.2 to +0.4); P(PASS) 0.02 | Winner's curse if fresh < +0.1 or new < +0.1 | Pick ps0.9, exponent 1.25: exploration +0.15 (new +0.48), fresh −0.15 (new +0.15, 6/10, seen −0.45) | **Hit** on dJ and new. **Refuted:** winner's curse, fresh < +0.1. The new-person gain (+0.15 to +0.24 across A05 / A27) costs seen people as much. |
| A28 cm slow band, deeper (new; pre-registered after wave 1) | Glossokinetic / slow mastoid view stage, λ 0.1, 0.15 or 0.2, with a row-permuted control | Fresh +0.03 (−0.2 to +0.25); pick − control +0.0 (−0.2 to +0.2); P(PASS) 0.01 | No usable information if pick − control < +0.1 | Pick λ.1: fresh +0.06; row-permuted control +0.05 | **Hit.** **Refuted:** pick − control +0.00, so the slow mastoid view carries no usable information. |
| D7 null calibration (diagnostic) | 8 pure-noise arms (row-permuted cm view stage) under the same pick-then-fresh protocol | Picked exploration +0.05 (−0.1 to +0.2); fresh −0.1 (−0.3 to +0.05); P(fresh ≥ +0.2) ≤ 0.1 | — | 8 noise arms. Picked exploration −0.04 [−0.16, +0.18]; fresh −0.03 [−0.22, +0.10]; P(fresh ≥ +0.2) 0/8 | **Hit** ×3. Every wave-1 / wave-2 fresh result (best +0.07) lies inside the null range [−0.22, +0.10]. |
| A29 WORD column × deep families (refine; pre-registered 03:53, from D3) | The WORD column of the member mix weights TF, Shallow and the SBN nets by γ 1.25, 1.5 or 2 | Fresh +0.0 (−0.3 to +0.3); WORD errors −4 (−15 to +6); P(PASS) 0.04 | Fresh < +0.2, or pick − core-γ control < +0.3 | Pick γ1.5: exploration +0.07, fresh −0.06 (new −0.22, seen +0.10); core-γ control −0.30 | **Hit.** **Refuted** (pick − control +0.24 < 0.3; fresh < +0.2). |
| A30 MI column × teacher (refine; from D3) | The MI column weights the teacher by γ 1.25, 1.5 or 2 | Fresh −0.05 (−0.35 to +0.25); P(PASS) 0.03 | Fresh < +0.2, or pick − SUB-column control < +0.3 | Pick γ1.5: fresh −0.11; SUB-column control −0.14 | **Hit.** **Refuted.** |
| D1 pair separability (diagnostic) | True-label causal LDA per session and view, for each class pair | PC128 SUB/WORD 75 % (65–85); cm ≥ 10 points below PC128; PC128 fixes about 40 % (25–55) of the chain's SUB/WORD errors | — | PC128 SUB/WORD 78.0 %, cm 59.2 % (18.8 below), phys 60.9 %, PC128+cm 77.8 %. MI/WORD 83.7 %, MI/SUB 87.8 %. PC128 fixes 61 % of the chain's SUB/WORD errors; cm 53 % (about chance) | **Hit, hit, miss.** The fix rate is higher than predicted, but every view is near chance on the chain's own errors, and cm adds nothing in a joint LDA |

**Outcome of §5 (11 Nov, 04:02 CET).** 29 runs (25 arms and 4 diagnostics) were scored locally against their
pre-registrations, plus one combination that was not run.
- **Gate results:** 0 PASS and 0 SCREEN+. The best fresh dJ is +0.07, inside the null-calibration range
  (−0.22 to +0.10).
- **Calibration:** 43 of 54 numeric predictions hit their 80 % range (80 %).
- **Misses on mechanisms, all too optimistic (6):** A01, A02, A03's SUB/WORD error count, A07, and two A09 ceilings.
- **Misses on diagnostics and nulls (5):**
  - A06's null was more harmful than predicted;
  - D1's fix rate was higher;
  - D3 had two misses;
  - D5's error concentration was higher.
- **Ceilings:**
  - true-label seeds +0.93 J (WORD seeds +0.42);
  - true-label common-mode / physiology views ≤ +0.11;
  - run metadata with the chain's own counts ≤ −0.6.
- **Reading:**
  - The label-free session layer is at a local optimum on these people.
  - Small reweightings are zero-sum: near-tie correct windows (13 % of correct) outnumber near-tie errors.
  - The remaining J has to come from new base information, not from re-weighting the session stage.

## 6. Wave W12: pair specialists for the WORD confusions (written 11 Nov 10:25 UTC, before any W12 code or result)

**Why this wave:** last night's diagnostics leave pair confusions as the one target with headroom.
- The session stage is at a local optimum.
- More seeds do not help: single-seed bases average 79.7, three-seed bases 80.0, the six-seed pool 80.0.
- The 10 Nov rescoring moved the whole field (23 of 25 common board submissions dropped, median −0.6), so it gives no
  chain-specific lever.
- The errors are pair confusions: MI↔WORD 37 %, SUB↔WORD 36 %, MI↔SUB 27 %. The true class is the runner-up in 73 %.
- WORD has the weakest recall (73.7 %), and some family is right on 76–92 % of the chain's errors.

**Mechanism:**
- A 3-class softmax net shares one representation across all three pair boundaries.
- A net trained only on one boundary (SUB/WORD, MI/WORD, or WORD vs rest) may estimate that split more sharply, with
  different errors.
- SessionBN statistics still see every window of the session.
- The specialist re-splits the chain's pair mass before rebalancing: P_a' + P_b' = P_a + P_b, blended with weight β.
- A multitask twin (3-class plus pairwise heads) tests the same idea as an ordinary member.

**Evidence against (prior):**
- Codex's linear class-pair readouts lost to multinomial heads on the source screen.
- Night arms A03 (pair covariance) and A29 (WORD column × deep families) failed.

| Arm | What | Prediction (80 % range) | Refuted if |
|---|---|---|---|
| S-SW | SessionBN FBCNet + SessionBN ShallowConvNet, SUB vs WORD, seeds 0–2 | Pair BA on new people 66 (60–72); chain nested dJ +0.10 (−0.20 to +0.40) | Pair BA ≤ the 3-class net's own SUB/WORD split, or dJ ≤ 0 |
| S-MW | Same, MI vs WORD | Pair BA 70 (64–76); dJ +0.05 (−0.20 to +0.35) | Same |
| S-WR | Same, WORD vs rest | BA 65 (60–71); dJ +0.05 (−0.25 to +0.35) | Same |
| MT | SessionBN FBCNet with auxiliary pairwise heads as a member (extra, or replacing its plain twin) | Standalone new ≈ twin ± 1; dJ +0.05 (−0.25 to +0.30) | — |
| Wave | Best arm after nested selection | P(gate pass: nested dJ ≥ +0.5, halves ≥ 0, seen ≥ −0.3) ≈ 10 %; P(any arm ≥ +0.2 nested) ≈ 30 % | — |

**Sealed if a gate passes:** about 0.6–0.7 × the nested dJ (the small-gain transfer rate).

**Calibration note (11 Nov 10:50 UTC, before any W12 training result).** The scorer test exposed the comparators: the
existing 3-class SessionBN FBCNet (B2_pool member 6) already splits the pairs at SUB/WORD 74.2, MI/WORD 77.5 and
WORD-vs-rest 72.7 BA on new people. So the pair-BA predictions above (66 / 70 / 65) were set too low and will probably
miss low. The chain dJ predictions are unchanged. The refutation "pair BA ≤ comparator" now has a known bar.

| Item | Observed (21 tasks: 11 from array 3086618, 10 from the packed job 3087109) | Verdict |
|---|---|---|
| Pair BA on new people, specialist vs the 3-class FB member's own split | SUB/WORD 72.6 vs 74.2; MI/WORD 77.0 vs 77.5; WORD-vs-rest 69.7 vs 72.7 (the full mix: 79.2 / 81.5 / 76.7) | **Refuted** by the stated rule: every specialist is at or below the 3-class net. (The predicted 66 / 70 / 65 missed low, as the calibration note said.) |
| Chain dJ, pair re-splits (β 0.25 / 0.5 / 0.75) | SUB/WORD −0.74 / −2.90 / −3.97; MI/WORD −0.70 / −1.96 / −3.00; WORD-vs-rest −0.56 / −2.51 / −3.38 | **Miss, far below the range** (+0.05 to +0.10 predicted) |
| Chain dJ, multitask member | extra at weight 0.5 / 1 / 2: −0.09 / −0.00 / −0.67; replacing its twin −0.04 | Hit (predicted +0.05, −0.25 to +0.30) |
| Nested selection, gate | nested −0.20 (halves −0.03 / −0.37); seen +0.20 at the pick | **FAIL**; fresh bases not opened |

**Reading.** A net trained on one boundary is no better at that boundary than a 3-class net, and it is worse
calibrated against the rest of the chain: replacing even a quarter of the chain's pair split costs 0.6–0.7 J. The
three-class representation is not what limits the pair confusions. Pair specialists are closed.

## 7. Scherer 2015 member test (written 11 Nov ~10:30 UTC, before any code or result; research use only; time corrected at 10:50 UTC, an earlier draft said ~11:15)

**Licence status:** shipping needs your decision and, ideally, written clearance (the question to the organisers is
pending). This arm only measures the value.

**Mechanism:** Scherer 2015 (BNCI 004-2015; 9 people with stroke or spinal-cord injury; hand MI, subtraction and word
association; 2,130 windows) is the only public corpus with all three of our tasks.
- **Expected effect:** extra people should make the SUB and WORD class directions more person-general, and WORD is our
  weakest class.
- **Round 5 (19 Oct):** at loss weight 0.1 with the interpolation null space filled (`~n~w0.1`), it gave +0.7
  all_loso on a much weaker rebalanced ensemble (62.8 → 63.5, 7/10 people up) and +0.1 seen.
- **Limits:** the source files are high-passed at about 4 Hz, 22 of 41 channels are interpolated, and hand MI
  transfers worst (35 % recall).

**Arms:** retrain chain members with Scherer added to every training fold, against PROTEUS-only twins from the same
code and seeds (control). Member = mean of seeds 0–2. The chain swaps the member in, against the twin swapped in.
- **SessionBN aligned FBCNet** (FB slot): weight 0.1 and 0.25, all three Scherer classes.
- **Raw ShallowConvNet:** the same.
- **SUB + WORD only:** Scherer hand-MI windows get loss weight 0 but stay in the alignment and normalisation statistics.

| Arm | Prediction (80 % range) | Refuted if |
|---|---|---|
| Member standalone new (vs twin) | +0.8 (−0.5 to +2.0) | ≤ 0 for both architectures |
| Chain nested dJ (all classes, best weight) | +0.15 (−0.25 to +0.50) | ≤ 0 |
| Chain nested dJ (SUB + WORD only) | +0.20 (−0.20 to +0.60) | ≤ the all-classes arm |
| WORD recall in the chain | +1.0 (−1 to +3) | ≤ 0 |
| P(gate pass: nested ≥ +0.5, halves ≥ 0, seen ≥ −0.3) | ≈ 15 % | — |

| Item | Observed (23 of 24 tasks; the SUB+WORD-only FB arm has seeds 0 and 2) | Verdict |
|---|---|---|
| Member standalone vs its PROTEUS-only twin (new / seen) | SessionBN FBCNet k 2: +1.63 / +2.87 at weight 0.1, +1.03 / +2.94 at 0.25; k 5: +1.28 / +1.13, +0.84 / +1.64; raw ShallowConvNet: +1.89 / +0.84, +2.65 / +0.01. SUB+WORD-only: +0.1 to +1.3 new | **Hit** (predicted +0.8, range −0.5 to +2.0; upper half) |
| WORD recall of the members on new people | +2.4 to +4.4 points in every arm | Supported at member level |
| Twin check | the retrained FB twins equal the shipped seed files exactly; the raw Shallow twins agree on 91 % of labels (training nondeterminism) | Harness valid for F; R twins are a noisier control |
| Chain, condition swapped in against the twin swapped in (B2_pool / B2_s345) | F at 0.25: +0.19 / +0.28; F at 0.1: −0.03 / +0.16; R: −0.6 to −0.2; both: −0.6 to +0.2. WORD recall in the chain −0.04 at the overall pick | **Miss** on WORD (predicted +1.0) |
| Nested selection | all 9 arms: −0.23 (halves −0.27 / −0.20); best overall F at 0.25, +0.23 | **FAIL**; fresh bases not opened |

**Reading.** Scherer makes the individual nets clearly better (1–3 points, WORD recall +3), and the chain absorbs all
of it. This is the fourth time today that a stronger member or head did not move the chain (weight edits, SessionBN
families, the regularised personal head, now Scherer). Member strength is not what limits the chain.

## 8. GPU-batched eigendecomposition speed probe (written 11 Nov ~10:45 UTC, before any code; time corrected at 10:50 UTC, an earlier draft said ~11:40)

**Mechanism:**
- **Where the time goes:** in the A100 profile (Codex 3073240), the two tangent cores' eigendecompositions take 40 of
  124 ms per window. They are many small 41×41 symmetric matrices handled on the CPU.
- **Current state:** the fast (overlap) package hides the CPU path behind the GPU nets but stays CPU-bound, at about
  77 ms per window.
- **Change:** one batched float64 `torch.linalg.eigh` per window on the A100 removes most of that path.
- **Constraint:** the worker image has no compilers, so this must stay pure PyTorch.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Labels, all 2,655 public windows, worker-image replica on A100 | Identical (P ≈ 0.9) | More than 2 label flips |
| Max posterior difference vs the fast package (final stage) | ≤ 1e-9 (1e-12 to 1e-6) | > 1e-5 |
| A100 solver time vs the fast package (ABBA) | 0.70 (0.55–0.85) | > 0.95 |
| CPU fallback (no CUDA) | Bitwise identical to the fast package | Any difference |

| Item | Observed | Verdict |
|---|---|---|
| Labels / posterior difference | | |
| Time ratio | | |
| CPU fallback | | |

## 9. SEQ: Graz cue-order prior (written 11 Nov ~11:15 UTC, before any chain score; scratch `runs/archive_scratch_11nov/seq/`)

**Finding (labels only, no EEG).** In all 42 public Graz runs (14 sessions, acquisition and online), the cue order is a
weakly shuffled repeating WORD → SUB → MI cycle: y_i equals "WSM"[i mod 3] in 56 % of positions (chance 33 %); the run's
first cue is WORD in 28 of 42. The 42 BrainHero runs show nothing (34–39 %). This fits the OpenViBE Lua stimulator,
which builds an ordered list and applies a limited number of random swaps (class counts stay 10 or 12).
Information content: 0.155 bit per Graz window.

| Check | Prediction (80 % range) | Observed | Verdict |
|---|---|---|---|
| Oracle structure (true interface and run position), prior × mix before rebalancing, weight 1 | dJ +2.5 (1.5 to 4.0); Graz cells +5 | dJ −2.94 (B2_pool); Graz cells −4.7 | **Miss; refuted** |
| Same, on the final output | dJ +1.5 (0.8 to 2.5) | −2.01 at weight 1; −0.46 at 0.5 | **Miss; refuted** |
| Fitted weight (LOPO, temperature a and prior weight g by NLL) | — | g = 1.01–1.04 at every stage (the prior is real and independent); a = 1.8–2.3 (posteriors are under-confident) | — |
| Gain in Graz window accuracy at the fitted weights, by stage | — | mix +4.1 (model expects +3.9); rebal +1.6 (+3.6); SessionEM +1.0 (+3.5); TSEM +0.7 (+3.1); TSEM cell BA −0.24 | — |
| Best fixed weight on the TSEM output (in-sample, one base) | — | Graz cells +0.8 at 0.05–0.1 → about +0.4 J, with oracle structure | Upper bound |

**Reading.**
- The prior alone is 56 % accurate, and the chain does not already use it: its accuracy is the same on and off pattern
  (76.0 / 77.1 %), and its off-pattern errors do not favour the pattern class (11.0 % against 11.5 % expected).
- It still adds almost nothing after the session stages. A flip to the pattern class is right 84 times and wrong 74
  times at the TSEM stage: the chain's confidence does not mark its own errors well enough to say where a 2.5 : 1 prior
  should override it. This is the same wall as "label-free selection has failed every time".
- As a label source for session templates it is worth about 4 clean windows per class per session
  (32 × 0.34²), which is negligible.
- My error: I priced the prior by its standalone accuracy instead of by its 0.155 bit and the chain's calibration.
- **Value for us: ≤ +0.4 J with oracle run structure, less label-free. Not pursued.** A weaker, better-calibrated base
  gains more (+3 points on Graz cells at the mix level), so other teams may be using it.
- **Open for the user:** whether to tell the organisers (it is a protocol artefact, not neural decoding).

## 10. BIO0: label-free spectral biometric for returning-person identification (written 11 Nov ~11:28 UTC, before any result; scratch `runs/archive_scratch_11nov/bio/`)

Feature: per-window log-PSD shape (CAR, 2–40 Hz, per-channel gain removed), session mean over the first n windows,
correlated with each gallery person's mean. Probes: all 14 sessions, leave-one-session-out; genuine trials exist only
for P04 and P10 (6 probes).

| Check | Prediction (80 % range) | Observed | Verdict |
|---|---|---|---|
| Rank-1 correct, n = 10 | 5 of 6 (4–6) | 4 of 6 | Hit (low end) |
| Rank-1 correct, n = 60 | 6 of 6 | 6 of 6 | Hit |
| Genuine above the largest new-person score, n = 10 | 4 of 6 (2–6) | 2 of 6 | Hit (low end) |
| Genuine above the 90th percentile of new-person maxima, n = 20 (refutation bar: ≥ 4) | — | 3 of 6 | **Refuted for this feature** |
| Full session | — | 5 of 6 above every new-person score | — |

**Reading.** Closed-set identity is in the spectrum (6/6 by window 60, all three P10 sessions clean from window 5), but
this raw score cannot reject new people early: P04's genuine scores (0.3–0.5) sit inside the new-person range
(0.4–0.6). It is not better than the shipped class-fingerprint gate. A discriminant trained on external multi-session
corpora (within-person against between-person variance) is the untested next step; its payoff is capped by the fact
that only P04-like people gain from a personal head.

## 11. BIO1: discriminant person identification trained on external multi-session corpora (written 11 Nov ~11:38 UTC, before any result; code `experiments/bio_id.py`; Galvani job 3087013)

**Mechanism.** BIO0 failed open-set because a raw correlation has no model of what varies within a person across
sessions. A two-covariance Gaussian model (between-person, within-person across sessions, window noise / n) fitted in
PCA space on external people (Lee2019, Stieger2021, Yang2025, Kumar2024, GuttmannFlury2025, Zhou2020) gives a
same-person log-likelihood ratio that accounts for n. Held-out external people measure it honestly; PROTEUS is scored
leave-one-session-out with the model fitted on external data only.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| External, held-out people, 10-person gallery, 1 gallery session: correct accepts at 2 % false accepts, n = 20 windows (best view) | 55 % (30–80) | < 30 % |
| Same, n = 60 | 70 % (45–90) | — |
| External rank-1 at n = 20 | 85 % (65–95) | — |
| PROTEUS at the external 2 % threshold, n = 20: genuine P04/P10 probes accepted | 4 of 6 (2–6) | < 3 |
| PROTEUS new-person probes above that threshold (of 14) | 1 (0–3) | > 4 |

**Gate for building routing v4 on it:** external correct accepts ≥ 60 % at n = 20 and 2 % false accepts; PROTEUS ≥ 5 of
6 genuine accepted with ≤ 1 of 14 impostors at the externally fixed threshold.

| Item | Observed (Galvani 3087096, 18 min; best view = PSD shape; the band-correlation view is poor, rank-1 75–81 %) | Verdict |
|---|---|---|
| External, held-out 67 people: correct accepts at 2 % false accepts, n = 20 | 75.0 % (PCA 64), 76.3 % (PCA 128), 71.8 % (PCA 32) | **Hit** (predicted 55, range 30–80) |
| Same, n = 60 | 86.8 % / 82.9 % / 85.9 % | **Hit** (high end of 45–90) |
| External rank-1, n = 20 | 95.6 % (n = 5: 92.3 %) | **Miss, above the range** (65–95) |
| External correct accepts at 1 % false accepts, PCA 64, n = 5 / 10 / 20 / 40 / 60 | 54.6 / 61.7 / 70.5 / 78.6 / 81.5 % | — (the shipped class-fingerprint gate: 10–42 % at session end on Lee2019 / Stieger2021) |
| PROTEUS genuine P04/P10 probes accepted at the external 2 % threshold, n = 20 (PCA 64) | 5 of 6 (n = 40 and 60: 6 of 6; n = 10: 4 of 6) | **Hit** |
| PROTEUS new-person probes above that threshold | 0 of 14 at every n (PCA 64); the largest is 0.17–0.18 against thresholds 0.20–0.23 | **Hit** |
| Gate | external 75 % ≥ 60 %; PROTEUS 5 of 6 with 0 of 14 | **PASS** → routing v4 (§14) |

**Reading.** Identity is in the gain-free spectral shape and survives across sessions once the within-person variation
is modelled. It needs no pseudo-labels, so it works from window 5–10, where the class-fingerprint gate cannot. The
margin on PROTEUS is thin for one case (P10 session 1 as a new person scores 0.18). The dev set cannot show the main
benefit: all four dev seen sessions are already routed by v3, so the gain from a higher acceptance rate on sealed
returning sessions rests on the external rates.

## 12. SBNX: SessionBN versions of untried member families (written 11 Nov ~11:40 UTC, before any code ran; Galvani smoke 3087028, array 3087029; scripts `runs/archive_scratch_11nov/sbnx/galv/`)

**Mechanism.** Session normalisation lifted FBCNet from 59 to 66 and ShallowConvNet from 58 to 62 on new people, and
the ensemble's gain is member diversity. TIDNet, CTNet, EEGConformer, ATCNet, EEGSimpleConv, FBMSNet, SCCNet and Deep4
were only ever tried with plain batch norm (45–48 % alone) or not at all. Same recipe as the FB slot, seeds 0–1.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Best family, standalone new (k 2, raw argmax, 2 seeds) | 61 (55–66); SBN FBCNet is 66 | — |
| Families that train and reach ≥ 58 new | 3 of 8 (1–5) | 0 |
| Best single family as an extra chain member, nested dJ | +0.15 (−0.20 to +0.50) | ≤ 0 for every family |
| P(gate pass: nested ≥ +0.5, both halves ≥ 0, seen ≥ −0.3) | ≈ 8 % | — |

| Item | Observed (15 of 16 tasks; the first, over-packed job was cancelled and the single-task array rerun) | Verdict |
|---|---|---|
| Standalone new / seen at k 2 (2 seeds) | FBMSNet 62.3 / 61.8 (1 seed); TIDNet 53.4 / 51.1; SCCNet 50.5 / 43.2; Deep4 50.3 / 46.7; CTNet 49.6 / 44.7; EEGConformer 48.4 / 43.4; ATCNet 47.1 / 44.4; EEGSimpleConv 40.7 / 40.7 | **Hit** on the best family (predicted 61); 1 of 8 reaches 58 (predicted 3, range 1–5: hit at the edge) |
| Extra chain member, best weight per family | TIDNet +0.08 at 0.5; SCCNet −0.07; FBMSNet −0.15; all others −0.28 to −0.59; every family is negative at weight 1 and 2 | **Hit** (predicted best +0.15, range −0.20 to +0.50) |
| Nested selection | both halves pick TIDNet at 0.5; held-out new +0.16 / −0.09 | **FAIL**; inside the null range |

**Reading.** Session normalisation does not rescue these families: only the filter-bank design (FBMSNet, like FBCNet)
reaches the strength of our existing members, and it adds nothing new to them. New architectures are closed as a source
of members.

## 13. WSP: profiling and editing core1's weights (written 11 Nov ~12:15 UTC, while the cache was building and before any profile or edit result; code `experiments/wsp_build.py`, `experiments/wsp_core.py`)

**Why core1.** It is our strongest single member (69.3 on new people), it is linear (a 3 × 44,239 logistic head on
standardised tangent features: 17 bands × {full, half 1, half 2} × 861 covariance entries, plus a slow block), and every
weight has a physical meaning (band, time half, channel pair). An edit is a matrix operation and scores in seconds.

**Rule for every edit:** it may use only the fold's training people. Each fold therefore carries "inner" heads refitted
with one more training person left out; the held-out person's rows are used only for scoring.

**Profile (descriptive predictions, 80 % ranges):**

| Quantity | Prediction |
|---|---|
| Held-out margin as a share of in-sample margin (whole head) | 12 % (6–25) |
| Share of weight energy in off-diagonal (connectivity) entries | 90 % (80–97) |
| Share of held-out margin from off-diagonal entries | 65 % (40–85): power entries transfer better per unit of weight |
| Bands above 30 Hz: share of weight energy vs share of held-out margin | energy 25 % (15–35), margin 12 % (0–25) |
| Slow block: share of held-out margin | 10 % (4–20) |
| Weights whose sign is unstable across leave-one-person-out heads (jackknife |mean| < sd) | 55 % of weight energy (35–75) |

**Edits (standalone = core1 alone on new people, person-equal; chain = route3geb replay with the edited core in the
core1 slot, mean of B2_pool and B2_s345):**

| Edit | Mechanism | Standalone new (80 %) | Chain dJ (80 %) |
|---|---|---|---|
| E2 jackknife shrinkage | each weight × m² / (m² + v), m and v its mean and person-jackknife variance over the inner heads | +0.5 (−0.5 to +1.5) | +0.05 (−0.30 to +0.40) |
| E1 group gains | each band × time-half block scaled by its inner held-out / in-sample margin ratio (mean 1, ≥ 0) | +0.3 (−0.7 to +1.3) | 0.0 (−0.4 to +0.4) |
| E3 band smoothing | weights averaged with the neighbouring overlapping bands (ρ 0.5) | 0.0 (−1.0 to +0.7) | −0.1 (−0.5 to +0.3) |
| E4 half-window tying | the two half-window weight blocks pulled to their mean (ρ 0.5) | −0.2 (−1.2 to +0.5) | −0.1 (−0.5 to +0.3) |
| E6 magnitude pruning | smallest 50 % / 90 % of standardised weights set to 0 | −0.1 / −1.0 | −0.1 / −0.4 |
| E8 jackknife soup | weights = mean of the inner heads | 0.0 (−0.5 to +0.5) | 0.0 (−0.3 to +0.3) |
| Controls | E1 gains permuted across blocks; E2 factors permuted across features | below the real edit | — |

**Gate:** chain dJ ≥ +0.5 with both people halves ≥ 0 and seen ≥ −0.3, then the four fresh bases. P(any edit passes) ≈ 6 %.
Prior evidence against: fully converged heads are worse than the early-stopped ones, the C sweep is flat around 10, and
domain-generalisation heads (group DRO, Fishr, person bagging) were negative on both cores.

| Item | Observed | Verdict |
|---|---|---|
| Reproduction of the stored core1 posteriors | max abs difference 7e-9 (all_loso), 0 (seen); argmax identical; chain J 80.0131 / 79.8474 reproduced exactly | Harness valid |
| Held-out / in-sample margin | 41 % (per person 1.9 for P04 to 10.2 for P08, against 12–14 in-sample) | **Miss, above the range** (6–25): the head transfers better than assumed |
| Off-diagonal entries: weight energy / held-out margin | 92.4 % / 82.4 % (power entries: 4.3 % of energy, 11.8 % of margin) | **Hit / hit**: power transfers 2.7× better per unit of weight |
| Bands above 30 Hz: energy / margin | 17 % / 4.3 % (above 24 Hz: 22 % / 5.8 %) | **Hit / hit** |
| Slow block: held-out margin | 5.8 % | **Hit** |
| Unstable-sign weights | 18 % of weight energy | **Miss, below the range** (35–75): the weights are stable across people |
| Where transfer concentrates | 10–14 Hz (9.7 % of energy, 18.3 % of margin), 3–6 Hz (5.8 / 12.5), 4–8 Hz (6.4 / 10.1); weak: 6–10 Hz (5.6 / 3.0) and everything above 24 Hz; full window 33 / 39, each half 31–33 / 27–28 | — |
| E2 jackknife shrinkage | standalone +0.05 (5/10), seen −0.07; chain −0.18 (control −0.11) | **Hit**; no gain |
| E1 group gains | standalone −2.12 (3/10); chain −0.46; permuted control −2.00 / −0.45 | **Miss, below the range**; any block rescaling of this size hurts |
| E3 band smoothing | standalone −0.80; chain −0.21 | **Hit** |
| E4 half-window tying | standalone −2.35 (1/10); chain −0.19 | **Miss** standalone; the halves carry different information |
| E6 pruning 50 % / 90 % | standalone −0.25 / −1.31; chain −0.08 / −0.69 | **Hit** |
| E8 jackknife soup | standalone −0.45; chain −0.27 | **Hit** |
| Gate | best chain dJ −0.08; nothing ≥ 0 | **FAIL** (as expected at ≈ 6 %) |
| E9 learned block gains (exploratory, added after E1 failed; `experiments/wsp_gains.py`: 52 gains fitted by cross-entropy on inner held-out people, ridge λ 0.3 / 1 / 3) | standalone −0.13 / +0.32 (7/10) / −0.01; chain −0.14 / −0.01 / −0.00. Every fold gives the slow block the largest gain (1.5–1.8× at λ 0.3) | No gain; the optimiser agrees the block weights are already right to within noise |

**Reading.**
- The fitted head is close to the best linear map of these features: every hand edit lowers the chain, including the
  two built from the transfer profile.
- **Margin share is not usefulness.** Blocks above 24 Hz hold 22 % of the weight for 6 % of the held-out margin, yet
  scaling blocks by their transfer ratio costs 2 points, the same as a random rescaling. The weights are a joint
  solution, and low-margin blocks cancel noise in the others.
- **The weights are not the bottleneck; the per-person class directions are.** The head keeps 41 % of its margin on
  unseen people on average but 15 % on P04, and no person-independent edit of one shared head can fix a person whose
  class directions differ.
- Useful by-products: the neuro-physiological picture (mu 10–14 Hz and theta 3–8 Hz carry the transferable signal;
  power transfers better than connectivity) and a 20-second edit-and-score loop for core1 (`experiments/wsp_core.py`).

**Weco search over edits of core1's head (11 Nov, run `ea473cae-53bd-41fd-963b-fa11446fd3f1`, 60 steps, 56 edits scored; task in `.weco/wsp-core1/`).**
The optimiser (default model) could rewrite one function that edits the head from the fitted weights, the
leave-one-more-person-out heads, the per-feature transfer statistics and the feature layout. It saw only the chain score
on people half G1; half G2, the seen people and J were logged out of its sight. Each step cost about 20 s locally and
about 0.07 credits (4.1 in total).
- One of 56 edits was positive on the half it optimised (+0.11), and that edit was −0.11 on the hidden half (J +0.04).
- Median edit: −0.44 on its half, −0.12 on J. Eight edits were marginally positive on the hidden J (best +0.07), all
  inside the null range.
- Its attempts were the structured ones suggested (rank-k spatial denoising, projection onto the subspace shared by the
  inner heads, contrast-specific rescaling, boosting power entries and the slow block).

**Verdict:** an automated search agrees with the hand edits: the fitted head cannot be improved by editing its weights.
Weco works here as a tool (cheap steps, a hidden half as the honest check); the target was already at its optimum.

## 14. Routing v4: the BIO1 identity gate in front of the personal heads (written 11 Nov ~12:30 UTC, before any routing score)

**Mechanism.** Routing v3 opens at window 60 on a class-direction fingerprint that needs pseudo-labels; on external
data it accepts 10–42 % of genuine sessions at 1 % false accepts. The BIO1 score needs no labels and accepts 62 % at
window 10 and 70–80 % at windows 20–60. v4 keeps everything else of v3 (PS1 head at 3× in the core1 slot of a second
tail) and takes, per window, the larger of the two ramps: r = max(r_v3, r_bio), with
r_bio = clip((s − 0.23) / 0.05, 0, 1) from window 10 (0.23 = the external 1 % threshold at 20–60 windows; model
= PSD shape, PCA 64, fitted on all 224 external people; PROTEUS only centres it and provides the galleries).

**What the dev set can and cannot show.** New people: any routing is an impostor routing, so the dev set tests safety.
Seen people (P04 and P10 sessions 2–3, one gallery session each): v3 already routes all four, so the dev gain is only
the earlier start. The larger expected benefit, more returning sessions accepted on sealed, is not visible on dev and
rests on the external rates in §11.

| Check (primary: max of the two ramps, start window 10) | Prediction (80 % range) | Refuted if |
|---|---|---|
| New-person sessions routed, all bases | 0 of 14 | ≥ 1 |
| Seen sessions with r_bio > 0 by window 20 (one gallery session each) | 3 of 4 (2–4) | ≤ 1 |
| Seen gain over v3, mean of B2_pool and B2_s345 | +0.5 (0.0 to +1.0) | < 0 |
| P10 change against v3 | −0.2 (−0.8 to +0.4) | — |
| dJ against v3 | +0.25 (0.0 to +0.5) | < 0 |

**Gate for a package build:** 0 new-person routings on all six bases; seen gain ≥ +0.3 on the exploration mean and
≥ 0 on each of the four fresh bases; P10 ≥ −0.3 on every base. Then stress tests (noise, constant, artefact and
amplitude-scaled sessions must not route) before any build.

| Item | Observed (exploration bases B2_pool / B2_s345; streaming class `experiments/bio_gate.Bio`) | Verdict |
|---|---|---|
| New-person routings at τ 0.23 | 1 of 14: P10 session 1, as a new person, scores 0.253 against P03 at window 76 and is routed for 34 % of its windows (cost −0.03 / −0.02 on new). The other 13 sessions peak at ≤ 0.09 | **Refuted** (predicted 0). The threshold came from fixed-n false-accept rates; the maximum over a whole session is higher |
| Seen sessions routed by the identity gate by window 20 | 4 of 4 (windows 15, 10, 10, 10); genuine maxima 0.295 / 0.478 / 0.353 / 0.375 with one gallery session | **Hit** (predicted 3, range 2–4) |
| Seen gain over v3 | +1.18 / +1.45 (P04 +1.83 / +2.38) | **Miss, above the range** (0.0 to +1.0) |
| P10 against v3 | +0.52 / +0.52 | **Hit** (above the centre) |
| dJ against v3 | +0.57 / +0.72 | **Miss, above the range** (0.0 to +0.5) |
| Variants | start window 20: +0.50 / +0.64; identity gate alone (no v3): −0.02 / +0.25 | The two gates are complementary: v3 keeps sessions routed late, the identity gate starts them early |

**Amendment (written 11 Nov ~12:45 UTC, before the external session-level calibration result).** The threshold is
re-derived at the session level: on held-out external people a probe is scored at every 5th window from 10 to 200 and
counts as accepted if it is ever above τ (job running: `BIO_STREAM` mode of `experiments/bio_id.py`). Rule fixed now:
τ = the smallest value with external session-level false accepts ≤ 2 %; ramp 0.05 unchanged; if that τ leaves fewer
than 3 of the 4 dev seen sessions routed early, also report the 5 % threshold. The gate above is then judged at the
amended τ on all six bases. Predicted: τ ≈ 0.27 (0.25–0.31); dev dJ +0.45 (+0.2 to +0.7); 0 of 14 new-person routings.

| Item | Observed | Verdict |
|---|---|---|
| External session-level calibration (720 impostor and 520 genuine session trials, 67 held-out people, one gallery session, scored every 5th window from 10 to 200) | false accepts / correct accepts: τ 0.23 3.2 % / 88.3 %; **τ 0.25 1.9 % / 86.9 %**; τ 0.27 1.5 % / 85.6 %; τ 0.30 0.6 % / 81.5 %. Wrong-identity accepts of genuine sessions 0.6 % at 0.25. Median first accept at window 10. Requiring 2–3 consecutive hits changes little | Rule gives **τ = 0.25** (predicted 0.27, range 0.25–0.31: hit at the edge) |
| Dev at τ 0.25, six bases (B2_pool, B2_s345, B2_s012, B2_s0, B2_s1, B2_s2) | dJ against v3: +0.52 / +0.72 / +0.59 / +0.53 / +0.45 / +0.72 (mean +0.59). Seen +1.04 / +1.45 / +1.18 / +1.06 / +0.89 / +1.43; P04 +1.56 to +2.60; P10 +0.52 / +0.52 / +0.52 / −0.24 / +0.02 / +0.26. New 0.00 to +0.02 | **Miss, above the range** on dJ (+0.2 to +0.7 predicted with centre +0.45; mean +0.59 inside, two bases at the top) |
| New-person routings at τ 0.25 | 1 of 14 on every base: P10 session 1 (0.253 against P03) from window 93, 9 % of its windows at ramp ≤ 0.06; no measurable cost. Expected count at a 1.9 % rate: 0.27 | **Refuted** as predicted (0), harmless in effect |
| Stress tests with the package gallery | zeros, white noise, common 50 Hz, constant offset, pink noise: max score ≤ +0.10. A real session scaled ×10 or ×0.1, time-reversed, or with one flat channel is still identified (0.68–0.71); with channels permuted or +20 µV white noise it is not (−0.03, −0.07). 18 Scherer 2015 sessions (other hardware): about −3. All 14 public sessions in-sample: 0.58–0.74, all correct | Safe failure modes |
| Gate | seen ≥ +0.3 on exploration (+1.25) and ≥ 0 on every fresh base ✓; P10 ≥ −0.3 on every base ✓; "0 new-person routings" ✗ (one, at negligible weight) | **Passes on effect, fails the zero-routing clause as written.** That clause was mis-specified for a gate designed to accept 2 % of impostors. Built as a checkpoint; not uploaded |

**Reading.**
- On dev the identity gate is worth about +0.6 J, all of it from starting the personal head 50 windows earlier for people
  it already routed.
- The part dev cannot show is the acceptance rate on sealed returning sessions: 87 % on external people against 10–42 %
  for the class-fingerprint gate. If sealed returning people gain from a personal head as P04 and P10 do on average,
  that is the larger effect; if they behave like P10 alone, it is small.
- The one false accept (P10's first session against P03) is a real person-pair resemblance, not a threshold artefact.

**Second amendment after the independent review (11 Nov ~16:20 UTC; rule unchanged, protocol made package-faithful).**
The review pointed out that the session-level calibration scored every 5th window and centred with a mean that included
the probe, while the package scores every window and never has the probe in its centring. Recalibrated on the same 67
held-out external people with every window from index 10 and the probe excluded (Galvani 3088231, 720 impostor and 520
genuine session trials, one gallery session):

| τ | False accepts | Correct accepts | Wrong identity |
|---|---|---|---|
| 0.23 | 3.8 % | 88.8 % | 0.8 % |
| 0.25 (first build) | 2.9 % | 86.7 % | 0.8 % |
| **0.26** | **1.9 %** | **86.7 %** | 0.7 % |
| 0.27 | 1.7 % | 86.7 % | 0.6 % |
| 0.30 | 0.6 % | 83.8 % | 0.2 % |

- The same rule (smallest τ with at most 2 % session-level false accepts) now gives **τ = 0.26**.
- Three-session galleries could not be tested externally: no exported corpus has more than three sessions per person.
- Dev at τ 0.26, six bases: dJ against v3 +0.52 / +0.66 / +0.52 / +0.66 / +0.45 / +0.65 (mean +0.58); seen +0.90 to
  +1.32; P10 −0.25 to +0.52; new exactly 0.00 with **0 of 14 new-person routings** (the P10-session-1 case scores 0.253).
- **Gate at τ 0.26: PASS on every clause as written.** The package is rebuilt as `cand_v41_bio` (τ 0.26, plus the
  review's assertion that the gallery has one row per personal head).

**Sealed result (upload 4, submission 1048575, 11 Nov).** Predicted +0.2 to +0.8 sealed points; observed **+0.267**
(0.8161546323 against 0.8134828319), duration 1,128.6 s. **Hit, at the low end.** The dev gain (+0.58 J) transferred at
0.46×. The larger hoped-for effect, more returning sessions accepted, is not visible: consistent with the eight
single-session public people gaining little from a personal head. By the reading rule fixed before the score this is
the neutral band, and the returning-person line stops.

## 15. PERS: who gains from a personal head? A within-session proxy for all ten people (written 11 Nov ~13:20 UTC, before any result; code `experiments/pers_proxy.py`)

**Why.** Routing v4's sealed value depends on how the eight single-session public people respond to a personal head,
and they have no later public session to test on. Dev knows only P04 (about +2) and P10 (about +0.3).

**Proxy.** For each person, split their first session: runs 1–3 are treated as the "public session", runs 4–6 as a
"later session" (own causal stream). core1 is refitted on everyone else plus the person's runs 1–3; a personal head
(PCA 256, per-stream standardisation, logistic regression C 0.1, as PS1) is fitted on runs 1–3 only; the mix puts the
head at 3× core1. For P04 and P10 the real cross-session version (session 1 → sessions 2–3) calibrates the proxy.
Core level only (no session stages), so absolute sizes are not chain sizes; the question is the spread across people.

| Quantity | Prediction (80 % range) |
|---|---|
| Proxy gain of the mix over core1, mean of 10 people | +4 BA points (0 to +8) |
| People with a gain ≥ half of P04's | 5 of 10 (3–8) |
| Spearman between a person's gain and their core1 weakness (100 − BA) | +0.4 (0 to +0.8) |
| P04 and P10, real cross-session gain as a share of their proxy gain | 60 % (30–100): the proxy has no between-session shift |

No gate: this sizes the bet on v4; it does not change the package.

| Item | Observed | Verdict |
|---|---|---|
| Proxy gain of the mix over core1, mean of 10 | +5.78 | **Hit** (predicted +4, range 0 to +8) |
| Per person (core1 → mix) | P05 +20.0 (62 → 82); P01 +11.1 (69 → 80); P04 +8.9 (52 → 61); P09 +7.8 (80 → 88); P07 +4.4; P06 +2.2; P10 +2.2; P08 +1.1; P02 0.0; P03 0.0 | — |
| People with a gain ≥ half of P04's | 5 of 10 | **Hit** |
| Spearman(gain, core1 weakness) | +0.52 | **Hit** (predicted +0.4) |
| Real cross-session gain, P04 / P10 (session 1 → sessions 2–3) | +2.6 (29 % of its proxy) / −1.0 (proxy +2.2) | **Miss, below the range** (predicted 60 %, 30–100): the session change removes most of the within-session gain |
| Same, with only runs 1–3 of session 1 as the public part | P04 +3.2; P10 +1.6 | — |

**Reading.**
- P04 is not an outlier: four other people gain as much or more within a session, and the gain follows weakness of the
  population model for that person, as the 7 Nov note guessed from two people.
- Most of the gain does not survive a session change at core level (P04 keeps 29 %, P10 turns slightly negative), so
  the cross-session robustness of the personal model, not identification, is now the limiting factor for returning
  people.
- With the two calibration points, a rough reading for the eight others is a mean cross-session gain of about +1 at
  core level, concentrated in P05, P01 and P09. The chain is kinder than the core (v4 gives P10 0 to +0.5).

## 16. SIM: does spectral similarity pick a useful person model for NEW people? (exploratory diagnostic, 11 Nov ~13:15 UTC; not pre-registered; log `evidence/wsp_bio1_11nov/simdiag.log`)

For each new-person session (14, leave-one-person-out), the identity score against each of the nine other people was
compared with how well that person's personal head decodes the session.

- Rank correlation between similarity and head accuracy: +0.21 at window 20, +0.27 at the session end, from −0.57 to +0.83.
- Balanced accuracy: head of the most similar person 47.1; mean head 44.1; best head (oracle) 53.3; similarity-weighted
  heads 51.5; uniform mixture of the nine heads 57.6; pooled core1 66.2.

**Reading.** Similarity is weakly informative, but weighting by it loses to a plain average, which loses to the pooled
model. Spectral resemblance to a public person does not say whose class directions a new person shares. Similarity
routing for new people is closed in this form.

## 17. PERS2: personal-model recipes on ten people (written 11 Nov ~13:35 UTC, before any result; code `experiments/pers_variants.py`)

**Why.** Every earlier personal-model choice (PS1 head, weight 3×) was made on P04 and P10 alone. The within-session
proxy of §15 gives ten people; the two real cross-session cases stay as the check that a recipe survives a session
change. Core level only.

| Recipe | Prediction: mean proxy gain over core1 (10 people) | Prediction: P04 / P10 cross-session |
|---|---|---|
| PS1 mix at 3× (the shipped recipe) | +5.8 (the §15 value) | +2.6 / −1.0 (the §15 values) |
| Person-upweighted core (weight 5 or 20 on the person's public windows) | +3 (0 to +6): smaller than PS1 within-session | +1.5 / 0.0: more robust across sessions than the 256-dimensional head |
| Upweighted core + PS1 | +6.5 (+4 to +9) | +3.0 / −0.5 |
| Personal recentring prior, k 10 (public session's reference instead of the population's) | not meaningful within-session | core alone +0.5 / +0.3 (−0.5 to +1.5): helps only the first windows |

A recipe is carried into the routed chain only if it beats the shipped one on the ten-person mean **and** on both real
cross-session cases.

| Item | Observed (gain over core1: ten-person proxy mean / P04 cross-session / P10 cross-session) | Verdict |
|---|---|---|
| PS1 mix at 3× (shipped) | +5.78 / +2.64 / −1.03 | Reference |
| PS1 mix at 1× / 6× | +3.56 / +4.77 / +0.79; +3.44 / +0.52 / −2.08 | A weaker head is better across sessions, worse within |
| PS1 with C 0.03 / C 0.3 (at 3×) | **+6.56 / +3.96 / −0.51**; +4.78 / +1.05 / −1.03 | C 0.03 beats the shipped recipe on all three: passes the rule |
| Person-upweighted core, weight 5 / 20 | +2.67 / +0.52 / +2.05; +0.56 / −1.33 / +4.42 | **Miss** on P04 (predicted +1.5); inconsistent between the two people |
| Upweighted core (5) + PS1 | +6.44 / +2.91 / −1.30 | Hit; no better than PS1 alone across sessions |
| Personal recentring prior, k 2 / k 10 (core alone) | +0.11 / +0.27 / −0.27; +0.44 / −0.80 / −0.53 | **Refuted**: the public session's reference is no better a prior than the population's |
| Carried into the chain: PS1 with C 0.03, routing v4, six bases | absolute J 80.14 / 80.24 / 80.07 / 79.91 / 79.91 / 80.59 (mean 80.14) against 80.53 / 80.57 / 80.73 / 79.71 / 80.17 / 80.93 (mean 80.44) with C 0.1 | **Loses 0.30 in the chain** (5 of 6 bases); the shipped recipe stays |

**Reading.** At core level a more regularised personal head survives a session change better, but the chain does not
want it: the head's job there is to sharpen the seeds of the session stage, and a flatter head does that less. Core-level
recipe comparisons do not predict chain behaviour for personal models; future personal-model ideas must be scored in the
chain. The cross-session loss remains unsolved.

## 18. TSCH: Scherer 2015 as an eighth corpus of the external teacher (written 11 Nov ~13:45 UTC, before any job ran)

**Mechanism.** The teacher (EEGNeX + IFNet trunks pretrained on seven corpora, one head per corpus, mapped to our
classes) carries the largest member weight (5) and is right on 42 % of the chain's motor-imagery errors. Its only
non-MI evidence is Shin 2017B (subtraction vs rest). Scherer 2015 has all three of our tasks from nine more people and
is now cleared by the organisers. Earlier it was only tried as extra rows for PROTEUS-trained members (§7), never in
the teacher.

**Runs.** `experiments/pretrain.py`, the shipped m7r recipe (30 epochs, rot 0.05), seeds 0–5, both trunks:
- **m8s** = the seven m7r corpora + `scherer2015t` (the Scherer windows with a 3-class corpus head);
- **m7c** = the seven corpora again, same seeds and hardware, as the control (the shipped m7r checkpoints were trained
  on other hardware, which moved results by about ±0.3 before).

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Teacher alone on new people (m8s − m7c, mapped posteriors) | +1.0 (−0.5 to +2.5) | ≤ 0 |
| Chain with the teacher swapped (m8s − m7c), nested over people halves | +0.20 (−0.20 to +0.60) | ≤ 0 |
| WORD recall in the chain | +0.5 (−1 to +2) | — |
| P(gate: nested ≥ +0.4, both halves ≥ 0, seen ≥ −0.3) | ≈ 15 % | — |

Limits known beforehand: Scherer's participants have stroke or spinal-cord injury, 22 of 41 channels are interpolated,
and the files are high-passed near 4 Hz.

| Item | Observed | Verdict |
|---|---|---|
| Pretraining | 24 of 24 jobs (2080 Ti, 26–70 min each); head export 3087589; teacher rebuilt locally (seed-mean head log-probs of EEGNeX + IFNet, balanced logistic map C 0.01 per fold) | — |
| Harness | the shipped recipe's own checkpoints (seeds 0–5) rebuilt this way give chain J 80.20 / 79.86 against 80.01 / 79.85 for the packaged 12-seed teacher; the control m7c gives 80.20 / 79.90 | Close enough for a paired comparison against m7c |
| Teacher alone on new / seen people | m7c 50.46 / 50.85; **m8s 53.73 / 54.15** (SUB recall 34.0 → 40.5, MI 65.9 → 67.2, WORD 47.9 → 47.9) | **Hit, above the range** (+3.3 against a predicted +1.0, range −0.5 to +2.5) |
| Chain with the teacher swapped at weight 5 (m8s − m7c) | **−0.98** (new −1.39: half G1 −2.70, half G2 −0.09, 3 of 10 people ≥ 0; seen −0.57) | **Refuted; miss far below the range** (−0.20 to +0.60) |
| Weight sweep (exploratory), m8s − m7c at weight 1 / 2 / 3 / 5 / 7 | +0.07 / −0.41 / −0.58 / −0.98 / −0.74; m7c peaks at weight 5 (80.05), m8s never exceeds 79.07 | No weight rescues it |
| Gate | — | **FAIL** |

**Reading.** With Scherer the teacher becomes a better three-class decoder by itself and a worse chain member. Its
value in the chain is not its accuracy but that it is different: a motor-imagery specialist whose errors are unlike the
PROTEUS-trained members'. Teaching it subtraction and word generation from nine more people makes it more like them.
This is the fifth and clearest case today that the chain rewards complementary members, not stronger ones. Scherer is
now closed in the members, the cores and the teacher.

## 19. PTSEM: the returning person's own class templates as a prior in the routed tail's TSEM (written 11 Nov ~14:05 UTC, before any result)

**Mechanism.** In the chain, personal information enters only through the PS1 head. TSEM builds its class templates
from the current session's confident windows and shrinks them towards zero with 20 pseudo-windows. For an identified
returning person, part of that shrinkage target can be the person's own class means from their public session(s), in the
same 128-PC view, z-scored per session: mu_k = (sum of selected windows + κ·r·mu_k_personal) / (n_k + 20), with r the
routing ramp. Class contrasts are personal and moderately stable across sessions (the v3 gate relies on exactly that:
genuine cosine 0.40–0.54), and the prior acts where the session's own templates are weakest, in windows 15–60.

| Check (on routing v4, τ 0.25; κ 5 / 10 / 20 picked on the exploration bases, judged on the four fresh bases) | Prediction (80 % range) | Refuted if |
|---|---|---|
| Seen gain over v4 at the pick | +0.3 (−0.2 to +0.8) | ≤ 0 on the fresh mean |
| P10 | 0.0 (−0.5 to +0.5) | < −0.3 on any base |
| New people | 0.00 (the prior is only active on routed windows) | any change beyond the one mis-routed session |
| P(gate: fresh seen ≥ +0.3, P10 ≥ −0.3 everywhere) | ≈ 25 % | — |

| Item | Observed | Verdict |
|---|---|---|
| κ sweep, exploration bases (absolute J; v4 without prior 80.534 / 80.570) | κ 5: 80.605 / 80.639; κ 10: 80.607 / 80.504; κ 20: 80.666 / 80.631. Pick κ 20: seen +0.27 / +0.12 over v4 | Below the +0.3 centre; inside the range |
| Fresh bases at κ 20, change against v4 | seen −0.13 / +0.76 / +0.69 / +0.51 (mean +0.46); J −0.06 / +0.38 / +0.35 / +0.26 (mean +0.23) | **Hit** on seen (≥ +0.3) |
| P04 against v4, six bases | +0.79 / +0.50 / +0.54 / +1.56 / +1.90 / +0.80 | Consistent gain |
| P10 against v4, six bases | −0.26 / −0.26 / −0.78 / −0.04 / −0.53 / +0.22 | **Fails the P10 clause** (< −0.3 on two bases); not monotone in κ, so partly noise (0.26 = one window) |
| New people | 0.00 | Hit |
| Gate | fresh seen ✓, P10 ✗ | **FAIL as written.** Six-base mean +0.19 J, at the edge of the null range |

**Reading.** The personal template prior helps the person the population model is weak for (P04, every base) and
costs the strong one a window or two. That is the same pattern as PS1 itself and as the ten-person proxy: personal
information pays in proportion to the person's weakness. A per-person trust weight is the obvious refinement, but with
two dev people it cannot be validated honestly, so the prior is not added to the checkpoint. Kept as the first thing to
revisit if a sealed upload shows that returning-person levers transfer.

## 20. FORGET: a forgetting reference for core1's causal recentring (written 11 Nov ~14:35 UTC, before any result; code `experiments/forget_ref.py`)

**Observation (B2_pool, final chain accuracy by window position).** New people: 63 % in windows 0–5, 70 % in 5–10,
75–79 % from window 10 to 90, then 76.6 % in 90–130 and 76.5 % after 130. Seen people: 83–86 % in windows 20–90, then
78.7 % and 76.4 %. So the cold start costs only the first ten windows, and accuracy falls again late in the session.

**Mechanism to test.** Every causal reference in the pipeline (core recentring, EA, SessionBN) is a cumulative mean
from the session start. If the recording drifts within a session (impedance, fatigue, the switch between interfaces),
a cumulative reference goes stale. An exponentially forgetting reference tracks it:
R_t = (k·prior + Σ ρ^(t−i) L_i) / (k + Σ ρ^(t−i)), half-life h windows. Alternative explanation: the online runs are
simply harder, in which case forgetting changes nothing or hurts. Test-time change only; the heads are unchanged.

| Check (core1, heads from the weight-space cache) | Prediction (80 % range) | Refuted if |
|---|---|---|
| h = ∞ reproduces the cached core1 posteriors | max difference < 1e-6 | otherwise the harness is wrong |
| h = 80: standalone new | +0.3 (−0.5 to +1.0) | ≤ 0 |
| h = 80: accuracy in windows ≥ 90 | +1.0 (−0.5 to +2.5) | ≤ 0 |
| h = 30: standalone new | −0.5 (−2.0 to +0.5): too noisy a reference | — |
| Chain dJ with the best h in the core1 slot | +0.1 (−0.3 to +0.5) | ≤ 0 |

| Item | Observed | Verdict |
|---|---|---|
| Harness | h = ∞ reproduces the stored core1 to 6e-8 | Valid |
| h = 80 | standalone new +0.05 (6/10), seen +0.79; accuracy in windows ≥ 90: new 65.4 → 65.6, seen 62.7 → 63.9 | **Hit** on new (inside the range), late windows +0.2 against a predicted +1.0 (inside the range, near its lower end) |
| h = 30 | standalone new −0.48 (3/10), seen +1.04 | **Hit** |
| Chain, core1 slot | h 80: −0.04 (new −0.09, seen 0.00); h 30: −0.01 (new −0.16, seen +0.13) | **Refuted** (≤ 0) |

**Reading.** A reference that tracks drift does not recover the late-session loss: core1 itself is 10 points lower in
windows ≥ 90 for the seen sessions with either reference. The late windows are the online runs with live feedback, and
they are harder to decode, not staler. Forgetting references are closed for the cores.

## 21. CSCH: Scherer 2015 rows in core1's head (written 11 Nov ~14:50 UTC, before any result; code `experiments/core_scherer.py`)

**Why, despite §7.** core1 is the one member whose strength the chain is most sensitive to (it sits in the slot at
weight 2 with the Hilbert core, and the personal heads replace it), and it has never been fitted with Scherer. Cheap:
one local run.

**Setup.** core1 features for PROTEUS + Scherer (null space filled); scaler fitted on the PROTEUS training rows only;
logistic head with sample weight 0.1 or 0.25 on Scherer rows; for Scherer rows the blocks below 4 Hz and the slow block
are set to the PROTEUS mean (the source files are high-passed near 4 Hz). Weight 0 must reproduce the cached core1.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| core1 standalone new | +0.6 (−0.6 to +1.8) | ≤ 0 at both weights |
| Chain dJ with the core in its slot (exploration bases) | +0.05 (−0.3 to +0.4) | ≤ 0 at both weights |
| P(gate: nested ≥ +0.5, halves ≥ 0, seen ≥ −0.3) | ≈ 6 % | — |

| Item | Observed | Verdict |
|---|---|---|
| Harness (weight 0) | reproduces the stored core1 to 7e-9 | Valid |
| Scherer weight 0.1 | standalone new −4.00 (2/10), seen −2.68; chain −0.89 | **Refuted** |
| Scherer weight 0.25 | standalone new −3.31 (2/10), seen −1.54; chain −1.10 | **Refuted** (predicted +0.6, range −0.6 to +1.8: miss, far below) |

**Reading.** Scherer rows hurt core1 badly, the opposite of what they do to the nets. The tangent features of
recordings with 22 interpolated channels and no content below 4 Hz are not comparable to PROTEUS's, and a linear head
on 44k features cannot ignore that the way a small net with session normalisation can. Scherer in the cores is closed.

## 22. Slot weight of the personal head under routing v4 (exploratory, 11 Nov ~15:20 UTC; not pre-registered)

Absolute J on the six bases (B2_pool, B2_s345, B2_s012, B2_s0, B2_s1, B2_s2):
- weight 3 (shipped): 80.53 / 80.57 / 80.73 / 79.71 / 80.17 / 80.93;
- weight 2: 80.08 / 80.04 / 80.14 / 79.32 / 79.65 / 80.20 (about −0.5 on every base);
- weight 4.5: 80.7438 80.5044 80.4782 80.0991 80.6449 80.9329 

The shipped weight stays unless 4.5 is clearly better on all six; any change here is tuned on P04 and P10 alone.

## 23. TSEM-R: recency-limited seeds in TSEM (written 11 Nov ~17:55 UTC, before any result)

**Why.** The late-session loss (§20) is not a stale recentring reference. But the class templates of TSEM are also
cumulative: seeds are the most confident windows of the whole session so far, so in the online runs (from about window
69) they are still dominated by the acquisition runs, where there is no live feedback. Rebalancing already shows that
recency matters in this chain (a 30-window share beats a cumulative one by about 0.45 J). Never tested for TSEM.

**Change.** Seeds are drawn only from the last N windows (N = 60, 90, 120); the per-session standardisation stays
cumulative; everything else as shipped. Applied in both tails of routing v3 (the shipped chain).

| Check (exploration bases) | Prediction (80 % range) | Refuted if |
|---|---|---|
| Best N: dJ | +0.15 (−0.3 to +0.6) | ≤ 0 for every N |
| Accuracy in windows ≥ 90, new people | +0.5 (−0.5 to +1.5) | ≤ 0 |
| N = 60 | −0.2 (−0.8 to +0.3): too few seeds | — |
| P(gate: fresh ≥ +0.4, new ≥ +0.2, seen ≥ −0.2) | ≈ 10 % | — |

| Item | Observed (exploration bases; shipped chain J 79.930) | Verdict |
|---|---|---|
| N = 120 | −0.12 (new −0.06, seen −0.18); late-window accuracy unchanged (76.5 %) | **Refuted** |
| N = 90 | −0.41 (new −0.59) | **Refuted** |
| N = 60 | −0.93 (new −0.66, seen −1.21) | **Miss, below the range** |

**Reading.** TSEM wants every seed it can get; older seeds are not stale. Unlike the class shares (where a 30-window
view wins), the class templates are stable over the session. Recency in TSEM is closed; the late-session loss stays
unexplained beyond "the online runs are harder".

## 24. TTT: per-session gradient fine-tuning of the nets (written 11 Nov ~19:35 UTC, before any job ran; code `runs/archive_scratch_11nov/ttt/galv/ttt.py`)

**Why now.** Every adaptation in our chain is closed-form (statistics, templates). The nets themselves are frozen at
test time. Earlier test-time training (T-TIME, entropy and self-labels, round 12) used a single net's own predictions
on a much weaker system. Two things changed: the chain now supplies pseudo-labels that are right about 77–80 % of the
time, and the sealed runtime is 19 of 60 minutes, so a 3–4× heavier inference is affordable (the leader's jump came
with a 4× runtime increase).

**Design.** Per test session, every 10 windows from window 30, a copy of the trained SessionBN net is fine-tuned for
20 full-batch steps on all earlier windows of that session and predicts the next 10. Labels: the chain's own causal
outputs (`pl20`, packageable) or the true labels (`or20`, `or60`: ceilings). FBCNet and ShallowConvNet, seeds 0–2.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Ceiling: `or60` member standalone, new people | +8 over the frozen net (+4 to +14) | < +3: the nets cannot use session labels |
| Ceiling in the chain (`or60` replacing its frozen twin) | +1.0 J (+0.2 to +2.5) | < +0.5: the method class is dropped whatever `pl20` does |
| `pl20` member standalone, new people | +3 (+1 to +6) | ≤ 0 |
| `pl20` in the chain | +0.2 J (−0.3 to +0.8) | ≤ 0 |
| P(gate: nested ≥ +0.5, halves ≥ 0, seen ≥ −0.3) | ≈ 12 % | — |

| Item | Observed | Verdict |
|---|---|---|
| Ceilings | | |
| Pseudo-label variant | | |

## 25. Solution-space map and the wave it triggered (written 11 Nov ~19:35 UTC, before any result of the new jobs)

**Why.** Questions were being run one small job at a time, leaving most Galvani lanes idle. The map below lists the
method classes for this problem, what was tested, and what is thin. "Closed" covers only the recipes tested.

| Axis | Method class | Status |
|---|---|---|
| Features | Covariance / tangent space; filter-bank CNNs | Dense: two cores, about 65 nets, 48 + 77 core configs |
| Features | Raw-time CNNs, transformers, foundation models | Closed: 20 architectures, 4 pretrained models, 8 more with session normalisation today |
| Features | Phase coupling | Partly covered by the Hilbert core; Codex's rank-correlation heads failed |
| Features | **Amplitude-envelope coupling** | **Never tested → AEC core (this wave)** |
| Features | Cross-frequency coupling, source-space priors, time-frequency images | Never tested; no code path yet |
| Data | PROTEUS labelled windows; other window offsets | Dense; crops rejected |
| Data | External labelled corpora (MI teacher, Scherer, Shin, Zyma) | Dense; Scherer closed in members, core and teacher today |
| Data | **PROTEUS unlabelled recordings (rest runs, inter-trial) for self-supervised pretraining** | **Never used; explicitly allowed; needs a training pipeline** |
| Cross-subject | Alignment, session normalisation, augmentation, domain-generalisation losses | Dense |
| Test time | Statistics and template adaptation (rebalancing, SessionEM, TSEM, early boost) | Dense: 29 + 8 arms, at a local optimum; Weco search running |
| Test time | **Gradient fine-tuning of the nets inside the session** | **Thin: only entropy / self-label variants on an old weak system → TTT, TTT2 (this wave)** |
| Sequence | Class balance per run; Graz cue-order regularity | Used; ≤ +0.4 J for our chain |
| Sequence | Reconstructing the cue generator | Being tested (labels only); a leakage question, the user's decision |
| Persons | Identification and personal heads | Done: +0.27 sealed; personal templates not adopted |
| Combination | Fixed weights, stackers, per-session weights, weight edits | Dense; closed |

**Wave.** TTT (6 tasks, running), TTT2 (15 tasks: all five SessionBN member families × 3 seeds, six fine-tuning
variants), AEC core (1 task). Predictions for TTT are in §24.

| AEC check | Prediction (80 % range) | Refuted if |
|---|---|---|
| Standalone new (lr10) | 45 (38–55) | — |
| Extra chain member, best weight | +0.1 J (−0.2 to +0.4) | ≤ 0 |

| Item | Observed | Verdict |
|---|---|---|
| AEC | | |

## 26. CFC core and REST4 (written 11 Nov ~19:55 UTC, before any result)

**CFC: cross-frequency coupling core** (`runs/archive_scratch_11nov/aec/cfc_core.py`). For four slow/fast band pairs
(4–8 with 13–30 and 30–45; 8–13 with 30–45; 2–6 with 8–13) the joint covariance of the slow-band signals and the
slow-filtered fast-band log-envelopes, through core1's machinery. Mental arithmetic and verbal tasks are known for
frontal-midline theta and theta-gamma coupling; no member sees coupling between bands.

**REST4: PROTEUS's own rest periods as a fourth class** (`runs/archive_scratch_11nov/rest4/`). The recordings contain
far more signal than the 2,655 task windows. In Graz runs the 4 s before each cue (cue −4.5 to −0.5 s) are a clean
no-task period, interleaved with the tasks in the same runs. Nets are trained 4-way (MI / SUB / WORD / rest) and used
3-way. The external teacher's "task vs rest" heads were load-bearing, yet PROTEUS's own rest was never used. The
held-out person's rest windows are excluded from training. Control: the same code without rest rows.

| Check | Prediction (80 % range) | Refuted if |
|---|---|---|
| CFC standalone new (lr10) | 42 (36–50) | — |
| CFC as an extra chain member, best weight | +0.1 J (−0.2 to +0.4) | ≤ 0 |
| REST4 member standalone new, vs its twin | +0.8 (−0.7 to +2.5) | ≤ 0 for both architectures |
| REST4 in the chain (member replaced, vs twin replaced) | +0.15 J (−0.3 to +0.6) | ≤ 0 |
| P(any of AEC, CFC, REST4 passes: nested ≥ +0.5) | ≈ 12 % | — |

| Item | Observed | Verdict |
|---|---|---|
| CFC | | |
| REST4 | | |
