# Search-narrowing decisions, 17 Oct – 10 Nov 2026 (read-only reconstruction)

Root: `<repo>`. All `docs/...` citations below are under this root; line numbers are from the current files. Sources read: all 71 commits up to 10 Nov (subjects and bodies), every file in `docs/archive/`, `docs/REVIEW_5NOV.md`, `docs/SEALED_UPLOADS.md` (to line 215), `docs/COMPETITION.md`, `docs/AUDIT_SBN.md` (recipe introductions), and the later documents only where they judge an earlier closure (`docs/REVIEW_12NOV.md`, `docs/PREDICTIONS_12NOV.md:848-869`, `docs/LEDGER.md:1120-1152`).

Abbreviations: dl = `dev_loso`, ds = `dev_session`, dla = `dev_loso_all`, al = `all_loso`, ss = `seen_session`, sel = mean(ds, dla), J = person-equal 0.5 new + 0.5 seen (from 5 Nov).

## 0. Pipeline versions (needed to read "tested on which pipeline")

| Date | Version | Level |
|---|---|---|
| 18 Oct | v1 / v2 tangent space alone | dl 49.4 / 56.2 |
| 18 Oct | R2 ensemble (v2 + 10 ShallowConvNets); r3 (+FBCNet, rebalancing) | dl 63.0; dl 65.1, al 62.8 |
| 19–20 Oct | r6 / r6nf (windowed rebalancing, unaligned FBCNet) | dl 67.2–67.8, al 64.9–66.0 |
| 25–27 Oct | r8t4, r9 | sel 76.2, 77.7; al 66.3, 66.95 |
| 28–29 Oct | R10, R14, R15 | six-seed sel 78.30, 78.46, 78.68; dla 72.1, al 67.0 |
| 30 Oct | R19 / R19e (session EM) | dla ≈ 74.9, al ≈ 69.3 |
| 2 Nov | cand_nxif (R19e + EEGNeX/IFNet teacher) | sel +2.91 over R19e; sealed 0.7795 |
| 5–6 Nov | a115, cand_sbn, REC, +core1, +Hilbert core, +TSEM | sealed 0.8197 |
| 7 Nov | + routing v3 + gated early boost (route3geb) | sealed 0.8240 |

Two facts about the evidence base that apply to everything below:
- Tracked `evidence/` folders stop at `sealed-candidates-2026-10-26` and resume at `seq_bio_11nov`. For 27 Oct – 10 Nov the evidence is the documents plus gitignored `runs/` and session scratchpads. `docs/archive/HANDOFF.md:311-314` says the scratch "is not guaranteed to survive a reboot"; `docs/REVIEW_5NOV.md:396-397` records macOS deleting scratch files. Several 30 Oct closures cite only scratch logs.
- Only P04 and P10 have three public sessions (28 cells = 8 people × 2 + 2 people × 6). Every "seen" measurement rests on these two people.

---

## A. Lock-ins, chronological

**A1. Input contract and metric (17 Oct, `3de267e`; `docs/archive/BASELINE_RESULT.md`).** Fixed by the rules: one 41 × 480 window at 120 Hz, cue +1 to +5 s, official preprocessing, cell-mean balanced accuracy. Baseline Torch-Linear 37.75 % (one seed; a second run gave 37.82 %, unexplained, `docs/archive/SETUP_CHECK.md`). The single 4 s window was never changed at test time. Extra windows at other offsets were tried only as training rows (B, features F13).

**A2. Validation protocols (18 Oct, `887aa6d`; `docs/archive/OPTIMIZATION.md:25-38`).** `dev_loso` (8 non-test people, 20 cells, P10 = 6 of them) for selection; `dev_session` (P10 sessions 2–3, 4 cells) as later-session probe; `official` (P04, P09) and `all_loso` (28 cells) "evaluation only". Stated noise: about 7 points per cell. Alternatives compared: none recorded. Re-opened 21 Oct, 26 Oct, 30 Oct, 5 Nov (report 1).

**A3. Causal session recentring, prior k = 5 (18 Oct; `OPTIMIZATION.md:62-66`).** Evidence: v2-family standalone, deterministic: no alignment 42.2 dl vs aligned 49.4; +8 on official. Alternatives: k = 20 (loses 1–4), training-time causal recentring, dispersion stretching (≤ 1.5, "within noise, simplest kept", lines 80-82). Nets use Euclidean alignment with the same k 5 (aligned vs raw ShallowConvNet 53.6 vs 53.2 dl, 3 seeds). Re-opened: label-shift-aware reference (20 Oct), scalar RMS (28 Oct), per-interface (29 Oct), band-limited (30 Oct), trace clip R20 (31 Oct), all closed; changed only on 6 Nov, when core1 took log-Euclidean recentring with k 2 and "smaller k" was named a driver (`docs/REVIEW_5NOV.md:185-186`).

**A4. v2 feature set as the tangent core (18 Oct; `OPTIMIZATION.md:42-49`).** Seven bands (adds 1–4 Hz) + session-centred < 1 Hz waveform, logistic regression C 0.1. Evidence: dl 56.2 vs v1 49.4; al 54.9 vs 51.5; better on 9 of 10 people (sign test p ≈ 0.02). Alternatives: channel means, unaligned filter bank, 6 vs 7 bands, LDA vs LR, C 0.01–1. C = 1 was better standalone on all four protocols on 19 Oct (57.3 / 69.9 / 56.4 / 60.0 vs 56.2 / 68.3 / 54.9 / 57.7; `OPTIMIZATION_R6.md:9-19`) and was not adopted because the in-ensemble change was within noise. Re-opened only on 6 Nov: the coverage audit states the core "was designed in one day on 18 Oct and never redesigned (3.6 % of saved experiments are linear)" (`REVIEW_5NOV.md:160-162`); a 48-config search then gave 69.3 vs 58.8 standalone (line 184).

**A5. Probability-averaging ensemble with fixed family weights (18 Oct, `8212015`; `OPTIMIZATION.md:127-153`).** v2 + 5 aligned + 5 raw ShallowConvNets in equal thirds: dl 63.0 vs 56.2; al 60.4 vs 54.9 (9 of 10 people). Weights "were not tuned" (v2 0.3–0.7 gave 58.5–62.0). Alternatives then: one-seed members, single streams. Re-opened and kept: geometric mean and per-member rebalancing (18 Oct), learned combiner (20 Oct), per-session weighting oracle (20 Oct), calibration (26 Oct), weight grids (19 Oct, 26 Oct), stacker and 8,197-candidate combiner re-tune (6 Nov), single-model replacements (7–8 Nov).

**A6. Sealed models train on all ten public people; warm-up board is not a target (18 Oct; `OPTIMIZATION.md:72-79`).** Basis: argument (sealed set includes later sessions of public people) plus the 99.2 % leakage replay. Honest warm-up builds use official train, later train + validation (26 Oct). Never re-opened.

**A7. Net training recipe (18 Oct).** 40 epochs, AdamW, one-cycle, no validation-based stopping, no augmentation. Alternatives: classic augmentation (18–19 Oct, not adopted), recipe sweep of the R9 families (27 Oct, flat). Rotation ε 0.05 added 29 Oct.

**A8. Class rebalancing (18 Oct α 0.5, k 5; 19 Oct W 30, α 1, k 5; `OPTIMIZATION_R3.md:33-56`, `OPTIMIZATION_R4.md:64-87`).** Evidence: five protocols, r3 ensemble, plateau over W 24–45; al +1.4, ss +3.0. Chosen "among the first five variants tried". Re-opened: Sinkhorn (20 Oct), exact counts oracle (28 Oct), α 1.15 adopted 5 Nov (a115), α 1.25 / 1.3 rejected (2, 5, 7 Nov).

**A9. FBCNet as the added net family (18 Oct; `OPTIMIZATION_R3.md:73-108`).** Architecture sweep, 8 nets × 3 seeds, aligned input, dl/ds, standalone. Only FBCNet and IFNet were then mixed. IFNet (+0.2) dropped partly because it "would need a second plain-torch re-implementation". Five-seed confirmation: al +0.8, ss −0.2.

**A10. Scherer 2015 in, then out (19 Oct in, 25 Oct out; `OPTIMIZATION_R5.md:21-26`, `SEALED_PLAN.md:43`).** In: user decision. Gain: dl +0.5 (five seeds; the three-seed +0.9 "was partly selection noise"), al +0.7, 7 of 10 people up. Out: user decision on the CC BY-NC-ND licence, not on evidence. The documents later note the licence was cleared on 11 Nov (outside this period).

**A11. Unaligned FBCNet as fifth family; equal family weights (19 Oct; `OPTIMIZATION_R6.md:43-68`).** dl +1.5, ds +1.9, al +1.3, ss +0.9, five seeds. "v2 at 0.15" evaluated better (al +1.0, ss +0.9) but was not the dl winner and was not taken.

**A12. dev_loso_all replaces dev_loso for selection (21 Oct, `b5930de`; `OPTIMIZATION_R8.md:87-92`).** Trigger: CBraMod gained +2.2 dl but +0.1 al. One case.

**A13. Learned-FIR FBCNet family, r8t4 (25 Oct).** Found through a bug (`1631413`). After the fix: dl 69.1 / ds 80.9 / dla 71.4; al +1.4, ss −0.7; reproduced on fresh seeds (`OPTIMIZATION_R8.md:386-403`). On 5 Nov and 7 Nov this family was the only one removable at no cost (`REVIEW_5NOV.md:119-121, 344`).

**A14. sel = mean(ds, dla) and the 57 % new / 43 % returning weighting (25–26 Oct; `OPTIMIZATION_R8.md:412-416, 438-439`).** The 57/43 split is an inference from the organisers' text. It was used to reject soft routing ("≈ −0.1") and to call the explorer arm "about neutral". The organisers withdrew the numbers on 5 Nov (`REVIEW_5NOV.md:19-20`).

**A15. r9: SincShallowNet + FBLightConvNet (27 Oct, `ea58271`).** Chosen from 32 configs on seeds 0–2 (sel +2.3), confirmed on fresh seeds 3–5 (+1.5); al +0.6, ss +1.1.

**A16. Multi-corpus pretraining for SincShallow only, corpus set m7 (28 Oct; `OPTIMIZATION_R10.md:59-115`).** Six seeds: sel +0.9, al +0.5, ss +0.7; later seed sets +0.2 / 0.0. The seven MI corpora stayed the pretraining set until 3 Nov ("barely explored", `REVIEW_2NOV.md:69`).

**A17. Rotation ε 0.05 (R14) and rotation-pretrained trunks (R15) (29 Oct).** Six-seed pooled: each adds about +0.2 sel (`STRATEGY_R16.md:8-16`). Both were adopted below the +0.5 rule that was written the next day. T9 on 1 Nov found R15's dev score optimistic by about 0.5 (`OVERNIGHT_2NOV.md:134-139`).

**A18. Adoption rule (30 Oct; `STRATEGY_R16.md:20-21`, `HANDOFF.md:70-71`).** ≥ +0.5 pooled over six seeds, both seed halves positive, paired per-person signs.

**A19. Session EM stage, R19 / R19e (30 Oct, `c90ac64`, `91bc0c4`).** Frozen config λ 0.5, n0 60, m0 20, shrink 0.7, start window 10; gaze view 32 values, then 16 event rates. Evidence: saved six-seed R15 probabilities, shift nulls, nested selection, seed halves; dla +0.93 / +1.67 (R19), further +1.26 / +0.53 (R19e); ss −0.18 for R19. Caveats in the record: the event definition "had already been seen on these posteriors"; ds is one person; the two critiques' nested estimates disagreed (+0.33 vs +1.35) and "no nested log is saved" (`GROUND_UP_REVIEW.md:164`).

**A20. The "What NOT to do" list (30 Oct; `GROUND_UP_REVIEW.md:390-400`, repeated in `HANDOFF.md:184-235`).** The largest single block of closures. It lists, among others, "no seen-participant routing in any form", "no Laplacian/CAR streams", "SessionNorm", "no pseudo-label … TTA". Three of these shipped a week later (section B flags).

**A21. Code freeze and stateless members only (31 Oct – 1 Nov; `SEALED_PLAN.md:369-371, 400-423`).** Reason: the warm-up phase was the only real-worker test bed. After it, new arms had to be `weights.npz` or a stateless TorchScript member. Kept on 2 Nov after one offline test (`OVERNIGHT_2NOV.md:443-468`). Lifted on 5 Nov for SessionBN members (+29 lines, `REVIEW_5NOV.md:141`).

**A22. Teacher design (1–2 Nov).** Frozen externally pretrained nets → head log-probs → logistic map → one member. Then m7r4 (C 0.01, w 4), then EEGNeX + IFNet (C 0.01, w 5, 12 seeds) = cand_nxif. Evidence: pre-registered replications on unseen pretraining seeds 6–11 and 12–23 (+0.65 to +0.72 sel over m7r4). Weakness stated on 2 and 5 Nov: P10 carries about 65 % of sel and 88 % of the teacher's +2.91; without P10 +0.99; P09 −5.9; sealed forecast "+1.0 ± 1.2, not +2.9" (`REVIEW_2NOV.md:38-43`, `REVIEW_5NOV.md:49-52`). The teacher was never isolated on the sealed set (`REVIEW_12NOV.md`, "uploads … never used as probes").

**A23. Person-equal J and selection on people halves / fresh bases (5 Nov; `REVIEW_5NOV.md:112-115`).** See report 1.

**A24. SessionBN members (5–6 Nov).** Arm D (FBCNet k 2 / k 5): pre-registered fresh-seed confirmation was a "narrow FAIL" (+0.30 J vs bar +0.40). REC (adds FBLight k 5, SincShallow k 20, raw FBCNet k 20): held-out seeds +0.56 J over a115 (SE 0.15, 6/6).

**A25. core1, Hilbert core, slot weights (6 Nov).** core1 `car_ov17_fs2_t0.1_le_k2` + LR C 10: standalone 69.3 vs 58.8; people-half selection +8.5 / +10.0. The pre-registered weight rule failed on a 0.0002 tie; "the user adopted fs2 at weight 1.5 after this tie" (`AUDIT_SBN.md` cand_rec_ts "Decision"). Hilbert core: +1.30 J vs packaged REC, "about +0.8 to +1.1" after selection.

**A26. TSEM (6 Nov).** PC128 of core1 features, λ 0.15, ρ 0.7, m0 20, shrink 0.7. Pre-registered confirmation on fresh member seeds: +1.71 J; four fresh bases mean +1.30. Its family "was chosen on all 10 people".

**A27. Routing v3 + gated early boost (7 Nov, `c4d600d`).** Routing: four fresh bases dJ +0.99 to +1.12, new 0.00, zero impostor routings, gain almost all P04 (+3.4 to +4.2; P10 0.0 to +0.7). GEB: +0.54 cross-fit, honest increment over plain boost +0.18 (p 0.09). "The 4 seen probe sessions were used across v1, v2 and v3"; fresh bases "vary member seeds only; the people are still the same 10".

**A28. Upload policy (7 Nov).** "Push to .83 first": no upload without a clearly better dev estimate. Runtime 44.8 of 60 min made new stages budget-bound.

---

## B. Closed or set-aside directions, by theme

Quality scale: SOLID (in the then-current full system, adequate controls), WEAK (standalone only, early pipeline, very few people, one configuration, or confounded), ARGUMENT (never run). "Reversed" marks a direction that later worked.

### B1. Features and input representation

| # | Idea | Date, source | How tested | Result and reason | Quality |
|---|---|---|---|---|---|
| F1 | v2 knobs: prior k, C, LDA head, train-time causal recentring, dispersion stretch | 18 Oct, `OPTIMIZATION.md:80-82`; C again 19 Oct | v2 standalone, deterministic, dl/ds | ≤ 1.5 points; "simplest kept" | WEAK. **Reversed 6 Nov**: shrinkage (+6), sub-windows (+3.3), CAR, higher C, smaller k made core1 |
| F2 | Separate prior for slow features; drop per-window DC | 18 Oct, `OPTIMIZATION.md:195-200` | v2 standalone | ≤ 0.8 | WEAK, harmless |
| F3 | xDAWN / ERP-covariance member | 20 Oct, `OPTIMIZATION_R7.md:122-136` | alone 45–47 dl; in r6 at two weights | 66.5 / 67.8 vs 67.8 | WEAK (one design, r6, never retested) |
| F4 | Frozen REVE embeddings + LR | 19 Oct, `OPTIMIZATION_R4.md:44-51` | alone 44.7 dl; member at 0.1 / 0.2 in r3 | 64.6 / 64.1 vs 65.1 | WEAK (probe only) |
| F5 | 30–58 Hz power (EMG) | 30 Oct, `OPTIMIZATION_R10.md:487` | alone (dla 33.6); mixed into R15 with shift null | inside the null | SOLID |
| F6 | v2 slow-block scaling | 30 Oct, line 490 | inside R15, six seeds | ds −0.73, sel +0.06 | SOLID |
| F7 | Band-limited (4–40 Hz) EA reference for nets | 30 Oct, lines 460-466 | 3 families retrained, seeds 0–2, alone and swapped into R15 | alone −3 to −9; R15 78.2–79.3 vs 79.6 | SOLID (3 seeds; the linear proxy that motivated it "did not transfer") |
| F8 | Per-session alpha-peak band shift | 29 Oct, lines 291-294 | "closed, no run" | peak judged unstable | ARGUMENT |
| F9 | Re-referenced views (CAR, Laplacian) | 30 Oct, `GROUND_UP_REVIEW.md:240, 397` | none: "absorbed by learned spatial filters; EA(Lx) = rotation" | dropped | ARGUMENT. **Reversed**: CAR is in both shipped cores (6 Nov). `LEDGER.md:1127-1129` calls the argument the one that "wrongly closed CAR". Laplacian was refuted on evidence only on 12 Nov |
| F10 | CAR for deep members | 5 Nov, `REVIEW_5NOV.md:117-118` | seeds 0–2, each member swapped for its CAR twin, J | −0.29 | SOLID for nets. The side claim "CAR lifts v2 by +1.85" was confounded: CAR reached only the slow block because "the feature cache ignores the input" (line 188-189) |
| F11 | Scalar per-session RMS normalisation | 28 Oct, `OPTIMIZATION_R8.md:567-573` | seeds 0–4, swap and add in r9 | 77.56–77.72 vs 77.79 | SOLID for r9 |
| F12 | Time-segment net members (first 2 s, last 2 s, first 3 s) | 27–28 Oct, lines 548-554 | seeds 0–2, added to r9 | dla never up by more than 0.4 | SOLID for nets. Related later gain: half-window covariances in core1 (+3.3) |
| F13 | Task-period crops as training rows | 20 Oct (confounded by the FIR bug), 25 Oct retest, 7 Nov for the core | 25 Oct: 3 seeds in r6nf; 7 Nov: nested and fresh bases | 67.28 / 80.17 / 69.59 vs 67.18 / 80.49 / 69.37; core −0.47 | SOLID after the retest |
| F14 | New label views: cardiac, respiration, band power, ERP / alpha lateralisation, saccade direction, feedback period | 30 Oct, lines 574-626 | saved six-seed R19e probabilities, alone and as EM view, shift nulls | views ≥ 45 % repeat the ensemble's errors; independent ones 36–39 % | SOLID for R19e (scratch logs only) |
| F15 | Mirror augmentation; direction-consistency gating | 30 Oct, lines 624-625 | band-power LR / tangent space standalone at 45–51 % | no gain | WEAK (weak linear proxy) |
| F16 | Spectral normalisation (CMMN) | untested 30 Oct; run 6–7 Nov | two bases after TSEM | −0.85 / −0.99 | SOLID |
| F17 | Core variants: delay embedding, more bands, 3 sub-windows, MDM, OAS/LW, AIRM, third core, per-session z-scoring | 6–7 Nov, `REVIEW_5NOV.md:186-187, 208-212, 457-460` | core search + stack, nested | none beat; z-scoring overlaps TSEM | SOLID |
| F18 | Codex auxiliary heads (channel power, rank covariance, lag energy, coherency, motifs, bispectrum, …) | 8–10 Nov, `PROTEUS_SOURCE_10NOV.md`, `PROTEUS_CHANNEL_POWER_BLEND.md` | 10 % vote on a weaker "overlap" base; 7 source people, session 1 only; bootstrap seeds | best +0.7 to +0.8; none passed the +0.8 gate; the source-fitted blend weight was 0 % | WEAK for our chain (different, weaker base). Closed in the full chain only on 11 Nov (−0.21) |

### B2. Model classes

| # | Idea | Date, source | How tested | Result | Quality |
|---|---|---|---|---|---|
| M1 | EEGNet, TIDNet, EEGNeX, CTNet, Conformer, ATCNet, Deep4 | 18 Oct, `OPTIMIZATION_R3.md:73-87` | standalone only, 3 seeds, one recipe, dl/ds | 41–53 dl | WEAK (standalone accuracy at 41–53 %; never in a mix) |
| M2 | Unaligned IFNet / TIDNet / EEGNeX as members | 20 Oct, `OPTIMIZATION_R7.md:12-35` | 5 seeds in r6; EEGNeX +1.2 dl on every seed | al ±0, ss −0.5 → "selection noise" | WEAK-to-SOLID: full r6 test, but the rejection used the "evaluation-only" protocols. **Partly reversed**: EEGNeX + IFNet became the best teacher (2 Nov); EEGNeX in the trainable-FIR slot of a115 gave J +2.17 vs +1.65 and was not followed (`REVIEW_5NOV.md:123-124`) |
| M3 | 16 more braindecode architectures (R9) | 27 Oct | 3 seeds, each as extra family in r8t4 | two adopted, rest ≤ baseline on dla | SOLID |
| M4 | CBraMod | 20–21 Oct, `OPTIMIZATION_R8.md:18-92` | up to 6 seeds, 4 protocols, paired seed test | dl +2.2, al +0.1, ss −2.2 | SOLID for r6nf |
| M5 | LaBraM, fine-tuned REVE | 20–22 Oct | 2–3 seeds, two recipes each, in r6nf | −1.5 dl; no gain | SOLID-ish. One REVE comparison used the bug-affected FBCNet families |
| M6 | Classic augmentation | 18–19 Oct | 3 seeds, in r3 mix | aligned ShallowConvNet +3.3 alone, mix ±0.5 | SOLID for r3; early pipeline |
| M7 | Balanced loss, GroupDRO | 25–26 Oct | 5 seeds, in r8t4 | 73.9–76.6 vs 76.2 | SOLID |
| M8 | Weight averaging, same-class mixing | 28 Oct | 8 families, 5 seeds, in R10 | within ±0.2 | SOLID for plain nets |
| M9 | MixStyle, participant-adversarial | 30 Oct, `OPTIMIZATION_R10.md:401-418` | seeds 0–2, in R15 | adversarial hurts; MixStyle "neutral" | Adversarial SOLID. **MixStyle WEAK**: `PREDICTIONS_12NOV.md:866` found it never ran on ShallowConvNet (output identical on and off), so two of six rows are baseline reruns |
| M10 | Pretraining ShallowConvNet / FBCNet / FBLight trunks (R10, R16) | 28–30 Oct | 3–6 seeds in r9 / R15 | lose about 1 on dla, or +0.3 below noise | SOLID |
| M11 | Unified rest-anchored pretraining (R17) | 30 Oct | 2–3 seeds, alone and in R15 | 77.8–78.9 vs 79.6 | SOLID-ish |
| M12 | Rotated copies for v2 | 29 Oct | alone and in R14 | 78.3–78.9 vs 79.3 | SOLID |
| M13 | ROCKET member, co-training | 1 Nov, `OVERNIGHT_2NOV.md:94-125` | in R19e; co-training with pre-registered control and replication | negative; co-training equals extra sinc weight | SOLID (ROCKET lost two seeds to a node fault) |
| M14 | Fine-tuned EEGNeX / IFNet members | 2 Nov, wave 7 | 6 seeds, with and without teacher, pre-registered | every member lowers sel | SOLID on sel |
| M15 | SPD network with session SPD batch norm | 30 Oct "open but unlikely" (argument); run 6 Nov | own implementation, standalone and stack | J 54 vs 68.5; negative | SOLID (one implementation) |
| M16 | Learned session-conditioned network | 6 Nov | several seeds; meta-trained on 659 external sessions | −1.8 to −2.7; parity at best | SOLID |
| M17 | Single deep model; lean ensembles | 7–8 Nov | through the full route3geb chain | −2.8 to −12 J | SOLID |

### B3. Training data

| # | Idea | Date | How tested | Result | Quality |
|---|---|---|---|---|---|
| D1 | Dreyer 2023 in the 3-class loss | 19 Oct, `OPTIMIZATION_R5.md:93-101` | one seed, all four members, r3 | every member collapses (−5 to −11) | WEAK by count; large effect and a clear mechanism. Later used for pretraining |
| D2 | Scherer for v2 and aligned ShallowConvNet | 19 Oct | v2 deterministic; nets 3 seeds | "every variant lost on dev_session" (4 cells, P10 only) | WEAK |
| D3 | Shin 2017 A+B as labelled data (R13) | 29 Oct | all families, seeds 0–2, swapped into R10 | all lose on dla | SOLID; the block-confound explanation is argument |
| D4 | Mental-task corpora in pretraining (ptmt, ptm9) | 28 Oct | seeds 0–2 | below m7 | SOLID-ish |
| D5 | Teacher: arithmetic corpora (wave 9), more MI corpora (10m), unlabelled PROTEUS adaptation (10u), M1-re-referenced corpora (wave 11) | 2–5 Nov | seeds 0–5, pre-registered gates on sel | −0.5 to +0.2; 10m raised teacher-alone dla by 5 with no ensemble gain | SOLID on sel, but sel is P10-dominated |
| D6 | Segment recombination in teacher pretraining | 2 Nov, wave 8 | seeds 0–5 | −0.48 | **WEAK, confounded**: reached only the first corpora of each batch (`PREDICTIONS_12NOV.md:867`) |
| D7 | Participant bagging; label-noise reweighting | 30 Oct, `GROUND_UP_REVIEW.md:251-252` | bagging "falsified by a proxy" (ensemble swaps); label noise by a count (2.8 % low-p windows) | — | WEAK / ARGUMENT |
| D8 | More seeds per family | 27 Oct, 1 Nov | 10 seeds; seeds 6–11 swap | no gain; −0.83 | SOLID |

### B4. Test-time adaptation

| # | Idea | Date | How tested | Result | Quality |
|---|---|---|---|---|---|
| T1 | Within-session self-training | 19 Oct, `OPTIMIZATION_R4.md:31-42` | one design (LDA on 64 PCA dims of v2 features, r3 pseudo-labels), 3 mixing weights, saved probabilities | 61.9–65.4 vs 65.1 | WEAK. **Reversed** (see T5) |
| T2 | Selective prototype TTA (T3A) | 20 Oct, `OPTIMIZATION_R8.md:16` | 48 settings, v2 features, r6nf | 62.6–66.9 vs 67.2 | WEAK (v2 features at 56 %). **Reversed** (T5) |
| T3 | Test-time batch-norm adaptation | 19–20 Oct, `OPTIMIZATION_R7.md:37-65` | 3 seeds; FBCNet alone 59.1 → 64.0; in rebalanced mix within noise | "redundant with rebalancing" | SOLID for test-time-only AdaBN on r6. On 30 Oct it was extended **by argument** to trained-in session normalisation (`GROUND_UP_REVIEW.md:243`: "Dropped"; line 203 dismisses the BEETL trick). **Reversed 5 Nov**: SessionBN FBCNet 59.1 → 66.0 alone, shipped 6 Nov |
| T4 | Gradient TTA (T-TIME) | 28 Oct | 2 families, 3 seeds, 5 strengths, paired control, in R10 | every variant loses after rebalancing | SOLID |
| T5 | Label-free per-session refits on the ensemble's own labels (v2 MAP, adaptive LDA, discriminant-subspace EM, head refit) | 30 Oct round 3, `OPTIMIZATION_R10.md:540-566` | true-label oracles up to +5.4 dla; label-free on saved R19e probabilities with nulls | label-free ≈ 0; "exhausted" | SOLID for R19e and v2-era features. **Reversed 6 Nov**: TSEM (confident-seed class templates in the new core's PC128) passed a pre-registered confirmation (+1.71 J). `REVIEW_5NOV.md:204`: "weak base +0.3, strong +1.4" |
| T6 | Label-shift-aware reference | 20 Oct | true-label oracle, v2 only | +0.05 dl | WEAK (v2 standalone), but an oracle; re-closed 6 Nov in the stack (−1.35) |
| T7 | Time-shift test-time augmentation | 27 Oct | 5 paired seeds, r8t4 | within noise | SOLID |
| T8 | Per-session logit bias / affine map | 28 Oct | oracle, R10 seeds 0–2 | ≤ +2.6 in-sample | SOLID as a bound |
| T9 | EM variants (population prior, per-interface, per-run-type, λ, recency, clipping) | 30 Oct | six-seed halves on R19e | flat or negative; n0 30 gave +0.4, "below the adoption rule" | SOLID |
| T10 | EA reference trace clip (R20) | 31 Oct | all EA families retrained, six seeds, pre-registered | dla +0.08, ds −0.13 | SOLID |
| T11 | In-session teacher adaptation | 2 Nov | offline, seeds 0–11 | −0.3 to −1.5 sel | SOLID; used to keep the code freeze |

### B5. Sequence and protocol structure

| # | Idea | Date | How tested | Result | Quality |
|---|---|---|---|---|---|
| S1 | Trial-order prior | 19–20 Oct, `OPTIMIZATION_R4.md:66-68, 85-87`, `OPTIMIZATION_R8.md:13` | transition counts over 84 runs | "random permutation … nothing beyond rebalancing" | WEAK and wrong in fact: Graz runs follow a weak WORD → SUB → MI cycle (`docs/archive/README.md:29-30`). Bounded properly on 30 Oct (position alone 54 %; ≤ +0.3 with oracle run boundaries) |
| S2 | Sinkhorn balancing over the last W windows | 20 Oct, `OPTIMIZATION_R7.md:129` | 18 settings, r6 probabilities | dl 66.4–68.2 vs 67.8; ds 79.9–82.9 vs 80.5 | SOLID for r6; the ds gain was set aside as 4-cell noise; no later retest recorded |
| S3 | Exact per-run counts | 28 Oct, `OPTIMIZATION_R10.md:207` | oracle run boundaries, R10 | al +0.38, **ss +1.81** | SOLID as a bound. "Run structure is spent" rests on the al figure |
| S4 | Per-interface alignment or stages | 29 Oct (v2 only, oracle); 30 Oct (EM); 2 Nov (agent, fold-specific); 6 Nov (oracle ≤ +0.10) | oracle interface | no gain | First test WEAK, later ones SOLID |
| S5 | Session clustering | 25 Oct (v2 PCA, oracle map 0.37–0.39); 30 Oct (ocular / slow views) | standalone | far below the ensemble | WEAK then SOLID-ish |
| S6 | Position schedules; interface-switch resets | 30 Oct | position schedule −0.1 to −1.1 (scratch `pos.py`) | — | Schedule: scratch only. Resets: listed closed in `HANDOFF.md:201-202`; test not recorded |

### B6. Person-level methods

| # | Idea | Date | How tested | Result | Quality |
|---|---|---|---|---|---|
| P1 | Identify the returning person, route to a personal model | 19 Oct, `OPTIMIZATION_R4.md:17-29` | one fingerprint (7-band log-covariance), 2 returning people, analysis only | P04 placed nearest P09; "routing was not built" | WEAK |
| P1b | Soft routing with personal v2 models | 25 Oct, `OPTIMIZATION_R8.md:344-365` | PCA + multinomial LR identity, λ fixed before evaluation, in r6nf | ss +1.0, al −1.0; rejected under the assumed 57/43 mix | SOLID for that identity feature |
| P1c | Covariance / PSD distances; gaze library | 28 Oct, 30 Oct | distance overlap; "0 of 4 correct at window 30 even with oracle labels" (scratch) | "blocked"; "no routing in any form" | WEAK (analysis, 4 probe sessions) |
| P1d | Seen-person specialist | 5 Nov | identification ≤ 10 of 12 sessions, cannot reject new people | "closed again" | SOLID-ish |
| — | **Reversal** | 7 Nov | class-direction fingerprints on pseudo-labelled contrasts in PC256; pre-registered on four fresh bases | dJ about +1.0, zero impostor routings; shipped; sealed +0.43 with GEB | Routing was closed four times on spectral identity features and worked with a class-direction feature. On dev the gain is P04's |
| P2 | Eigen-participant MAP | 30 Oct | labelled oracle in v2 space (scratch) | −0.8 to +0.3 | WEAK (v2 space) |
| P3 | Per-person member-family routing | 30 Oct | cross-fitted | −0.2 to −2.6 dla | SOLID-ish |
| P4 | Pre-cue (privileged-information) teacher | 30 Oct | "weak, possibly leaky teacher" | dropped | ARGUMENT |
| P5 | Personal deep models in the routed tail, early routing, full MAP, Hilbert personal heads | 7 Nov | pre-registered on fresh bases | below gates; early routing fails the P10 bar | SOLID |

### B7. Combination

| # | Idea | Date | How tested | Result | Quality |
|---|---|---|---|---|---|
| C1 | Geometric mean; rebalancing per member | 18 Oct, `OPTIMIZATION_R3.md:110-112` | r3, dl only | 65.5 / 66.0 vs 65.1, "within noise" | WEAK (positive, dl only, no later retest recorded) |
| C2 | Learned combiner / stacking | 20 Oct; 6 Nov | nested leave-one-person-out | 64.8–66.4 vs 67.8; −4.8 | SOLID twice |
| C3 | Per-session member weighting | 20 Oct; 6 Nov | oracle selection | below the fixed mix; +0.02 | SOLID |
| C4 | Temperature calibration | 26 Oct | per-family temperature, r8t4 | −0.26 dl / +0.90 ds / −0.76 dla | SOLID |
| C5 | Member confusion model (Dawid–Skene) | 30 Oct | true-label oracle +0.93; label-free "not run" | closed | ARGUMENT for the label-free version (refuted by a run on 11 Nov) |
| C6 | Teacher fusion: late fusion, recipe mixes, pooling rules, map edits, agent-tuned parameters | 2–5 Nov | seeds 0–11, hidden people and seeds | none transfers to held-out people | SOLID |
| C7 | Post-TSEM re-tune of chain constants | 7 Nov | 243 configs, nested + fresh bases | nested −0.21 | SOLID |

### B8. Engineering decisions that narrowed the search
- Runtime self-limit ("we keep ≤ 30 min") and a 10 MB package: `REVIEW_5NOV.md:157-159` lists these with the code freeze and stateless members as "constraints we imposed on ourselves, not the rules" (835 of 3,600 s used on upload 1).
- IFNet not added on 18 Oct partly for porting cost; EEGNeX on 20 Oct likewise.
- Bugs that confounded verdicts: trainable FIR banks (20–22 Oct; invalidated FBCNet variants, r8v, the first crop test); pretraining without rotation (2 Nov; two screens invalid, not used); CAR reaching only the slow block (5 Nov); MixStyle and segment recombination no-ops (found 12 Nov).

---

## C. Ideas mentioned with no recorded test (up to 10 Nov)

1. Resting-state runs as alignment priors (`OPTIMIZATION.md:114`; "proposed R1, never tested", `STRATEGY_REVIEW.md:57`).
2. Label-free mismatch detector that down-weights the slow block for people like P04 (`OPTIMIZATION.md:261-263`). Only a global scaling was tested (F6).
3. EEGPT ("downloaded", 20 Oct handoff) and EEGDM (`EXPERIMENTS.md:25-30`). ST-EEGFormer was proposed 20 Oct and first run 11 Nov.
4. Foundation-model recipes: LoRA, Mixup, linear-probe-then-fine-tune with a shared head, seed weight soups ("Seed weight averaging was not tried", `OPTIMIZATION_R8.md:62-63`).
5. SPD-bias information maximisation (SPDIM) from `STRATEGY_REVIEW.md:107-109`.
6. Larger models or test-time ensembles using the GPU budget (`STRATEGY_REVIEW.md:54`).
7. Self-supervised pretraining or test-time training on unlabelled PROTEUS EEG: rest runs, inter-trial periods, trial tails, raw 500 Hz (`STRATEGY_R16.md:60`, `GROUND_UP_REVIEW.md:255, 439`, `REVIEW_2NOV.md:140-142`, `REVIEW_5NOV.md:158-159`). Wave 10u used unlabelled task windows through an adversary only.
8. Hierarchical teacher with an "MI vs cognitive" head (`REVIEW_2NOV.md:143-144`).
9. Projecting the Rathee 2021 MEG three-task corpus to EEG (`REVIEW_2NOV.md:80`).
10. Learned per-window TTA (BFT) and in-graph TTA inside the teacher (`REVIEW_2NOV.md:89, 99`); the 5 Nov agent row says only "≤ base or refuted on prior evidence".
11. Beta lateralisation over fronto-temporal pairs; aperiodic normalisation (`REVIEW_2NOV.md:73`).
12. The three "cheap diagnostics" of `GROUND_UP_REVIEW.md:363-367`: oracle whole-session reference gate, v2 split into covariance-only and slow-only, and the amplitude anomaly (honest warm-up score rises from 53.6 to 55.0 when input is scaled ×0.25). Results not recorded.
13. Per-band EA inside FBCNet (conditional follow-up, never started).
14. Routing: a null gallery or minimum-contrast check ("not built"), and a per-person trust weight from the public leave-one-out deficit ("natural next step", `REVIEW_5NOV.md:306-307`).
15. A second TSEM stage on PCs 129–256 ("possible late variant"); α 1.3 as an upload variant.
16. EEGNeX in the trainable-FIR slot (J +2.17 vs +1.65 on about 70 configs): no confirmation recorded.
17. FBCSP + LDA and TS ensembles, listed as never tried in `REVIEW_5NOV.md:161-162`.
18. Sealed probes that were staged and never uploaded: cal_v2 (v2 + rebalancing calibration probe), a115, cand_nxif_emx, cand_m7r4, R19e as the no-teacher arm, cand_rec_ts2 as the paired TSEM ablation, tsem256, routing v1 / v3 / v3 + ungated boost, the "zero feet heads" teacher edit.
19. A lean weights-only rebuild (29 min instead of 41.9), "not built or replayed" by 10 Nov.
20. Cross-fitted honest version of wave 10u (not run because the upper bound failed).

---

## Report 1. Dev metric and validation protocol over time

| Date | Selection measure | People it depends on |
|---|---|---|
| 18 Oct | `dev_loso` (main), `dev_session` (probe); al, ss, official "evaluation only" | 8 dev people; P10 is 6 of 20 dl cells and all 4 ds cells |
| 20 Oct | Rule: dl > +1 on a plateau, ds / al / ss not down (`STRATEGY_REVIEW.md:74-79`) | al and ss enter decisions de facto (EEGNeX, CBraMod, soft routing) |
| 21 Oct | `dev_loso_all` replaces dl | same 8, with P04 / P09 in training |
| 26 Oct | sel = mean(ds, dla); sealed effect weighted 0.57 new / 0.43 returning | P10 ≈ 65 % of sel (`SEALED_PLAN.md:374`) |
| 30 Oct | six seeds pooled, ≥ +0.5, both halves, per-person signs, shift nulls | same; seed sets 0–2 and 3–5 differ by 1.4 |
| 31 Oct | Review: P04 / P09 and ss "are no longer clean holdouts" (`SEALED_PLAN.md:381-382`) | — |
| 1–3 Nov | Teacher screens: sel Δ over R19e, replication on unseen pretraining seeds; from 2 Nov also "without P10" and worst person | P10 gives 88 % of the teacher's gain |
| 5 Nov | J = 0.5 × person-equal al (10 people) + 0.5 × person-equal ss (P04, P10); selection on people halves, confirmation on "fresh bases" | P04 and P10 each ≈ 30 % of J; fresh bases vary member seeds, not people |
| 6–7 Nov | Reading rule: big steps transfer ≈ 1:1, small stacked gains ≈ 0.6× | one calibration pair of uploads each |

Later verdicts on this: `docs/archive/README.md:23-24` ("Dev weights P04 and P10 equally; on the sealed set P04 probably has 3 of about 46 returning sessions. Every returning-person gain so far is P04's"); `REVIEW_12NOV.md` §2 (gates of +0.3 to +0.5 sit inside the ruler's noise).

## Report 2. Early weak-pipeline results that closed a direction

Closed on a 45–67 % pipeline and later reversed:
- Pseudo-label / prototype adaptation (T1 at 65 %, T2 on v2 features) → TSEM, 6 Nov.
- Test-time BN on r6 (T3), then session normalisation by argument → SessionBN, 5–6 Nov.
- Person identification on one spectral fingerprint (P1, 19 Oct) → routing v3, 7 Nov.
- v2 knobs at 56 % standalone (F1) → core1, 6 Nov.
- Re-referencing by argument (F9) → CAR in the cores, 6 Nov.

Closed on an early pipeline with no later retest recorded up to 10 Nov:
- Geometric mean and per-member rebalancing (C1, r3).
- Sinkhorn balancing (S2, r6).
- xDAWN / ERP member (F3, r6); frozen REVE (F4, r3).
- Seven architectures dropped on standalone accuracy alone (M1). `docs/archive/STATE_11NOV.md:113` lists SessionBN versions of TIDNet, CTNet and EEGConformer as never tried.
- Classic augmentation, weight averaging, same-class mixing, MixStyle on the later SessionBN nets (`PREDICTIONS_12NOV.md:852`: mixing and frequency scaling "never ran on SessionBN nets").
- Dreyer in the 3-class loss (one seed); Scherer for v2 (decided on 4 cells of P10).
- Label-shift and per-interface alignment oracles on v2 alone; session clustering on v2 features; eigen-participant MAP in v2 space; mirror augmentation and consistency gating on 45–51 % linear proxies.
- Alpha-peak band shift (never run).

## Report 3. Upload history

Warm-up (real worker, P04 / P09 test; scores are not evidence where trained on those people):

| Date | ID | Package | Score | What it taught |
|---|---|---|---|---|
| 18 Oct | 1000481 | v2 + 10 ShallowConvNets, official train | 0.4411 | worker equals local replay bit for bit |
| 18 Oct | 1000482 | v1 | 0.4973 | v1 beats v2 on the P04-heavy board; board and sealed goal disagree |
| 18 Oct | 1000615 | r3, official train | 0.49 | FBCNet port and rebalancing run on the worker |
| 26 Oct | 1010506 / 1010508 | r6nfg / r8t4g sealed ZIPs | 0.9973 / 1.0000 | GPU path and runtime (24–26 s) |
| 26 Oct | 1010748 | r8t4, train + validation (honest) | 0.5304 | honest level; P04 0.493 |
| 27 Oct | 1014114 | r9, train + validation (honest) | 0.5358 | below the plausibly honest cluster at 0.55–0.59 |
| 29 Oct – 1 Nov | 1018364, 1021075, 1021330, 1024097, 1024447, 1026185 | R10, R19, R19e, guarded R19e, TorchScript proof, teacher package | 0.983–0.997 | code paths proven before the freeze |

Sealed (`docs/SEALED_UPLOADS.md:76-204`):

| Day | ID | Package | Score (rescored 10–11 Nov) | Runtime | What it taught |
|---|---|---|---|---|---|
| 5 Nov | 1034698 | cand_nxif | 0.7795 (0.7723) | 835 s | Rank 9 of 46, 3–5 points under the leaders. Triggered the approach tournament, the lifting of the code freeze and the core redesign. The scheduled 00:17 check-in had not uploaded |
| 6 Nov | 1038262 | cand_rec_ts2_tsem | 0.8197 (0.8080) | 2,497 s | +4.02 for dev J +3.7 to +3.9: large steps transfer about 1:1. The paired ablation upload was skipped |
| 7 Nov | 1040847 | route3geb | 0.8240 (0.8135) | 2,687 s (3,732 s on rerun) | +0.43 for an honest dev +0.75: small stacked gains transfer about 0.6×. 4th at the time |

No upload isolated the teacher, SessionBN, the cores, TSEM, routing or the early boost separately. The rescoring changed the three packages by −0.72, −1.17 and −1.05 points; the cause is recorded as unknown.

## Later documents that call an earlier closure wrong
- `docs/REVIEW_12NOV.md:18-20`: "Three of the winners (session normalisation, re-referencing, routing) had been closed by argument a week before they shipped."
- `docs/LEDGER.md:1127-1129, 1149`: the re-referencing argument "wrongly closed CAR"; for the Laplacian the closure was "wrong in kind but right in outcome".
- `docs/PREDICTIONS_12NOV.md:848-869`: re-read table (training tricks tested only on plain nets; band shift never run; MixStyle and segment-recombination no-ops; core variants entered the chain as posteriors only).
- `docs/archive/README.md:19-39`: list of assumptions in the archive now known to be wrong.

My reading of the routing case differs slightly from the 12 Nov sentence: routing was measured four times before it shipped, but always with spectral identity features; only the 30 Oct "in any form" extension was argument.

## Files
- `<repo>/docs/archive/` (OPTIMIZATION.md, OPTIMIZATION_R3–R8.md, OPTIMIZATION_R10.md, STRATEGY_REVIEW.md, STRATEGY_R16.md, GROUND_UP_REVIEW.md, HANDOFF.md, SEALED_PLAN.md, REVIEW_2NOV.md, OVERNIGHT_2NOV.md, WAVE10_3NOV.md, STATE_7NOV.md, PROTEUS_*.md, README.md)
- `<repo>/docs/REVIEW_5NOV.md`
- `<repo>/docs/SEALED_UPLOADS.md`
- `<repo>/docs/COMPETITION.md`
- `<repo>/docs/AUDIT_SBN.md`
- `<repo>/docs/REVIEW_12NOV.md`, `docs/PREDICTIONS_12NOV.md` (lines 848-869), `docs/LEDGER.md` (lines 1120-1152)

Not covered: `docs/archive/STATE_11NOV.md` and everything from 11 Nov on, except the cross-references above. `docs/AUDIT_SBN.md` was skimmed (recipe introductions only), as asked.
