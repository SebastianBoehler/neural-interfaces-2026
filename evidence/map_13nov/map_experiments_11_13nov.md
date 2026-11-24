# Inventory of experiments from 11 Nov 2026 onwards (ledger §1–§119), by solution axis

Scope and caveats. I read in full: `docs/STATE.md`, `docs/REVIEW_12NOV.md`, `docs/LEDGER.md` (1,682 lines, ends at §119, unchanged while I read; last commit 55b6ee4), `docs/PREDICTIONS_11NOV.md` (§1–§33), `docs/PREDICTIONS_12NOV.md` (§34–§65), `docs/SEALED_UPLOADS.md`, `docs/V52_BUILD.md`, `studies/12nov/README.md`, the file list of `studies/11nov/` (no README there) and the docstrings of `studies/13nov/*.py`. Nothing was edited or run. Section numbers are global (§1–§33 = 11 Nov, §34–§65 = night of 11–12 Nov, §66–§119 = 12–13 Nov). All 119 sections are placed below; multi-arm sections are listed with their arms. Numbers are dev J unless "new" or "seen" is stated. I did not read pre-11-Nov documents, so "never touched" means "not in these documents".

Tags: CHAIN = inside the full chain, all six bases (or pick on two, confirm on four fresh), with a control. CHAIN-PARTIAL = in a chain but only B2_pool/B2_s345, one configuration, a proxy chain, posterior-only swap, or no control. ALONE = standalone or core-level only. ORACLE = uses true labels or true metadata. ARGUMENT = closed without a run.

Three path facts that qualify many tags:
- Many 12–13 Nov "six bases" runs use the "grid chain without routing" (§72, §90, §92, §103b, §109; §117 "grid chain"), not the shipped v4.1/v5.1 chain.
- Every new core variant entered as posteriors only; TSEM's view and the routing features stayed the shipped core1's (§52 audit; restated in §99).
- All TTT-family pseudo-labels come from the v3 chain on B2_pool (§98 note on Leonardo).

---

## 1. Table by axis

### (a) Window-level features and representations

| § | What | Result | Verdict | Tag |
|---|---|---|---|---|
| 13 | Ten hand edits of core1's head (shrinkage, block gains, smoothing, tying, pruning, soup), E9 learned block gains, Weco search (56 edits) | best chain −0.08; Weco best +0.11 on its half, −0.11 on the hidden half | closed | CHAIN-PARTIAL (mean of two bases, posterior only) |
| 20 | Forgetting reference for core1's recentring | alone new +0.05, chain −0.04 / −0.01 | refuted; §52 relabels it "closed for the posterior only" | CHAIN-PARTIAL |
| 25 | Amplitude-envelope coupling core | 50.4 alone; −0.6 / −1.0 / −2.5 as extra member | refuted | CHAIN-PARTIAL (extra member, fixed weights, old stages) |
| 26 | Cross-frequency coupling core | 58.4 alone; −0.6 to −2.8 | refuted | CHAIN-PARTIAL (same) |
| 31 | Temperature-calibrated extra cores | CFC −0.36 instead of −0.60 | still negative | CHAIN-PARTIAL (exploratory) |
| 5 A09 | True-label view oracles: common mode, physiology | ≤ +0.11 | ceilings; label-free view arms moot | ORACLE |
| 5 A10, A11, A16, A21, A28 | Common-mode, slow/EMG, task-physiology views as stage or inside TSEM | fresh −0.20 to +0.07; pick minus null ≤ +0.19 | refuted | CHAIN |
| 5 A25 | core1 refitted with left-right mirrored MI | fresh −0.25; control −0.03 | refuted | CHAIN |
| 33, 34 | Slow potentials and gaze within and across people | 57–62 % within, 47–49 % across | no shortcut | ALONE |
| 36 | REPCEIL: within-session supervised ceilings in session-own spaces | four-way fusion 81.4–82.4 against 74.9 in population PCs | "the 76 % level was core1's PC space" | ORACLE (non-causal, leave one run out) |
| 49 (side) | core1 44k features reduced to 512 PCs | −4.0 | diagnostic | ALONE |
| 54, 54b | TSEM in 64 / 256 / 512 / 1,024 components; 1,024 for new people only | k 64 −1.3; k 1,024 new +0.61 / +0.59 on two bases, +0.08 on four fresh | refuted | CHAIN |
| 58 probe | Eye-movement views A–D, LDA across people | 44.2–51.1 | signal exists | ALONE |
| 58 EMVIEW | Extended SessionEM gaze view | −0.83 to −2.77 | refuted "at its shipped settings" | CHAIN-PARTIAL (two bases, stage strength not re-chosen) |
| 58 COREFUSE | 104 eye features inside core1 | −0.45 / −2.73 / −3.53 at core level | refuted | ALONE (never in the chain) |
| 58 OCUNET, 58b | Learned eye-movement net: member, final stage, SessionEM view | 53.2 alone; member −0.01 to −1.62; stage −0.7 to −6.9; view +0.21 / +0.33, fresh +0.29, six-base +0.28 with none negative | dropped at +0.29 against a +0.30 mark | CHAIN |
| 71 | Frozen foundation embeddings (CBraMod, LaBraM, REVE, ST-EEGFormer) as TSEM view | probes 35–44 % (random-init REVE 42.8); second TSEM −0.50 to +0.39; appended −0.56 to +0.22; as member −1.4 to −3.9 | refuted | CHAIN-PARTIAL (two bases, random-init control) |
| 72, 73 | Nets' classifier inputs as a second TSEM view | choosing seeds +0.55 / +0.49; fresh nets +0.20 [−0.25, +0.68] on B2_pool; six-base +0.07 (random null −0.07); fbraw −0.16 | refuted | CHAIN |
| 74 | Discriminative views from core1's 256 PCs: pooled LDA, PLS-DA 16, per-person LDA, random | PLS-DA +0.45 to +0.60; random null +0.25 / +0.28 | "edge of resolution"; never taken to six bases | CHAIN-PARTIAL (two bases) |
| 77 | BANDMAP: per band and family, own-session against from-others | P04 own best CSP 4–8 Hz 60.6, slow 59.3; from others best 49; 0 of 19 bands ≥ 50 | diagnostic; "closed in feature space" | ALONE |
| 86 | Five sensor views consulted on the chain's hard third | views alone 36.9–52.4; chosen arm −0.71 [−1.02, −0.40] on four fresh bases | refuted | CHAIN |
| 99 | Surface-Laplacian core1; CAR + Laplacian | alone 58.9 / 66.2 (shipped 69.3); chain new −1.07 / −0.48 | refuted | CHAIN-PARTIAL (six bases, posterior-only swap) |
| 103a | core1 with the window split in thirds | 68.6 alone; chain −0.62 | refuted | CHAIN-PARTIAL (six bases, posterior only) |
| 116 | Per-band population models weighted per session by agreement | alone 51.2 / 51.2 / 51.6 (equal, agree, oracle); fused −0.57 to −0.94 | refuted | CHAIN (as a β 0.10 add-on) |

### (b) Model classes for the population model, and whole alternative systems

| § | What | Result | Verdict | Tag |
|---|---|---|---|---|
| 4 | ST-EEGFormer-small fine-tuned as member | 53.9 alone; nested +0.16 at weight 0.5 | gate failed; fresh bases not scored | CHAIN-PARTIAL |
| 6 | Pair specialists (SUB/WORD, MI/WORD, WORD/rest) and a multitask member | pair BA at or below the 3-class net; re-splits −0.56 to −3.97; multitask −0.09 / 0.00 / −0.67 | closed | CHAIN-PARTIAL |
| 12 | SessionBN versions of 8 untried families | best alone FBMSNet 62.3 (1 seed); best extra member TIDNet +0.08 at 0.5 | closed | CHAIN-PARTIAL (2 seeds, fixed weights) |
| 69 | FAIR: 64 stored candidates re-screened with the chain co-tuned, cross-fitted | mean −0.04, range −0.73 to +0.15; duplicate null 0.00 to −0.20; candidate left out in 51 of 64 | "none was closed wrongly" | CHAIN-PARTIAL (B2_pool only; stopped before two simpleconv settings and the TIDNet family) |
| 68 | Two complete chains (cores, nets) fused late | A 74.1, B 77.6 alone; best fusion −0.69 [−0.97, −0.38]; products −0.9 to −6.3 | refuted | CHAIN-PARTIAL (two bases) |
| 70 | Learned recurrent sequence stage replacing the hand stages | alone 64.6 / 65.6 against 80.2; on chain output −11.0 / −9.5; untrained +0.08 | closed "as built" | CHAIN-PARTIAL (400 steps in 24 dims, 1,500 steps in 8 dims; training posteriors came from folds that saw the held-out person) |
| 97 | S2: two-layer net on tangent PCs, band power, slow, gaze, with its own stage | alone 60.5; with stage 62.2; product −2.5 to −6.0 | refuted | ALONE for the system; CHAIN for the product; one configuration |
| 98 | S3: causal transformer trained on synthetic sessions | 5–28 % window accuracy, below chance, seed 0 only | refuted, "not debugged further" | ALONE (one seed, defect unexplained) |
| 115 | EARLY: window-local system needing no session statistics | 44.8 on windows 0–15 (chain 68.6); fused −1.15 to −2.5 | refuted | CHAIN (add-on) |
| 87 | Focal loss (γ 1, 2) on four SessionBN recipes | alone about +0.5; replacing families +0.07 [−0.19, +0.32] / −0.27; co-tuned −0.16 to +0.23 | refuted | CHAIN |
| 93 | GroupDRO, participant-balanced loss | −0.34 / −0.22 / +0.04; P04 +0.5 to +0.7 | refuted | CHAIN |
| 94 | Training length ×2, ×0.5 | +0.00 [−0.17, +0.19] / −0.60 | refuted | CHAIN |
| 57c | Own-space T·C·S fitted on other people as a final stage | 60.9 alone; −0.5 to −3.3 | refuted | CHAIN-PARTIAL (two bases) |

### (c) Training data and augmentation

| § | What | Result | Verdict | Tag |
|---|---|---|---|---|
| 7 | Scherer 2015 as member training rows | members +1 to +3 alone; chain best +0.19 / +0.28; nested −0.23 | fail | CHAIN-PARTIAL (twin control) |
| 18 | Scherer as eighth teacher corpus | teacher +3.3 alone; chain −0.98; no weight rescues it | refuted | CHAIN-PARTIAL (two bases, control m7c) |
| 21 | Scherer rows in core1's head (22 interpolated channels) | alone −4.0 / −3.3; chain −0.89 / −1.10 | refuted | CHAIN-PARTIAL |
| 53 | Scherer on the 19 real shared channels | −3.18 / −3.65; PROTEUS → Scherer 43.0 | refuted | ALONE |
| 26 | REST4: rest periods as a fourth class | alone +0.3 to +2.5; chain +0.12 / +0.29 (new +0.48: G1 +0.06, G2 +0.91) | positive, below +0.5, not adopted | CHAIN-PARTIAL (bases not stated) |
| 39 | People curve for core1 | +8.13 per doubling; 7 → 9: +3.86 | diagnostic (design cut after a 70-minute stall) | ALONE |
| 48 | People curve through a core-only chain | +7.32 per doubling; stages add only +0.8 to core1 alone | "weaker than intended" | CHAIN-PARTIAL |
| 56 NETPC | People curve for two net families | 7 → 9: +1.40 and +0.80 | nets flatten | ALONE (one seed) |
| 46b, 47 | Seven generators of virtual people for core1: rot, cap, gain, wn, fsc, mix, gmix | best −0.01; others −0.04 to −3.25 | all refuted | ALONE (never in the chain) |
| 49 | Seven people-efficient estimators and noise, in 512 PCs | none above +0.3; LDA with person-equal means +1.1 over logistic in that space | refuted | ALONE |
| 51 | SessionBN nets: chunk rotation, chunk frequency scaling, combo | alone mean −0.02 to −0.64; chain −0.09 to −0.85 | refuted | CHAIN-PARTIAL (two bases, in-harness twins) |
| 56 cmix4, 57a swa50 | Same-class mixing; weight averaging | chain −0.16 / −0.10; −0.58 / −0.57 | refuted | CHAIN-PARTIAL |
| 100 | Shifted crops as extra rows for the personal members | seen −0.04; 0.00 | refuted | CHAIN |
| 6 (note) | More net seeds | single-seed bases 79.7, three-seed 80.0, six-seed 80.0 | "more seeds do not help" | observation |
| 47 notes | ds007554, MEG set, time-shifted windows, left-right reflection | not fetched or not rerun; §63 corrects that ds007554 ran as a teacher corpus on 2 Nov (+1.67 alone, −0.98 chain) | closed | ARGUMENT |
| 63 add. | Arithmetic-vs-rest and word-vs-rest heads as separate members | not queued (second teachers were −0.1 to +0.1 on 2 Nov) | open, low prior | ARGUMENT |

### (d) Test-time adaptation inside the session

| § | What | Result | Verdict | Tag |
|---|---|---|---|---|
| 5 A03, A04, A05, A27 | Pair-specific covariance and exponents in TSEM | fresh −0.17 to +0.07; A05 new +0.24, A27 new +0.15 with seen −0.45 | refuted | CHAIN |
| 5 A06–A08 | Dawid–Skene before and after TSEM, 4 groups | fresh −0.11 to −0.63; nulls −1.07 to −1.53 | refuted | CHAIN |
| 5 A12, A13, A19, A20 | Seed selection: Sinkhorn, recursive, margin-ranked, class quota | fresh −0.08 to −0.46 | refuted | CHAIN |
| 5 A17, A18 | PC16 template stage; shrinkage schedule | −0.08; −0.15 | refuted | CHAIN |
| 5 A26 | Oracle TSEM seeds | +0.93 (new +0.66, seen +1.19); WORD +0.42 | ceiling | ORACLE |
| 13 / 31 (index) | Second Weco search over the whole session stage, 92 evaluations | best +0.6 on its half, negative on every hidden measure | closed | CHAIN-PARTIAL; only the index row and STATE §7 record it |
| 23 | Recency-limited TSEM seeds | −0.12 / −0.41 / −0.93 | refuted | CHAIN-PARTIAL |
| 24, 31 | TTT waves 1–2: nets fine-tuned on chain labels | alone +4 to +7; four families dJ +0.09 (new +0.64, seen −0.46); lr 1e-3 +0.33 | fails nested rule (all in G2) | CHAIN-PARTIAL (bases not stated) |
| 24, 28, 31 | Same with true labels | one net +0.66 / +0.89; four families +1.63; final stage +1.62 | ceiling | ORACLE |
| 28 | Fine-tuned nets as a final stage, chain labels | −0.53 to −2.19 | closed | CHAIN-PARTIAL |
| 28 | Better past labels: two-directional smoothing, run-balanced assignment | 75.4–77.3 % against 76.6 % | capped near 77 % | ALONE (true run boundaries) |
| 31 | Label feedback round 2 | +0.07; label accuracy unchanged | closed | CHAIN-PARTIAL |
| 32 add. | One fine-tuned net next to two frozen | new +0.37 to +0.66; seen −0.38 to +0.59 | "runtime-feasible form" | CHAIN-PARTIAL |
| 30 | Unsupervised clustering of a session | 44–65 % against chain 76.6; supervised 5-fold 75.9 | closed | ALONE / ORACLE |
| 35 | In-session logistic head in population PCs | true labels new +1.88 / +2.11, seen +0.52; chain labels new −0.81 / −0.59, seen −1.69 / −1.30 | closed | CHAIN-PARTIAL + ORACLE |
| 38 | Session-own FBCSP, tangent, slow members refitted in session | true new +2.5 to +2.9, seen +1.8 to +2.5; chain labels −0.14 to −0.62 | closed | CHAIN-PARTIAL + ORACLE |
| 60 | TTT in package form | alone +4.8; chain new +0.27 [−0.47, +0.92] / +0.17; true labels +1.83 / +1.76 | not built | CHAIN-PARTIAL |
| 81 | TTT3B with 6–8 seeds | ple20 +0.02; plec20 +0.16; plelr20 +0.14; true labels +1.77 to +2.09 | refuted | CHAIN |
| 83 | Why chain labels are worth nothing | chain labels −0.28; errors moved +0.39; true labels on the same confident windows +0.55; true on all windows +2.9 | "which windows" is worth +2.4; self-training closed "in principle" | CHAIN-PARTIAL + ORACLE |
| 88 | Own model from the same session's other runs, on the chain's hard third | 78.3 % against chain 60.8 %; right on 72.4 % of chain errors there | hard windows are decodable | ORACLE (non-causal) |
| 89 | Label spreading from easy to hard windows | true seeds 50.9 %; chain seeds 47.0 % | no-go | ALONE (30 PCs, k 7 and 15) |
| 90 | Hindsight session stages | +0.92 [+0.20, +1.60], all in the first fifth | bound | ORACLE (grid chain) |
| 92 | Population prior for TSEM templates, earlier start | −0.32 to −1.20; chosen arm −0.50 | refuted | CHAIN (grid chain) |
| 91 | TTT4: fix20, ent20, cons20 | fix20 +0.30 (every interval crosses 0); ent20 −0.29; cons20 −0.23; seen −0.5 to −1.3 | refuted by rule | CHAIN |
| 101 | FIX2, six seeds | fixA +0.36, fixS +0.37, fixLR +0.36, fixW +0.27, fix40 +0.19, fixPL +0.01, fixT −0.18; seen −0.7 to −1.8 | lever +0.4 "not met, narrowly"; positive, replicated, not built | CHAIN |
| 106 | Reduced forms from stored outputs | delta1 +0.26, delta2 +0.34, one6 +0.15 | delta1 passes | CHAIN |
| 108 | TTT6: cum10, cons2, soft, step5, head, view2 | no result in the ledger | open | — |
| 113 | TTT7: GPU augmentation and cheaper schedules | no result in the ledger | open | — |
| 103c | Labelled Procrustes with whole-session true class means | LDA 64.6 → 89.1 | reading mark met | ORACLE (non-causal, includes test windows) |
| 104 | Procrustes with causal means | true labels 68.7; label-free 63.8; fused −2.6 to −5.8 | refuted | ORACLE + CHAIN (add-on member) |

### (e) Use of protocol structure

| § | What | Result | Verdict | Tag |
|---|---|---|---|---|
| 2 | Order fingerprint H1–H3 against the sealed drops | −1.95 to −3.83 under reorderings; none matches | hypotheses rejected | CHAIN |
| 3 | Order-robust re-tune | never run | — | not run |
| 62, b, c, d | Ten stream cuts and orders against four sealed numbers | a two-number match appeared, then was rejected by the routed packages | no cut or order fits; chain loses 3–5 under reordering | CHAIN-PARTIAL (two bases, only stage state cut) |
| 5 A01 | Run-count prior inside true runs, chain's own counts | −0.76 to −4.24 | refuted | ORACLE metadata |
| 5 A02 | Same, label-free runs | fresh −0.75 | refuted | CHAIN |
| 5 A24 | Rebalancing window 60 or cumulative | fresh −0.44 (new −1.16) | W 30 is what the chain exploits | CHAIN |
| 9 | Graz cue-order prior, true interface and run position | prior alone 56 %; weight 1 −2.94; best fixed weight about +0.4 J | "≤ +0.4, not pursued" | ORACLE, one base |
| 55 | Same with a learned fusion | −0.67 / −0.74 (new +0.23 / −0.17; seen −1.56 / −1.30) | refuted | ORACLE, two bases |
| 27 | Reconstructing the cue generator, 247 M seeds × 12 variants | no match; glibc, other seeding, Lua 5.3+ not tested | refuted for this family | ALONE |
| 33 | Feedback leak in online runs | none beyond 0.05 | no leak | ALONE |
| 103b | Interface as a stage variable | rebalancing per interface −0.41; teacher weight per interface −0.01 | closed | ORACLE (grid chain) |
| 109 | Run type as a stage variable | rebalancing −0.10; SessionEM + TSEM per run type −2.98 | closed | ORACLE (grid chain) |

### (f) Person-level methods

| § | What | Result | Verdict | Tag |
|---|---|---|---|---|
| 10 | Raw spectral correlation biometric | rank-1 6/6 at n 60; cannot reject new people early | refuted | ALONE |
| 11, 14 | Two-covariance identity model on 224 external people; routing v4 / v4.1 | external 86.7 % accepts at 1.9 % false accepts; dev +0.58 (seen +0.90 to +1.32); **sealed +0.267** | adopted | CHAIN + sealed |
| 15 | Ten-person within-session proxy for the personal head | +5.78; real P04 +2.6, P10 −1.0 | diagnostic | ALONE (core level) |
| 16 | Similarity routing for new people | most similar head 47.1; weighted 51.5; pooled core1 66.2 | closed "in this form" | ALONE (not pre-registered) |
| 17 | Personal recipes; C 0.03 head taken into the chain | proxy +6.56; chain −0.30 | refuted | ALONE, then CHAIN |
| 19 | Personal class templates as TSEM prior | fresh seen +0.46, J +0.23; six-base +0.19; P10 < −0.3 on two bases | fails the P10 clause | CHAIN |
| 22 | Personal-head slot weight 4.5 | +0.12 (my mean of the six listed J values), mixed signs | not adopted | CHAIN (not pre-registered) |
| 29, 31 | PFT: nets fine-tuned on a returning person's session 1 | seen +1.44 to +2.05 (P04 +2.4 to +4.5; P10 −0.5 to +0.5) | positive, P04's | CHAIN-PARTIAL |
| 31 | Personal head on ocular and slow features; family re-weighting for seen | −1.5 to −7; −0.0 to −2.9 | closed | CHAIN-PARTIAL |
| 32 | PFT proxy, standalone nets | +5.67, every person gains | "worth a build" | ALONE |
| 37 | Own-space T, C, S averaged into the head's slot; tempered-product arm | −0.68 to −1.32; product +1.07 / +1.45 (best of 20) | refuted; product unstable (§40) | CHAIN-PARTIAL |
| 40 | PFT and own-space together | PFT +0.94 / +1.20; BOTH +2.01 / +1.88 | not redundant | CHAIN-PARTIAL |
| 41 | Ten-person proxy for own-space stage; per-person space gate | T·C·S +4.17; P·T·C·S +5.0; gate −0.09 to −0.42 | gate dropped | CHAIN-PARTIAL (proxy) |
| 42 | 12 real session pairs on a person-naive chain | +10.35 (P head +5.9); second session +1.6 | positive | CHAIN-PARTIAL |
| 43 | Dose-response and fresh-base confirmation, β 0.10 | fresh seen +1.93 (P04 +4.00, P10 −0.14); with PFT +2.34 | passes; prediction not on disk first | CHAIN |
| 44 | Nine member variants | proxy within 0.5; V8 not better on the real chain | plateau | CHAIN-PARTIAL |
| 45 | Stage where the real gate routes | seen +1.79 (P04 +3.51, P10 +0.07); new 0.000; dJ +0.89 | built; **upload 5 sealed +0.361** | CHAIN + sealed |
| 46a | Perfect recognition from window 0 | −0.24; with stage −0.43 | closed | ORACLE |
| 50 | Up-weighted personal core | −0.26 to +0.01; adds nothing to the stage | refuted | CHAIN-PARTIAL (posterior only) |
| 57b | Template prior × slot gain on top of the stage | −0.13 to −1.05 | refuted | CHAIN-PARTIAL |
| 64 | Warm-started personal head | real gate −0.13 to −0.65 | closed | CHAIN-PARTIAL |
| 65 | Forced impostor routing | v4.1 −1.58; stage extra −0.6; worst person −12.6 | negligible sealed cost | CHAIN-PARTIAL |
| 67 | 10 × 10 own-space transfer matrix | mean 46.1; best source 56.8 against pooled 60.9; pick by agreement 55.2 | "a map, not a method" | ALONE |
| 78 | Personal nets on top of the stage | pfh60 +0.17; pf60 +0.25 (P10 −0.63) | lever refuted | CHAIN |
| 79, 80 | Trust-scaled β | signals anti-correlate with best β (−0.65 to −0.94); TRUST2 +0.12 | refuted | ORACLE diagnostic, then CHAIN |
| 82 | Proxy with the person in the population | stage +3.2 to +3.8; in-population nets +4.5 to +5.1 over LOSO | validity "passes by the letter, barely" | CHAIN-PARTIAL (proxy) |
| 84, 85 | Stacked personal correction | proxy +1.7 over the stage; real −0.18 | refuted | CHAIN |
| 95 | β on six bases | 0.05 +1.18; 0.10 +1.79; 0.15 +1.94; 0.20 +1.18 | β 0.10 kept | CHAIN |
| 96 | Personal nets + stage, proxy | +2.4 to +2.7 over the stage; 8 of 10 ahead | built as v5.2 | CHAIN-PARTIAL (proxy) |
| 110 | Person-type experts | oracle expert +2.1 window accuracy; fused −2.7 to −4.2 | refuted | ORACLE + CHAIN (add-on) |
| 111 | Stage forms ranked on the proxy | β 0.25 +1.7 to +1.9 over v5.1's form; real β 0.15: P04 +0.40, P10 −0.11 | rule not met | CHAIN-PARTIAL + CHAIN |
| 114 | Adaptive personal LDA system | proxy +0.83 to +1.00; real −0.81 (P10 −1.57) | refuted | CHAIN |
| 119 | Personal nets + stage at β 0.15 | no result | open | — |

### (g) Combination of members and stage tuning

| § | What | Result | Verdict | Tag |
|---|---|---|---|---|
| 5 A29, A30 | Class-column weights for deep families and the teacher | fresh −0.06; −0.11 | refuted | CHAIN |
| 5 A15 | Combination of the arm picks | not run | — | not run |
| 31 | Log-linear pool with fitted weights; mix confidence exponent | pool +0.3 to +1.2 at mix level, −1 to −2 in chain; exponent 0.9 +0.2, 1.1 −0.4 | closed | CHAIN-PARTIAL (exploratory) |
| 66 | 243-setting retune under J and a sealed-like weighting | shipped is rank 6 of 243; nested −1.44; per-person best +2.4 | not mis-tuned | CHAIN-PARTIAL (B2_pool, plain chain) + ORACLE |
| 75 | TSEM λ | λ 0.20 +0.15 [−0.12, +0.41]; cross-fit −0.07 | λ 0.15 stays | CHAIN |
| 76 | Per-session choice among seven stage variants | oracle +1.44 / +1.00; agreement vote +0.63 / +0.32; others −0.14 to +0.27 | no criterion | ORACLE + CHAIN-PARTIAL |
| 61, 61b | Bagging chains or stage settings | −0.51, −0.07, −0.27 / −0.16; window bag +0.03 on fresh | closed | CHAIN |
| 117 | Per-session choice among 17 weightings | oracle +1.2 to +2.1; label-free −0.16 to +0.04 | oracle open, no criterion | ORACLE + CHAIN |
| 118 | Learned selector from window 60 | oracle +0.96 to +1.36; selector 0.00 on every base | refuted; recorded as a no-op | ORACLE + CHAIN (one model) |

### (h) Evaluation methodology

| § | What | Result | Tag |
|---|---|---|---|
| 5 D7 | Eight pure-noise arms | fresh −0.22 to +0.10; picked exploration −0.16 to +0.18 | CHAIN |
| 5 D1, D3, D5 | Pair separability, family accuracy on chain errors, per-session anatomy | see part 5 | ALONE / ORACLE |
| 52 | Soft gates; audit of silent no-ops | findings in part 2 | audit |
| 63 | Independent review | bases not independent; person SE 0.5–1.1; a synthetic independent 70 % member is worth new +3.1; softened core1 reaches 72.3 after stages; v5.1 estimate cut to +0.3 | review |
| 73, 74 nulls | Random second views | single draws −0.27 to +0.58; mean +0.14 / +0.20 | CHAIN |
| 75 | Exploration bases overstate stage effects | λ 0.20: +0.64 / +0.30 there, about 0 on the other four | CHAIN |
| 98 note | Dev split by session index | P04 as new 56, 55, 50; P10 77, 86, 82 | ALONE |
| 105 | Profile of the v5.1 replay; sealed expectation for v5.2 +0.20 (−0.10 to +0.50) | see part 5 | one base |
| Uploads 4, 5 | Dev to sealed transfer | v4.1 0.46×; v5.1b 0.41×; proxy to sealed 0.19× | sealed |

### (i) Runtime and engineering

| § | What | Result |
|---|---|---|
| 1 | Fast solver | bitwise parity on 2,655 windows; adopted; 1,128.6 s sealed against 3,732 s |
| 8 | GPU-batched eigendecomposition | result table is empty in `PREDICTIONS_11NOV.md`; the index calls it "speed probe only". Whether it ran is not recorded |
| 45 add., uploads | v5.1 and v5.1b checks | passed; 1.06–1.08× paired, 1.14× on the sealed worker; no worker-image replay of the v5.1b rebuild |
| 102, `V52_BUILD.md` | v5.2 (300 personal traces, 85 MB) | local check passed; in-sample 0.99462; not uploaded |
| 107 | Fine-tuning cost on an A100 | full recipe about 142 s per session, delta1 about 20 s, delta2 about 39 s; vmap gives no gain |
| 112 | v5.2 runtime | the 3.9× was a cold-start artefact; warm 1.40×; v5.2b 1.27× with outputs equal; about 24–27 min sealed; worker-image replay and the GPU `--ref-schedule` run pending |

Not experiments: §34 (plan), §59 (summary), §102 (preparation).

---

## 2. Closures I consider weak, and why

**A. New members judged as one more member at fixed weight with the old stages.** §4, §12, §25, §26, §6 (multitask), §58 (OCUNET as member). §69 re-screened 64 of them co-tuned, but on B2_pool only, at weights 0.5 / 1 / 2, still as posteriors, and it stopped before the TIDNet family, which was §12's best extra member. `REVIEW_12NOV.md` §6 says every build was "scored on all six bases with person intervals"; the ledger shows §68, §70, §71 on two bases and §69 on one.

**B. New cores swapped in as posteriors only.** §13, §20, §21, §50, §99, §103a. `REVIEW_12NOV.md` §5.3 asked for "a new core with the session stage and routing looking through its features". That build never ran: §99 made the rebuild conditional on the posterior-only swap passing.

**C. Whole systems scored only as add-ons, in one configuration.**
- §97 S2: one architecture, fed 256 PCs although §49 showed 512 PCs already lose four points.
- §98 S3: below chance on one seed. The cause is a guess ("most likely the sequence structure"), not debugged, and the first run had a rotation bug.
- §70 SEQ: two settings, with a leakage caveat in its inputs.
- §68: both "systems" share the teacher and the same training people.
- §110, §114, §115, §116: each fused as a β 0.10 or fading product into the frozen chain.

**D. Core-level gates used to close directions.** §46b, §47, §49, §53 and COREFUSE never entered the chain. §63 point 4 says they "filtered on the wrong quantity", since standalone core accuracy barely predicts chain value (§7 and §18 show both directions). §49 also ran in a space already four points down.

**E. One hyperparameter setting or one form.**
- §89: one feature recipe.
- §118: one ridge model that never switched.
- §80: one map.
- §104: the alignment ran only as an extra LDA member. §103c's pre-registered follow-up was "a label-free version on TSEM's templates", which did not run.
- §109: only full separation of statistics.
- §55: fusion fitted on new-person rows and applied to seen; new alone was +0.23 / −0.17.
- §58 EMVIEW: stage strength not re-chosen for a larger view.
- §92: two prior strengths.

**F. Results resting on P04 and P10, or on P04 alone.** Every "seen" number. Gains that are P04's: §14, §19, §29/§31, §43, §45, §58b, §78, §95; §31 TTT's new-person gain is "all of it in half G2 (P04)". Closures decided by P10 alone: §19 (0.26 = one window), §78 pf60 (P10 −0.63 against −0.5), §111 β 0.15 (P10 −0.11), §114, §85. P10 is at 91 %. The proxies (§41, §82) and upload 5 (about +0.6 per non-P04 returning session, on an assumed cohort) suggest the other eight are not P10-like.

**G. Closed by a small margin against a pre-set bar.**
- §58b: +0.29 against +0.30.
- §101: +0.36 to +0.37 against +0.40.
- §91: +0.30 against +0.40.
- §60: missed by 0.07.
- §73: +0.20 against +0.25.
- §111: P10 −0.11.
- §114 proxy: +0.83 to +1.00 against +1.0.
- §26: +0.29 against +0.5.
- §110: oracle +2.1 against a "< +2 closes" rule, recorded as refuted "at the mark".

**H. What ran differs from the description.**
- §51/§52: `cmix` and `fsc` were silently ignored by `SessionNet._train`, so the October closures never ran on SessionBN nets (re-run in §51, §56, §57a). MixStyle never ran on ShallowConvNet and is still untested on SessionBN nets. Segment recombination never reached the teacher's last three corpora.
- §78 against v5.2 (`V52_BUILD.md`): the dev form used five families fine-tuned on session 1 and also swapped them in the gate's shadow tail. The package swaps 30 SessionBN traces on routed windows only, and 12 of the 30 are fine-tuned from retrained twins, not the shipped nets.
- §43: the confirmation's predictions reached disk after the result.
- §39: design reduced after a stall.
- §81: the "one fine-tuned seed" section failed and is unreported.
- §91: the first scoring printed +0.00 with no files. §101: scored on partial files once. §114: a "one-base smoke" ran all six bases. All are recorded as fixed.
- §8 and §3 have no result.
- The session-stage Weco search has no detailed record in the documents I read, and §52 says its scratch was not audited.
- The ledger index still gives v5.1's sealed estimate as 0.820–0.825; `STATE.md` says 0.819 (0.817–0.824).

---

## 3. Oracle and ceiling measurements, and what label-free methods captured

| § | Oracle | Size | Label-free capture |
|---|---|---|---|
| 5 A26 | True TSEM seeds | +0.93 J (new +0.66) | none: A12, A13, A19, A20 all ≤ 0 |
| 5 A09 | True-label common-mode and physiology views | ≤ +0.11 | nothing to capture |
| 5 D3 | Some member family right on the chain's errors | 76–92 % by class | none: A29, A30, §117 |
| 9, 55 | Cue-order prior with true run structure | ≤ +0.4 J in-sample; learned fusion −0.7 | none built |
| 24, 28, 31, 60, 81 | TTT with true labels | four families +1.63 J; final stage +1.62; package form new +1.77 to +2.09 | +0.02 (ple20) to +0.36 (fixA), under a fifth |
| 35 | True-label in-session head, population PCs | new +1.9 to +2.1; seen +0.5 | −0.6 to −0.8 |
| 38 | True-label session-own members | new +2.5 to +2.9; seen +1.8 to +2.5 | −0.1 to −0.6 new; for returning people the own-space stage captures it from public labels (seen +1.79 dev, sealed +0.361) |
| 83 | True labels on the chain's confident windows only | +0.55 / +0.73 | the remaining +2.4 needs labels on windows the chain is unsure of |
| 36 | Within-session supervised, session-own spaces | 81–82 % against chain 76.6 % | none for new people |
| 30 | Supervised 5-fold in core1 PCs | 75.9 % | matched by the chain |
| 88 | Own model on the hard third (non-causal) | 78.3 % against 60.8 %; about +6 points of window accuracy overall by my arithmetic | none: §89 gives 47 %, §86 views lean to the chain's wrong class |
| 89 | Spreading true labels from easy windows | 50.9 %, below the chain | negative oracle |
| 90 | Hindsight stages | +0.92, first fifth | none: §92 −0.50, §115 −1.15 or worse |
| 66 | Best constants per person | +2.4 | nested −1.44 |
| 76 | Best of seven stage variants per session | +1.44 / +1.00 | agreement vote +0.63 / +0.32, two bases only |
| 117, 118 | Best of 17 weightings per session; from window 60 | +1.2 to +2.1; +0.96 to +1.36 | −0.16 to +0.04; learned selector 0.00 |
| 103c, 104 | Procrustes with true class means | +24.4 non-causal; +4.1 causal | −0.8 alone; −2.6 to −5.8 fused |
| 103b, 109 | True interface; true run type | −0.41 / −0.01; −0.10 / −2.98 | not applicable |
| 46a | True identity from window 0 | −0.24; −0.43 with the stage | the real gate is better |
| 79 | Best β per returning session | P04 0.15–0.40 (+7 to +8); P10 0–0.05 | TRUST2 +0.12 |
| 110, 67, 16, 116 | Best expert, source, head or band weights per session | +2.1 window accuracy; 56.8 against pooled 60.9; 53.3 against 66.2; 51.6 against 51.2 | selection loses to pooling |
| 63 | Synthetic member with independent errors, 70 %, weight 3 | new +3.1 | the teacher is the only real one (50 % alone) |
| 39, 48 | More training people | +8.1 per doubling at core level; +7.3 through a core-only chain | nothing substitutes; full-chain slope unknown |
| 41, 82, 96 | Within-session proxies for returning people | stage +3.2 to +4.2; nets + stage +5.7 to +6.3 | real sessions keep about half for P04, none for P10; sealed survival about 0.19× |

---

## 4. Positive results not built or not combined

| Item | Size | Status and stacking |
|---|---|---|
| Augmented-input fine-tuning (§91, §101, §106) | new +0.36 on six bases, replicated; delta1 +0.26, delta2 +0.34; seen −0.4 to −1.8 so it must be gated; delta1 about 20 s per session | not built. §108 and §113 are open. Never scored as one system with v5.2 |
| Personal nets on top of the stage (§40, §43, §78, §96) | real +0.17 to +0.25 (earlier +0.4 on four bases); proxy +2.4 to +2.7 | built as v5.2 / v5.2b, not uploaded; expectation +0.20 (§105) |
| β 0.15 (§95, §111) | seen +0.15 over β 0.10; proxy +1.0 to +1.2 | not adopted; joint test §119 open |
| Eye-net log posteriors in SessionEM's view (§58b) | +0.28 on six bases, none negative; seen +0.50 | not built; never stacked with the stage, which targets the same sessions |
| REST4 (§26) | chain +0.12 / +0.29; new +0.48 | not built; never stacked. §69 re-screened the rest-class nets as added members, not as replacements |
| Personal template prior (§19); slot weight 4.5 (§22) | +0.19; +0.12 | stacked with the stage in §57b on two bases: −0.13 to −1.05 |
| PLS-DA view (§74); net view (§73) | +0.45 to +0.60 on two bases over a +0.25 null; +0.07 on six | PLS-DA never taken to six bases |
| Agreement vote among stage variants (§76) | +0.63 / +0.32 | two bases only; §61's stage-setting bag was −0.27 / −0.16 |
| Mix confidence exponent 0.9 (§31) | +0.2 J | exploratory, no follow-up |
| MI/SUB pair covariance (§5 A05, A27) | new +0.15 to +0.24, seen down as much | never tried for the unrouted tail only |
| TTT at lr 1e-3, 20 steps (§31) | dJ +0.33, seen +0.04 | superseded; not re-read in that form |
| LDA with person-equal means in the reduced space (§49) | +1.1 over logistic | side finding, unused |

Stacking tests that exist: §57b (negative), PFT + stage (adds), §73 (two views add nothing), §61. A15 was never run. No run combines fine-tuning for new people, the returning stack, the eye-net view and REST4. §59 concludes from §57b alone that "small levers do not stack".

---

## 5. Where the error is

**By person.**
- Chain accuracy as a new person, P01–P10: 79.0, 90.5, 75.0, 53.6, 80.5, 82.9, 89.4, 85.6, 83.6, 81.6. Without P04 the mean is 83.1 (§67).
- P04 holds 263 of 622 new-person errors (42 %); P04, P10 and P03 hold 67 % (§5 D5).
- Every member scores 37–51 % on P04 against 52–71 % on the others. The mix is 48.6 against 73.2, and the stages add 5 points for P04 against 10 (§67).
- P04 is decodable from their own data: 76 % across their sessions, FBCSP 74 % within session 1. From anyone else the ceiling is 46–49 %. Their signal is theta CSP (60.6) and slow potentials (59.3), and the population's slow model reads them at 30 % (§36, §67, §77).
- Transfer between people is symmetric (r = +0.80); P01, P02, P07, P08 serve each other at 56–69 %. Best incoming transfer predicts chain accuracy (Spearman +0.78) and agreement with the chain predicts a source's accuracy (+0.83) (§67).
- P05 has SUB and WORD swapped in alpha CSP and loses with any within-session learner, true labels included (§77, §83, §90).
- Eye-movement features help about half the people (P02 +4.3) and mislead the rest (P04 −5.9) (§58).

**By session phase.**
- New people: 63 % in windows 0–5, 70 % in 5–10, 75–79 % from 10 to 90 (§20).
- Without P04: 72.7 % in windows 0–15, then 80.6, 82.4, 82.3, 86.0, 84.3; the first 15 windows cost about 0.9 (§63).
- By fifth of the session on the v5.1 replay: 73.1, 80.5, 76.5, 77.6, 75.1 (§105).
- 48 % of the first fifth is "hard" against 26 % of the last (§89). A perfectly warmed-up stage is worth +0.92, all in the first fifth (§90).
- Returning people drop from 83–86 % to 76–79 % after window 90, the online runs (§20).

**Hard windows.** The lowest-confidence third is 60.8 % right against 84.5 % elsewhere. The person's own model gets 78.3 % there. Every other sensor view picks the chain's wrong class (38–46 %) more often than the true one (27–34 %) (§86, §88).

**By class.** MI 81.0, SUB 75.2, WORD 73.5 (§105). Confusions: MI↔WORD 37 %, SUB↔WORD 36 %, MI↔SUB 27 %. The true class is the runner-up in 73 % of errors. A margin under 0.1 covers 38.7 % of errors and 13.0 % of correct windows, so small reweightings are zero-sum (§5).

**By interface.** Equal: BrainHero 76.7, Graz 76.5 window accuracy (§105); cell BA 80.0 and 80.3 (§30).

**By run type.** New people 75.4–77.3. Returning people: Graz calibration 87.5 against Graz feedback 79.2 (§105).

**New against returning.** New 76.6 % of windows, returning 82.9 %. P04 is 54 as new and 69 as returning; P10 82 and 91. In-population nets add +4.5 to +5.1 on the proxy (§82).

**Members and stages.** The 12-member mix is 0.6 above core1 alone before the stages (66.7 against 66.1), and the stages add about +10 to the mix but +0.8 to core1 alone (§31, §48). Members trained on the same people do about a point worse in the chain than independent noise of equal accuracy (§63).

**Sealed.** Upload 4: 0.8161546323, +0.267, 0.46× of dev. Upload 5: 0.8197668759, +0.361, 0.41× of dev, 1,287 s. The split into "P04 +0.10 and about +0.6 per other returning session" is arithmetic on a cohort text the organisers withdrew. Any reordering costs the chain 2–5 points (§2, §62).

---

## 6. Directions the documents themselves name as untested or self-imposed

**Self-imposed limits (`REVIEW_12NOV.md` §4, §7).**
- Uploads rationed and never used as probes. No upload isolated the teacher, the routing or the returning block, so the sealed new/returning split is unknown.
- One chain extended by additions.
- The 57 / 43 cohort rests on a withdrawn text.
- The warm-up board was dismissed.
- Runtime: 19–21.5 of 60 minutes used.
- Gate habit and "do not re-propose" lists; four listed items later shipped.

**Named and still not run.**
- A new core with the session stage and routing looking through its features (review §5.3).
- What to do once a P04-like session is detected (review §5.4).
- Gate recall against the gap between public and sealed session; three-session galleries were never tested externally (review §7, §14).
- Order robustness as risk control; §3 never ran.
- Members from another population with independent errors (§63); arithmetic-vs-rest and word-vs-rest heads as separate members.
- Self-supervised pretraining on PROTEUS's unlabelled recordings; source-space priors; time-frequency images (§25 map).
- MixStyle on SessionBN nets; segment recombination on the teacher's later corpora (§52).
- People-efficient estimators on the full feature set and inside the chain (§49).
- The full-chain people slope (§48).
- A fusion that tells "lookers" from "non-lookers" inside a session (§58).
- Other generator families for the cue order (§27).
- ds007554 as training rows and the 17-person MEG set (§47).
- A v4.1 / v5.1 reference in `replay/check.py`: listed in the review's plan, and no document records it as done.
- A learner using the session's cluster structure: §89 tried one form.

**Open.** §108, §113, §119; the worker-image replay and GPU `--ref-schedule` check of v5.2b (§112).

**Not measurable on dev, by the documents' own account.** The eight single-session returning people, and whether sealed new-person sessions are later, easier sessions than dev's first sessions.

---

## Files
- `<repo>/docs/LEDGER.md`
- `<repo>/docs/PREDICTIONS_11NOV.md`
- `<repo>/docs/PREDICTIONS_12NOV.md`
- `<repo>/docs/REVIEW_12NOV.md`
- `<repo>/docs/STATE.md`
- `<repo>/docs/SEALED_UPLOADS.md`
- `<repo>/docs/V52_BUILD.md`
- `<repo>/studies/12nov/README.md`
- `<repo>/studies/13nov/` (docstrings of `ttt6.py`, `ttt7.py`, `ttt_reduce_score.py`, `ft_bench*.py`, `profile_v52.py`, `gpu_cpu_spread.py`)
- `<repo>/studies/11nov/` (no README; scripts only)
