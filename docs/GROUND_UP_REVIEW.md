# Track 2 (PROTEUS): ground-up review, 30 Oct 2026

**Scope.** This review starts from first principles and covers what we know, what the literature offers, the remaining solution space, a ranked plan and the sealed-upload strategy. It draws on the round docs (`docs/OPTIMIZATION*.md`, `docs/STRATEGY_R16.md`, `docs/HANDOFF.md`, `docs/SEALED_PLAN.md`, `docs/COMPETITION.md`), the official submodule (`2026-competition/`, written `OC/` below), this session's research reports and diagnostics (scripts and logs under `/private/tmp/claude-501/-Users-sebastianboehler--claude-spares-spare-Pdn7MS/f1aede87-a130-4f66-8c95-5c7dc01a3281/scratchpad/`), and two adversarial critiques of each of the 13 candidates.

**Tags.** **[F]** is a fact with a source. **[I]** is an inference.

**Abbreviations.** ds = dev_session, dla = dev_loso_all, dl = dev_loso, al = all_loso, ss = seen_session, sel = mean(ds, dla).

**State correction.** The task context lists R16, R17 and R18 as running. All three have finished and are recorded in the working tree's `docs/OPTIMIZATION_R10.md` (lines 369–425):
- **R16** (rotation-pretrained FBCNet): six-seed pooled sel 78.98 / 79.01 against 78.68. On seeds 3–5 alone it is 78.26 / 78.11 against 78.23. Not adopted.
- **R17** (unified rest-anchored pretraining): 77.83–78.92 against 79.59. The pretrained head hurts. Closed.
- **R18**: MixStyle is neutral (±0.4). Participant-adversarial training hurts every family, more so at higher λ. Closed.

R15 therefore remains the best package. Nothing trained since R15 has cleared the adoption rule.

---

## 1. The challenge from first principles

### 1.1 Formal problem

**Domains and cells.**
- A domain is d = (participant p, session s). Each domain is split by interface c ∈ {Graz, BrainHero} into cells. Graz and BrainHero share a session but are scored separately.
- Score: J = mean over cells of BA_{d,c}, where BA is the mean recall over the union of true and predicted classes (`OC/benchmark_utils/metrics.py`, `OC/tracks/bci_decoding/objective.py`) [F].
- A cell holds about 95 windows from 3 runs, and every run is exactly or nearly class-balanced (10/10/10 online, 12/12/12 Graz acquisition) [F, `data/proteus_windows.npz`].
- Because cells are balanced, BA ≈ accuracy. For a known uniform prior, the Bayes rule is argmax of the posterior after correcting the classifier's marginal, which is what causal rebalancing does [I].

**Decision rule.** ŷ_t = f(x_1..x_t). The state resets per session (fresh deepcopy plus `reset_state`). The step index t is implicitly available. Not available at test time: run boundaries, interface, onset, participant (`OC/benchmark_utils/streaming.py`) [F].

**Information available.**

| Source | Train | Test |
|---|---|---|
| 2,655 labelled 4 s windows (cue +1 to +5 s; feedback starts at +3 s, so 2 s of each window are feedback) | yes | – |
| Metadata (participant, session, interface, run, onset) | yes | no |
| Earlier windows of the same session, unlabelled | yes | yes (causal) |
| Rest runs, inter-trial EEG, cue +5 to +8 s, raw 500 Hz EDF | yes (allowed, `scientific_faq_bci.md` l.58–63) | no |
| External public corpora (must be declared) | yes | – |
| EOG/EMG channels, Fp/occipital electrodes | no: 41 fronto-centro-parietal channels, M1 reference (`*_eeg.json`) | no |

**Shift structure.**
1. **Covariate shift.** Amplitude and spatial covariance differ between domains. Session alignment largely removes this.
2. **Conditional shift.** P(y|x) differs between participants. This is the dominant term [F, evidence in §2]. It cannot be identified from unlabelled x without a structural assumption [I; Zhao et al. 2019, arXiv 1901.09453; LLP rank argument below].
3. **Label shift.** Class proportions are uniform per run. Graz runs also follow a biased W,S,M position sequence: position alone predicts 54% leave-one-participant-out. On top of rebalancing it is worth at most about +0.3, even with oracle run boundaries (`scratchpad/posprior.py`) [F]. Closed.

**Why output-level label-free methods saturate** [I]. Every run has proportions 1/3 : 1/3 : 1/3, so the proportion matrix is rank 1. Learning-from-label-proportions therefore identifies only the pooled mean, not class means (Quadrianto et al. JMLR 2009; Hübner et al. 2017). This is the formal reason that Sinkhorn, exact counts and the order prior all top out at about +0.4.

**Budget.**
- Inference: 3600 s on one A100 [F].
- Projected sealed runtime: 11–13 min for about 21k windows, so 2.8–5× headroom (`docs/COMPETITION.md`, `docs/OPTIMIZATION_R8.md:521`) [F/I].
- Package: one `submission.py` plus weights. `submission.py` is already 414 lines.

### 1.2 What bounds performance

| Bound | Value | Source |
|---|---|---|
| Within-session ceiling (own labels, about 160 training windows) | mix 81.9. This is a *lower* bound on per-person separability, because the models are data-starved (ShallowConvNet-EA 42.6 there). | `docs/OPTIMIZATION_R8.md` |
| Same later sessions scored as unseen vs seen (own session 1 in training) | 56.6 → 72.0 (+15.4; BrainHero +18.6, P10 BrainHero +22–25) | diagnostics (d), `scratchpad/diag_mix.log` |
| Per-session logit bias / affine map, fitted with labels | +2.0 / +2.6 in-sample (al). Picked on the other interface: −1.9 dla (70.34 vs 72.11) | `docs/OPTIMIZATION_R10.md`; diagnostics (e) |
| Per-session oracle rebalancing or member weighting | in-sample +2.5 / +4.3 dla; on the other interface below default | diagnostics (f) |
| Exact per-run counts, oracle boundaries | al +0.38, ss +1.81 | `docs/OPTIMIZATION_R10.md` |
| Label-shift-aware reference (oracle) | +0.05 | `docs/OPTIMIZATION_R8.md` |
| Session clustering with oracle cluster-to-class map | 0.37–0.39 | `docs/OPTIMIZATION_R8.md` |
| Value of one more in-distribution training participant | about +0.85 | `docs/OPTIMIZATION_R10.md` |
| Errors near-tie: true class ranked 2nd | 65–69% of errors; 60% of errors at confidence < 0.5 | diagnostics (g) |

[I] The about 10-point gap between unseen participants (dla 72) and the within-session ceiling (82) lies in participant-specific decision directions, not in a bias or a reference. Label-free methods can only reach it through:
- (a) a better shared representation, which saturates with about 10 training participants (domain lower bound: Wang, Wu & Zhang, AAAI 2024; IRM needs more environments than spurious dimensions, Rosenfeld et al. 2021);
- (b) an *independent, low-dimensional* view in which a session's class structure can be estimated causally without reusing the classifier's own errors.

The gaze EM in §5 is the first working instance of (b).

### 1.3 Two populations, two ceilings

The organiser FAQ says the sealed set is later sessions of the 10 public participants plus 10 new participants [F]. Our estimate of 57% unseen / 43% seen by cells assumes all six sessions are used (`docs/OPTIMIZATION_R8.md:411–414`) [I]. The NeuralBench guide says "sessions 1–3 train, 4–6 test" [F, hint only].

**Unseen participants (about 57%).**
- Best local proxies: dla 72.11 (8 participants, P04 and P09 folds dropped) and al 67.03 (all 10, including P04 ≈ 49 and P09 ≈ 69).
- For randomly drawn new participants, al is the less biased estimate [I], because dla excludes the hardest known participant.
- Spread across participants is large: per-cell 59–89, bootstrap sd 2.6.
- Realistic ceiling in the remaining 22 days: +1 to +2.5 on dla. The gaze/event view gives about +1 to +2 on dev before winner's-curse shrinkage. Everything else is at most about +0.3.
- Sealed unseen expectation: about 66–69 [I].

**Seen participants (about 43%).**
- ds 85.25 is the optimistic end: one participant (P10), adjacent sessions, an easy recognisable case.
- ss ≈ 72 (R10-like proxy, P04 + P10) is the pessimistic end.
- Sealed seen sessions are probably sessions 4–6, further in time than our proxies, and nothing local measures that drift [I].
- Every public session is in training, which already captures most of the +15.4 seen-participant effect.
- The remaining lever (personal routing, oracle +2.8 on ss) is blocked by recognition: 0 of 4 correct own-template matches at window 30 even with oracle labels (`scratchpad/xsess_gaze.py`, `refute_lib.py`).
- Realistic headroom: about 0 to +0.5. Sealed seen expectation about 72–80 [I].

**Composite** [I]: J_sealed ≈ 0.57 × (66–69) + 0.43 × (72–80) ≈ 69–74. Every term carries participant-sampling uncertainty of about 2–3 points, shared by all teams.

---

## 2. What our evidence says

### 2.1 Step changes and their mechanisms

| Step | Gain | Mechanism [I] |
|---|---|---|
| Causal session recentering (R1) | +7 dl, +14.6 ds | Unlabelled session statistics remove nuisance covariance (`docs/OPTIMIZATION.md`) |
| < 4 Hz band + centred < 1 Hz waveform (R1) | +6.8 dl, +15.0 ds | A new signal: slow frontal potentials, largely gaze/ocular-like and display-locked. Signs are consistent: SUB frontal positive 8–9/10, WORD left-minus-right negative 8–9/10 (neuro report) |
| Heterogeneous probability ensemble (R2) | +6.8 dl, +5.5 al | Variance reduction across inductive biases (pairwise error correlation 0.33) |
| Causal rebalancing → windowed W=30 (R3–R4) | +2.3 dl, +6.8 ds, +3.0 al, +3.0 ss | Balanced-run prior: removes session class bias |
| Spectral front-end diversity (R3–R9) | dl 65.1 → 70.0, al 62.8 → 66.95 | Several band parametrisations (fixed FIR, learned FIR, sinc, FBLight's first 2.08 s) |
| Multi-corpus pretraining + rotation (R10–R15) | about +0.4 sel pooled R9 → R15 | Out-of-distribution priors; invariance to residual spatial rotation |

Every gain of 2 points or more landed by R4. Since R10 each adopted step is at most +0.3 pooled (R10 78.30, R14 78.46, R15 78.68).

### 2.2 The plateau and measurement noise

**Noise levels.**
- Seed noise on sel: about ±0.7. The same recipe gave 79.59 on seeds 0–2 and 78.23 on seeds 3–5, a 1.4 spread.
- Participant bootstrap sd: 2.6.
- Binomial sampling sd of a cell mean: about 0.8–0.9 (diagnostics (d)).
- Selection uses one dev_session participant (P10, 4 cells), which carries half the weight in sel.
- R10 → R15 was driven by ds on seeds 0–2 (+1.86 ds, +0.16 dla) and by dla on seeds 3–5 (+0.27 ds, +0.81 dla) [F, history §4 item 11].

**Implication** [I]. Any claimed gain below about +0.5 pooled over six seeds cannot be told apart from noise. Many rounds of choices on the same 8–10 participants make selection scores optimistic. The repo's rule (≥ +0.5 pooled over six seeds, paired per-participant signs) is correct and should not be relaxed.

**Ensemble saturation.** Single-model gains do not survive the rebalanced ensemble: AdaBN +4.9, classic augmentation +3.3, T-TIME without rebalancing +2.5, rotation on learned-FIR nets +4.4 alone (history C5). The rotation precedent shows roughly 15–20% pass-through from single family to ensemble. A new member needs about +2 alone to be worth +0.3–0.5 in the ensemble.

### 2.3 Where the residual errors sit (R15 six-seed, diagnostics report)

- **Confusion** (dla, rows MI/SUB/WG): `[[66,16,18],[9,77,14],[12,15,73]]`. MI is weakest.
- **Systematic deficits by participant:**
  - P10 unseen: MI recall 23–33%, and only 13–15% of windows are predicted as MI even after rebalancing. As a seen participant P10 reaches 67–81%.
  - P04: SUB recall 35–40%.
  - P09 and P03: WG recall 50–53%.
  - P10's class contrast correlates −0.40 with the group; P05's −0.08. P10 is the only participant with reversed frontal-midline theta for MI (+0.84 against a −0.59 mean). P05 shows reversed MI laterality.
  - P04 is a weak-signal participant (own-data ceiling 0.58; 19 µV RMS against 27–46), not a transfer failure.
- **Errors are near-ties.** Confident-and-wrong windows (≥ 0.6) are 4.4% of windows. Pseudo-labels at ≥ 0.7 are clean but skewed against WG (17% of predicted labels). This explains why confident-only adaptation failed.
- **Diversity.** On 94.7% of windows at least one member is right; the ensemble gets 72.0%. Leave-one-family-out:
  - all EA members −3.92 dla;
  - v2 −1.11;
  - ShallowConvNet −1.63;
  - FBCNet-TF −0.19 and FBLight +0.21 (about zero on unseen participants).
- **Position.**
  - Rebalancing warm-up matters only for windows 0–10.
  - The real dip is the first 5 windows of a run after an interface switch: −4 to −6 points. Models with no adaptation state show it too.
  - Fully closing it is worth +0.6–0.9; realistically recoverable about 0.3.
- **Calibration.** After rebalancing the ensemble is under-confident (oracle temperature 0.34–0.64). Only label decisions are scored, so this matters only for downstream gating.
- **Interface.** For unseen participants the Graz/BrainHero gap is noise. For seen participants BrainHero is higher in all 4 cells (−9.6 ss).

### 2.4 Positive signal found this session: the per-session gaze view

**Setup.** Causal MAP template EM on 32 gaze-proxy features (horizontal (F5+AF3+F3)−(F6+AF4+F4) and vertical (AF3+AF4)−(P1+P2+Pz), 16 bins), combined with the R15 posterior as a product of experts. Results on six-seed R15 (`scratchpad/neuro2/refute_gaze.log`):

| Protocol | Base | Gaze gain | Shift-null mean / sd / max | Per participant |
|---|---|---|---|---|
| dla | 72.11 | **+1.46** | −0.42 / 0.30 / +0.15 | mean +1.44 (se 0.55); 6/8 ≥ 0 |
| al | 67.03 | +1.14 | −0.32 / 0.20 / +0.07 | P09 +4.1, P04 −0.9 |
| ds | 85.25 | +0.43 | −0.68 / 0.69 / +0.28 | single participant |
| ss (proxy mix) | 72.00 | −0.18 | −0.83 / 0.75 / +0.52 | P10 +1.5, P04 −1.9 |

**Ocular-event view.** Appending 8 causal, sign-free ocular-event rate features (`audit_ev2.log`) gives increments over gaze alone of:
- seeds 0–2: dla +1.00, ds +0.83;
- seeds 3–5: dla +0.53, ds +1.84.

ss is −0.29 against −0.18 for gaze alone. The ds increments sit within the noise-view sd of 1.1–1.3.

**Caveats** [F from critiques]:
- Winner's curse: nested config selection gives +0.33 (se 0.84) in one critique and +1.35 (se 1.08) in the other, on different config families. This is unresolved, and no nested log is saved.
- P04 loses under every config.
- The code shrinks class means toward the session mean (zero), not toward a population template as the candidate text claimed.
- The feature design drew on per-participant d′ computed over all 10 participants, including P04/P09. That is mild design leakage for al.

---

## 3. What the literature and other fields say

### 3.1 Cross-subject decoding

- **Foundation models do not beat specialists on cross-subject MI.**
  - A controlled evaluation has 0 of 12 MI cells beating five-band log-power (arXiv 2610.32067).
  - EEG-FM-Compass finds FMs "comparable" to EEGNet/ShallowConv under LOSO (arXiv 2601.17883).
  - Our CBraMod, LaBraM and REVE runs agree (`docs/OPTIMIZATION_R8.md`).
- **Honest DG gains on few-subject MI are +1 to +3.** SREA gets +1.1 on IV-2a LOSO (arXiv 2602.01728). MAML gets +3, but with 54 subjects. Under fair selection, ERM matches DANN, IRM, MixStyle and others (DomainBed, arXiv 2007.01434). Our R18 result (adversarial hurts, MixStyle neutral) is the expected outcome with 10 domains.
- **Data scale dominates.** Cross-subject accuracy keeps rising up to about 40 training subjects (AdaBrain-Bench, arXiv 2507.09882). We have 10, and no public 3-class PROTEUS-like corpus exists (`docs/OPTIMIZATION_R10.md` search).
- **Alignment.**
  - EA belongs after temporal filtering (Wu 2025, arXiv 2502.09203 §III-C).
  - A Riemannian-mean reference beats arithmetic EA by +1.0–1.4 cross-subject on IV-2a (Wimpff 2024, arXiv 2311.18520, Table II).
  - TSMNet with domain-specific SPD BN is the strongest label-free SPD approach (Kobler, NeurIPS 2022, arXiv 2206.01323). Its gains are measured against unaligned baselines.
  - CMMN spectral normalisation helps the worst subjects in sleep data (arXiv 2305.18831); there is no MI evidence.

### 3.2 Test-time adaptation

- Published EEG TTA gains are measured against EA baselines *without* class-bias correction:
  - T-TIME: +3–6 over online EA (arXiv 2412.07228).
  - Wimpff: BN adaptation +7, entropy adds +1–2.
- NeuroAdapt-Bench (arXiv 2604.16926) finds gradient TTA usually harmful on EEG: Tent −10 to −31, SHOT down to −42.
- T3A falls below the no-adaptation baseline on MI in two EEG papers.
- Our R4, R7, R8 and R12 results reproduce all of this. Once rebalancing has removed the bias, only participant-direction error is left, and source-anchored entropy or prototype methods amplify it.

### 3.3 Other fields

- **Speech recognition:** eigenvoices (Kuhn et al. 2000), MAP relevance adaptation (Reynolds et al. 2000) and LHUC (arXiv 1601.02828). Unsupervised adaptation works with fewer than 5 to a few hundred parameters, thousands of frames, and first-pass targets cleaned by a language model. We have about 190 windows and about 65% first-pass accuracy. The lesson: adapt very few parameters, use a prior, and use a view independent of the classifier. The gaze EM follows this; the eigen-participant variant on v2 features failed its labelled oracle (−0.8 to +0.3, only P10 up; `scratchpad/eigen/oracle2.log`).
- **Genomics harmonisation (ComBat):** equivalent to recentering plus AdaBN with a prior, which we already run.
- **DomainBed and Wiles et al. 2022 (arXiv 2110.11328):** only pretraining and augmentation help consistently. This matches R10/R14/R15 at about +0.2 each.
- **Co-training** (Blum & Mitchell 1998) needs views that are conditionally independent given the class. v2 and the nets have φ ≈ 0.24 overall and higher on MI, so cross-view pseudo-labelling is weak (`scratchpad/view_diag.py`). A physiologically separate ocular view is the closest thing we have to an independent view.
- **Competitions:**
  - BEETL 2021 winner (Cogitat): trained-in latent per-subject alignment (arXiv 2202.03267). Its baseline lacked EA, and our R7/R18 evidence argues against it here.
  - NER 2015 and Grasp-and-Lift: participant bagging, refuted locally (`scratchpad/bagens.py`).
  - DecMeg 2014: transductive clustering, refuted (oracle 0.37).
  - Several winners exploited test-set structure. Our legitimate equivalent (same-session stream, balanced runs) is already in use.
  - Grasp-and-Lift: "Safe" and "YOLO" entries finished about equal, which supports keeping a simple anchor.

### 3.4 How R15 compares with the state of the art [I]

R15 already contains every element the literature supports for zero-calibration cross-subject decoding:
- causal EA and Riemannian recentering;
- filter-bank tangent space (v2 is the strong shallow baseline TSMNet compares against);
- heterogeneous filter-bank convnets;
- label-shift correction;
- EA plus channel-template pretraining (the MIRepNet recipe);
- rotation augmentation.

Organiser baselines on the public cross-subject split are 33–36 (`OC/tracks/bci_decoding/README.md:55–70`).

Literature gaps where R15 departs from best practice:
1. EA is computed on broadband data. 85% of the trace sits below 4 Hz, and 8–30 Hz between-session dispersion barely moves (6.55 → 6.52; `scratchpad/ana/fbea.py`).
2. The EA reference is an arithmetic mean, not a Riemannian one.
3. No SPD-network member.
4. No spectral (CMMN) normalisation.

The one external warning sign: our honest warm-up replay scores 0.536, while plausibly honest entries reach 0.55–0.65. The board is 6 of 8 cells from P04, our weakest participant, so the board sd is roughly 2–3 points [I]. The gap is suggestive, not conclusive, and the board may not be used for selection.

---

## 4. Solution-space map

| Lever family | Status | Evidence | Remaining headroom [I] |
|---|---|---|---|
| Session input alignment (EA, recentering k, stretch, train-causal, RMS, label-shift-aware, per-interface) | Adopted; variants closed | +7 / +14.6; oracle +0.05 | Band-limited EA reference for nets: open, +0 to +0.6 |
| Spectral normalisation (CMMN) | Open, untested | Literature only (sleep) | +0 to +0.5; overlaps v2 per-band recentering and band-EA |
| Hand-crafted features (slow wave, bands) | Adopted; ocular, theta and laterality members closed (−0.27 to −2.4 as members) | `mixev.py` | Spent as supervised members |
| Architectures (about 30 braindecode, FIR/sinc front ends) | Adopted set; rest closed | R3/R7/R9 sweeps | Saturated; FBCNet-TF and FBLight slots worth about 0 |
| SPD network (TSMNet) | Open but unlikely | v2's value rests largely on the slow block, which a covariance net cannot carry | +0 to +0.1 |
| Re-referenced views (Laplacian/CAR) | Proposed and dropped | Absorbed by learned spatial filters; EA(Lx) = rotation | about 0 |
| Augmentation | Rotation adopted; classic, frequency and symmetric congruence closed or dropped | R3/R4/R14; R8 RMS | about 0 to +0.1 |
| DG objectives (DRO, balanced loss, SWAD, mixing, MixStyle, DANN) | Closed (R8, R11, R18) | All ≤ noise or negative | about 0 |
| Trained-in session normalisation (SessionNorm/ARM-BN) | Dropped | R7 AdaBN redundant with rebalancing; R18 flat | about 0 |
| Pretraining (m7, m7r; unified rest head; FMs) | SincShallow m7r adopted; R16 FBCNet +0.3 not adopted; R17 and FMs closed | `OPTIMIZATION_R10.md` | Possible portfolio arm only (R16 +0.3) |
| External data in the 3-class loss | Closed (licence or confounds) | Scherer +0.6 excluded | Blocked |
| Ensemble weighting, stacking, calibration, position schedule | Closed | Oracle per-session weighting fails across interfaces; position schedule −0.1 to −1.1 (`scratchpad/pos.py`) | about 0 |
| Output label-shift correction | Adopted; spent | Oracle bias +2, counts +0.4, Graz position prior ≤ +0.3 | about 0 |
| Pseudo-label / entropy / prototype / BN TTA | Closed | R4, R7, R8, R12; literature negative | about 0 |
| **Per-session low-dimensional independent-view template EM (gaze, ocular events)** | **Open, positive, not yet confirmed** | dla +1.46 (6 sd above null); with events +2.2 on seeds 3–5 | **+0.3 to +0.8 sealed** |
| Seen-participant routing (fingerprint, decodability, gaze library) | Closed | 0/4 correct at window 30, even with oracle labels | about 0 |
| Label noise / trial quality | Falsified | 2.8% low-p windows; MI is the *cleanest* class within session | about 0 |
| Participant bagging | Falsified by a proxy | Ensemble swaps −0.4 to +0.10 | about 0 |
| Eigen-participant MAP (v2 space) / LUPI pre-cue teacher | Dropped | Labelled oracle −0.8 to +0.3 / weak, possibly leaky teacher | about 0 |
| Sealed portfolio under Force_Best | Partly planned; current slots near-duplicates | R14 and R15 differ on 0/2,655 labels | +0.15 to +0.3 |
| Unused in-domain unlabelled PROTEUS EEG (rest, inter-trial, 5–8 s) | Untested for self-supervision | – | Unknown; low prior given R17/FM results |

---

## 5. Ranked plan

The decision rule is unchanged: select on ds + dla only, six seeds pooled for anything trained, paired per-participant signs, never tune on al, ss or the warm-up board. Today is 30 Oct. Warm-up closes **1 Nov 23:59 CET**, and that warm-up phase is the only test bed on the real worker. Sealed uploads run 5–21 Nov.

### Rank 1: Causal per-session gaze + ocular-event template EM (product of experts after rebalancing)

**Mechanism.**
- The same screen layout is used for everyone, so each class produces consistent gaze/ocular patterns. Their person-specific form is stable within a session.
- A 40-dimensional physiological view (32 gaze bins + 8 sign-free event rates) is estimated per session by soft, column-balanced EM. Class means are shrunk toward the session mean (m0=20) and covariances shrunk at 0.7.
- R15's posterior is multiplied by the template likelihood raised to λ·n/(n+60), with λ=0.5, starting at window 10.
- It is low-dimensional and comes from a view separate from the classifier. That is why it escapes the confirmation bias that sank R4/R8, and the shift null confirms the information is real.
- Use **one joint view** so R15's posterior is not counted twice. Correct the docs: the prior is the session mean, not a population template.

**Realistic gain.**
- Dev: unseen +1 to +2. Seen 0 ± 0.3 (the machinery alone costs −0.7/−0.8 under the shift null on ds/ss; real gaze beats the null by about 1.1/0.65).
- Sealed: ≈ 0.57 × (+0.6 to +1.2) + 0.43 × (−0.3 to +0.1) ≈ **+0.3 to +0.6** [I].
- p(sealed gain ≥ +0.3) ≈ 0.45 [I]. The two critiques put p at 0.40–0.55 and expected gain at +0.35–0.5.

**Cost and properties.** About 1–1.5 days, no training. numpy, deterministic, so it adds no audit seed variance. Under 1 s per session. Adds about 50–60 lines to `submission.py`; channel indices and training-set global MAD constants go in `weights.npz`.

**First experiment** (Mac, saved probabilities, about 10 min; pre-registered, no new knobs):
1. Freeze run2: λ 0.5, n0 60, m0 20, shrinkage 0.7, 2 iterations, start at window 10. Freeze event thresholds at 4 and 8× the running MAD (k=5) over 4 bins.
2. Resolve the winner's-curse disagreement. On a fixed pre-registered family of 10 configs, run nested leave-one-participant-out selection on dla. Report nested gain, config-average gain and frozen-config gain, and **save the log**.
3. For gaze32 and gaze+evC each: 20 circular-shift nulls and a 40-dim Gaussian null; seeds 0–2 and 3–5 separately; per-participant deltas.
4. Truncate sessions to their first 60/90 windows and report the gain. Add a length guard if short sessions lose.
5. Report ss for P04 and P10 separately.

**Accept** if all hold:
- nested dla gain ≥ +0.5 and above the shift-null maximum on both seed sets;
- per-participant dla mean ≥ +0.8 with t ≥ 2 and at least 6/8 non-negative;
- config-average ds ≥ −0.3;
- ss ≥ −0.5 with P04 ≥ −2.0;
- gaze+evC kept over gaze only if its increment is ≥ +0.4 sel on both seed sets *and* beats the best of three noise-padded controls.

**Stop rule.** Drop it if any seed set gives dla < +0.5 or seen-side mean(ds, ss) < −0.8.

**Packaging (same day if accepted).**
- Add the stage after rebalancing in `Ensemble.predict`, with per-session buffers cleared in `reset_state`.
- Extend `scripts/build_sealed.py` so the reference path includes the stage, and require harness agreement 1.0.
- Check causality: predictions up to a random cut point must be identical when the stream is truncated there.
- Run the worker replica and an A100 replay.

### Rank 2: Sealed portfolio rebuilt from decorrelated arms, plus a validated defensive input contract

**Mechanism.** No new information. It exploits Force_Best order statistics.
- The current user-approved order (r9g, R10, R14, R10b, R15; `docs/SEALED_PLAN.md`, decided 29 Oct) spends slots on near-duplicates: R14 and R15 differ on 0 of 2,655 windows, R10/R10b/R14 on 1–6.
- The realised E[max] − E[R15] of that order is about +0.1 [I].
- Replacing duplicate or dominated slots with arms that have near-equal means but different errors is the only real portfolio lever.

**Realistic gain.** +0.15 to +0.3. p ≈ 0.3. The gaze arm's own gain is counted under rank 1.

**Cost.** About 0.5 day of analysis plus about 0.5 day per new arm build.

**First experiment** (Mac, saved probabilities):
1. Candidate arms, already trained only: R15; R15+gaze(+ev); R15 with the R16 FBCNet ptm7r swap (six-seed +0.33); R15 with v2 weight 0.15; R15 α2/W10; an EA-only or raw-only subset; r9g/R10 for reference.
2. Hierarchical bootstrap: resample participants within an unseen stratum (dla folds, weight 0.57) and a seen stratum (ds, and ss in half the draws as a diagnostic, weight 0.43), and resample seed sets.
3. Compute E[max of 5] for the current order and for a greedy alternative chosen on ds + dla only.

**Proposal threshold.** Put a change to the user only if the alternative beats the current order by ≥ +0.15 in E[max], no replaced arm costs more than −0.4 in expectation against R15, and the sign holds on both seed sets.

**Defensive contract.** Covers:
- reorder channels by `meta["ch_names"]` (runtime values "may differ", `OC/codabench/pages/participate.md:210`);
- guard `n_times` and `sfreq`;
- `nan_to_num` (already present);
- no blanket try/except that hides bugs.

Changing `submission.py` invalidates the worker verification of every staged ZIP (all share hash 396e3866). It is therefore worth doing **only together with** the rank-1 stage, in one `submission.py` revision:
- unit tests on permuted channels, extra meta keys and wrong `n_times`;
- harness agreement 1.0 on all 2,655 windows against staged R15 with the stage disabled;
- with user authorisation, a warm-up upload before 1 Nov 23:59 CET as a crash test.

That warm-up number is P04-dominated, and gaze EM hurts P04 slightly. **Do not read the score as evidence.**

If this cannot be done by 1 Nov, the sealed arms that use the new code rely on the worker replica plus A100 replay. Weigh that against keeping one arm on the proven code.

### Rank 3: Band-limited causal EA reference for FBCNet-EA and FBLight-EA

**Mechanism.** The nets' EA reference is dominated by power below 4 Hz, so the 4–40 Hz bands they read get almost no between-session spatial alignment. A reference computed on a 4–40 Hz filtered copy, applied to the unfiltered input, gives band-relevant whitening while keeping the slow signal in the input.

A linear proxy supports it:
- non-causal tangent-space LR: no alignment 45.9, broadband EA 47.2, 4–40 Hz EA 50.4 (7/8 participants up), per-band 53.95;
- log-Euclidean mean adds only +0.25 (`scratchpad/proxy.py`).

**Realistic gain.** +0.2 to +0.6 ensemble, expected about +0.3. p(adopt) ≈ 0.2–0.3. Single-family gains pass through at about 15–20%, rotation already buys partial invariance, and v2 overlaps.

**Cost.** About 2 days: about 15 lines across `deep.py`, `build_sealed.py` and `submission.py` (a new `ref_band` key in `_align` and `_input_key`), plus Galvani.

**First experiment** (CPU, about 10 min):
1. Rerun the proxy with **causal** running references (k=5, training-mean prior, deployment order) on ds and dla.
2. Mix its probabilities 1:7 into saved R15 pooled probabilities, once with the 4–40 Hz reference and once with the broadband reference.

**Gate to Galvani.** Causal proxy keeps at least +2 over broadband with 6/8 participants up, and the mixed R15 delta (4–40 minus broadband) is ≥ 0.

**Galvani step.** fbcnet_ea_rot05 and fblight_ea_rot05 with ref_band 4–40, seeds 0–2, seed-paired against their rot05 twins:
- single-family dla ≥ +1.0;
- replacing them in R15 gives ≥ +0.3 sel with dla ≥ 0 and 5/8 participants up;
- then seeds 3–5, adopt at ≥ +0.4 pooled with dla ≥ +0.2.

Leave ShallowConvNet-EA and the pretrained SincShallow on broadband. Drop the standalone log-Euclidean arm.

**Stop rule.** Either seed set below +0.1, or the step-0 gate fails. If it is not decided by about 8 Nov it cannot feed a sealed upload; stop.

**Conditional follow-up.** Only if rank 3 shows that band misalignment matters: per-band EA inside FBCNet (about 2 more days), or causal CMMN as the spectral-only alternative. Neither starts otherwise.

### Cheap diagnostics worth running only because they cost minutes (no adoption expected)

- **Symmetric congruence / prefix-reference augmentation gate (inference only):** re-score saved R15 EA nets with an oracle whole-session reference. If the gain is < +0.5 sel, close the line permanently. The window-position analysis already shows the early-window EA-minus-raw deficit is about 0.
- **TSMNet falsifier:** split v2 into a covariance-only and a slow-only member (Galvani CPU, about 1 h). If v2-slow recovers ≥ 80% of v2's −1.11 leave-out value, close TSMNet.
- **Amplitude anomaly** (history C11: warm-up rises 49.7 → 55.0 as amplitude scale goes ×4 → ×0.25): diagnose which members drive it on saved members. Diagnosis only, never selection. It is the one hint about why other teams may do better on P04-like participants.

### Sealed-upload strategy (5 slots, max 1 per day, Force_Best, hidden logs)

**Preconditions for every upload.**
- Live check of phase, quota and dates first: the pinned yaml says 10 uploads, `docs/SEALED_PLAN.md` says 5.
- Explicit user authorisation per upload.
- Config and zip hash committed before upload (top-3 audit retrains from the committed config, ±2σ).
- Worker replica, harness agreement 1.0, A100 replay.

**Proposed order.** This is a proposal to the user; the recorded order is the user's decision.

1. **Slot 1 (5 Nov): R15 as staged**, or r9g if the user prefers the worker-proven weights first. R15 shares r9g's `submission.py` and has worker replica 0.9960. It banks a score and confirms the sealed adapter.
2. **Slot 2: R15 + gaze(+events) stage**, if rank 1 passes. Deterministic, unseen-leaning, no extra audit variance.
3. **Slot 3: the strongest decorrelated arm from the rank-2 bootstrap.** Candidates: R16 FBCNet swap, the band-EA build if it passed, or a v2-weight variant. Choose by E[max] on ds + dla, not on al or ss.
4. **Slot 4: combination of the winners so far.** Reuse the exact trained weights of the components; do not retrain with fresh seeds, which brings back ±0.3–0.4 seed noise. Combinations are not additive, so pick by local evidence, not by sealed arithmetic.
5. **Slot 5: reserve.** A safe repeat or the next-best decorrelated arm. Never two seed copies before the last slot.

**Reading sealed results.**
- Sealed differences are exact on the scoring set, and Force_Best keeps the max, so the board does the selection.
- A sealed score is one aggregate number and gives little to learn from. Do not chase knob changes in response to differences under about 0.3.
- A crash costs a slot with no log, so never upload an arm that has not passed the replica.

### What NOT to do

- No more architectures, DG objectives (DANN, MixStyle, DRO, SWAD), recipe sweeps or more seeds. All are closed with numbers.
- No pseudo-label, entropy, prototype or BN-statistics TTA, and no clustering on v2/PCA features. Closed, and the literature is negative after bias correction.
- No output-level knobs (α, W, T, member weights, position schedules, Graz position prior) as adoption candidates. Oracle bounds are spent and cross-interface selection loses.
- No seen-participant routing in any form (fingerprint, decodability, gaze library).
- No foundation-model fine-tuning, no R17-style rest-anchored heads, no eegmat-into-R17.
- No Laplacian/CAR streams, participant bagging, label-noise reweighting, SessionNorm, eigen-participant MAP, LUPI pre-cue teacher, or slow-PCA/theta EM views. All were dropped with evidence this session.
- No tuning on al, ss or the warm-up board.
- No `submission.py` change that is not validated on the worker before it is used.
- No local training on the Mac; only one capped job via `scripts/run_capped.sh` if anything must run locally.

---

## 6. Honest outlook on ranking

- **What we know** [F]:
  - Organiser baselines score 33–36 on the public cross-subject split.
  - Our cross-validated unseen performance is 67–72.
  - On the warm-up board our honest replay (0.536) sits *below* the plausibly honest cluster (0.55–0.65).
  - That board is 6 of 8 cells from P04, has an sd of roughly 2–3 points [I], and many entries were trained on P04/P09.
- **What we cannot know.** Competitors' sealed methods, and whether their warm-up edge comes from better handling of weak, low-amplitude participants, which would carry over to unseen sealed participants, or from noise or leakage.
- **R15 is near the literature frontier for zero-calibration decoding** [I]. Further gains in this window are about +0.5 to +1 in expectation, mostly from rank 1. The expected sealed movement from all remaining work, about +0.5, is below the participant-sampling noise of the sealed set itself.

**What would have to be true to be confidently top 3** [I]:
1. The sealed unseen participants resemble dev participants more than P04 (low amplitude, weak signal), or competitors are equally weak on P04-like people.
2. Competitors do not have a better seen-participant mechanism. Everyone can train on all public sessions, so this is mostly a level field.
3. No sealed crash, and the winning config reproduces within ±2σ in the audit. Deterministic stages and seed-pooled ensembles help here.
4. The gaze/ocular signal carries over to sealed sessions: same display layout, same acquisition programme. The FAQ says the same programme, but later sessions and new participants may use different strategies.

None of these can be verified before the sealed phase. Being in the top group is plausible. Being *confidently* top 3 is not supported by the evidence. The warm-up shortfall is the main argument for humility, and the 33–36 organiser baselines the main argument for optimism.

---

## 7. Completeness check

**Could not verify.**
- Sealed composition: number of sessions per participant, whether all six sessions are included, run structure and counts in sessions 4–6, interface order, display layout in later sessions.
- The live quota and dates (sources disagree).
- The origin of the gaze-like signal: ocular vs cortical. There are no EOG channels; AF3/AF4 are the most frontal electrodes. The display-geometry interpretation (MI left, WORD right, SUB up) is inferred, since the protocol paper is not public.
- The nested gaze-EM estimate: the critiques disagree (+0.33 vs +1.35) and no log is saved.
- ss for actual R15 rotation members: ss uses an R10-like proxy because no `*_rot05` or `ptm7r` ss files exist.
- Whether Dreyer 2023 (pretraining corpus, same lab) shares participants with PROTEUS, which would inflate dl for pretrained members.
- Primary write-ups of the HMS 1st place, the KUL report and the JLShen solution.
- Whether `fill_null` missing from `pretrain.load_corpus` matters for low-channel corpora (Galvani-only data).

**Missing modalities and data.**
- No EOG/EMG, eye tracking, handedness or questionnaire metadata (questionnaires withheld).
- Handedness cannot be inferred label-free: the laterality-skew test failed.
- Unused in-domain unlabelled PROTEUS EEG: rest runs, inter-trial periods, the 5–8 s trial tail, raw 500 Hz. Low prior given the R17/FM outcomes, but untested.

**Open questions.**
1. Does the gaze/event effect hold on later sessions (4–6) of seen participants, and on participants who move their eyes less?
2. Why does shrinking amplitude raise the P04-dominated warm-up score (C11)? Is there a member-level scale fragility that matters for unseen low-amplitude participants?
3. Is the early-run dip after an interface switch (−4 to −6 points, also in models without adaptation state) behavioural, in which case nothing can be done, or partly a reference effect?
4. Does a band-limited reference change the ensemble's EA members, or only their solo scores?
5. How large is our seen-participant performance on sessions 4–6 relative to the ds (85) vs ss (72) proxies? This is the largest unmeasured term in the sealed score (43% weight, roughly a 13-point bracket).

**Repository files referenced:**
- `docs/OPTIMIZATION.md`, `docs/OPTIMIZATION_R3.md`–`R8.md`, `docs/OPTIMIZATION_R10.md` (R16–R18 results at lines 369–425)
- `docs/STRATEGY_R16.md`, `docs/STRATEGY_REVIEW.md`, `docs/HANDOFF.md`, `docs/SEALED_PLAN.md`, `docs/COMPETITION.md`
- `experiments/deep.py`, `experiments/features.py`, `experiments/models.py`, `experiments/dg.py`, `experiments/pretrain.py`
- `submissions/ensemble/submission.py`, `scripts/build_sealed.py`
- `2026-competition/benchmark_utils/{streaming.py,metrics.py,base_solver.py}`, `2026-competition/codabench/pages/{scientific_faq_bci.md,participate.md,terms.md}`

**Scratchpad evidence** (all under `/private/tmp/claude-501/-Users-sebastianboehler--claude-spares-spare-Pdn7MS/f1aede87-a130-4f66-8c95-5c7dc01a3281/scratchpad/`):
- `neuro2/refute_gaze.log`, `neuro2/audit_ev2.log`, `neuro2/slowpca_audit.log`, `neuro2/refute_lib.py`, `neuro2/nested_gaze.py`
- `diag_pos.log`, `diag_mix.log`, `diag_run.log`, `diag_ws.log`
- `proxy.py`, `pos.py`, `bagens.py`, `xsess_gaze.py`, `eigen/oracle2.log`, `ana/fbea.py`, `posprior.py`, `view_diag.py`