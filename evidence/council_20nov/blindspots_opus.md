# Council · outward blind-spot review (Opus) · Fri 20 Nov 2026

Scope: look outside our framing (cross-subject classification + domain adaptation + ensembles + returning routing) for
algorithm families, neighbouring fields and recent work we never used. The internal skeptic covers measurement errors.
Search window: 09:27 to about 10:00 CET. Sources: arXiv API sorted by date, OpenReview ICLR 2027 search, Hugging Face
models/datasets sorted by lastModified, GitHub search, OpenNeuro GraphQL, plus web search for neuroscience and speech.
Three local CPU diagnostics on `data/proteus_windows.npz` (2,655 public windows). They were not pre-registered: read
them as diagnostics, not results. Labels were used only for scoring.

## Bottom line

- The outside literature agrees with our evidence more than it contradicts it. Recent work says label-free cross-subject
  gains are mostly BN-statistic recalibration (ICLR 2027 BN study; LSTA), that self-supervised EEG FMs encode identity
  rather than task ("Subject Identity as a Shortcut"), and that for these three tasks the useful patterns are
  person-specific (Friedrich, Scherer and Neuper 2013). None of that offers a label-free way to the person's own
  class map.
- I found **no outside method with a credible ≥1 pp sealed gain by Saturday**. Best guess for the best case: +0.3 to
  +0.6 pp, and only from item 1, which sits in the part of the sealed set that was never measured.
- One framing blind spot remains. We treat recognition of returning people as solved: it is a fixed-threshold gate on
  one enrolled session. Speaker verification treats this as the core problem (score normalisation, sequential tests,
  enrolment from one session). About 38 % of sealed cells (the eight one-session public people) depend on it, and it was
  never measured on cross-day data like theirs.
- I tested a labelling search (TURTLE/HUME style), the most direct outside answer to "find the person's own classes".
  It fails even with oracle naming (section 3). Set it aside.

## 1. Framing critique

| Our framing | Outside view | Consequence |
|---|---|---|
| New-person decoding is the problem and returning people are near ceiling | P04/P10 have 3 sessions and many enrolment windows. The other eight returning people have one public session, and their visits are cross-day. In EEG biometrics, cross-session identification is reliably worse than within-session (Durham thesis; impostor-attack studies, 3–4 % FAR even with many sessions). | The sealed returning arm is unmeasured where it matters. Recognition is where the error sits there, not decoding. |
| Errors are person-level and need features | Friedrich et al. (2013): the best task combination changes per person and even per session. Scherer et al. (2015, PLOS ONE): 9 users, same task family. | The person-level error is a known property of brain-teaser tasks. No architecture fixes it without person data. |
| More source people = coverage | B5: the full system is flat in N for one family | Coverage is no longer the lever for the system. Matches the outside view that label-free cross-subject decoding caps out. |
| Adaptation means updating the model | Speech separates three things: speaker normalisation (VTLN, CMVN), speaker-adaptive training (SAT) and speaker identification (i-/x-vectors). | We have CMVN/SAT (EA, SessionBN) and identification. We do not have warp-style normalisation or proper verification calibration. Items 1 and 4. |

## 2. Ranked blind spots (top 5)

### 1. Open-set recognition as speaker verification: cohort score normalisation and sequential decisions

- **What.** Treat the identity gate as a verification system. (a) AS-norm/S-norm: normalise each probe-vs-gallery score
  by the mean and SD of that probe's scores against an impostor cohort, and of the gallery identity's scores against
  that cohort. The cohort is external people's sessions, shipped as fixed vectors. One threshold then holds across
  people and across days. (b) A sequential probability ratio test (Wald) on accumulated window scores replaces the
  fixed "first check at age 10, threshold .26". This lowers recognition age at a fixed false-accept rate.
- **Why it might work where lanes failed.** Every decoding lane is absorbed by the chain (89.8 % shared errors), but the
  personal route is worth +12 pp (P10) to +40 pp (P04) per routed session. Our own numbers show the gate is the weak
  link. External benchmark: 86.7 % true accepts at 1.9 % false accepts, mostly with m = 1 enrolment, so about 13 % of
  one-session returners fall to the new-person path. The Scherer gallery routes 144/144 impostor sessions, so the
  threshold does not transfer across montages and days. P04/P10 recognise at age 10–60, and windows before that run
  on the new-person path (P04 held-new ≈ 50 %).
- **Rules.** Allowed: in-session statistics only; the cohort is fixed parameters from public data; reset per session.
  Runtime is negligible (score vectors against roughly 200 cohort entries per check).
- **Size (honest).** Suppose true accepts rise from 86.7 % to about 95 % and the routed gain is about +10–15 pp per
  session. That gives 0.08 × 0.38 × 12 ≈ **+0.35 pp sealed**. Earlier recognition might add +0.1. P(≥ +0.2) ≈ 25 %:
  the 86.7 % may not describe PROTEUS cross-day returners, which could already be close to 100 %. Downside: if
  normalisation raises false accepts on new people, a wrong personal head costs roughly the FT and CSP value, −0.5 to −0.9
  pp on a misrouted session. The test must measure both sides.
- **Decisive test (Galvani CPU, 2–3 h, before Sat 10:00).** Use the multi-day corpora already staged at 40–41/41
  electrodes: NETBCI (19 people × 4 days) and ds007221 (two environments). Enrol day 1 with the exact package gallery
  construction (m = 1), stream day 2 through the package's own fingerprint path, and use all other people as
  impostors. The gallery holds the 10 public people plus the external enrolees. Report TA, FA and recognition age for
  the stock gate against AS-norm and against SPRT, at matched FA. Pre-register the rule: adopt only if TA rises ≥ 5
  points at FA ≤ the stock FA, and P04/P10 recognition age does not get later. If the gate is not separable from the
  package in time, report the number and leave it for the write-up.

### 2. Handedness and hemispheric reflection (neuroscience plus ICLR 2027 "SCoRE")

- **What.** PROTEUS asks for imagery of the *non-dominant* hand. A left-handed person imagines the right hand, so their
  MI pattern mirrors the population's. About 10 % of people are left-handed, so expect about 1 of the 10 new people.
  Language lateralisation (WORD) mostly does not flip (about 70 % of left-handers stay left-dominant), so a mirrored
  person's MI moves toward WORD's left-hemisphere pattern. Handedness is not in `participants.tsv` (age only).
- **Diagnostic I ran.** Mu (8–13 Hz) lateralisation of MI against the other tasks, averaged over C3/C4, CP3/CP4, C1/C2,
  FC3/FC4 and C5/C6. A negative t means right-hemisphere ERD, as expected for left-hand imagery.
  P08 t = −9.3, P09 −8.3, P10 −4.2, P01 −2.1, P07 −1.4, P06 −0.8, P03 −0.6, P04 −0.2, P02 +0.5, **P05 +1.95**
  (positive in Graz and in BrainHero, all five pairs positive). The strongly lateralised people are the accurate ones.
  P05 (70.6 %, one of the weak five) is the only one with a mirrored sign. It is weak evidence, not a finding.
- **Why it might work.** It is the only person-level error mechanism with a known physiological cause and a label-free,
  two-hypothesis test: orientation is one bit per session, which is far easier to infer than a full class map.
- **Rules.** Allowed (a per-session label-free choice). Cost: MI-sensitive members run on two inputs. With about 389 s
  of margin, use only the cheap FBC members.
- **Size.** At most about one new person, partly. ≤ **+0.3 pp sealed**, likely 0 to +0.15.
- **Decisive test (Galvani, 2–3 h).** Stream the held-new banks of cap250's raw FBC families on homologous-channel-
  swapped input (fixed 41-channel permutation; midline channels stay). Per person, compare mirrored against original BA.
  Ceiling: does P05 gain ≥ 5 pp while P08/P09/P10 lose? If so, register a label-free selector: after 40 windows, pick
  the orientation with the higher mean top-class margin of the MI-sensitive members. It must pick "original" for ≥ 9
  of 10 people before any chain arm runs. If the P05 ceiling is below +3 pp, stop.

### 3. Competition meta-strategy under Force_Best and free Failed runs

- **Decision rule.** Only the best sealed score counts, so the right objective is E[max(S, 0.8299)], not E[S]. Between
  candidates with similar means, prefer the one with more variance. A do-no-harm reading is wrong here, as Fable and
  Opus also note. The dev deltas over cap250 are hardcb +0.39 [+0.28, +0.52] and hardcb with heavy TTT +0.73
  (lane T, measured in the chain, two orders). Heavy does not fit 3,600 s, so this rule does not overturn hardcb today. It does
  say that a fitting adaptation arm with positive mean and wide interval (TTT-light, lite5, sparse20) should not be
  vetoed for its width.
- **Unknown.** Does a run that exceeds 3,600 s count as `Failed` (free, retryable) or as finished? Codabench usually
  marks an exceeded execution time as Failed, but our rules note does not say. If it counts as Failed, a runtime
  overrun costs only time, and the self-imposed 3,060/3,300 s margins could be traded for expected gain. Upload early
  enough Saturday to retry.
- **Not allowed, stated so nobody tries.** Using deliberately failing runs to probe the sealed set (for example,
  "fail if property X of the windows holds") extracts test information. Out.

### 4. Speech-style spectral normalisation: speed perturbation and per-session warp (VTLN analogue)

- **What.** ASR's most robust augmentation is speed perturbation (time-resampling by 0.9/1.0/1.1). Its matching test-time
  method is VTLN: choose one warp factor per speaker label-free, by maximum likelihood under the source model. EEG
  analogue: resample every training window by a factor in [0.88, 1.12], which shifts all rhythms together. At test time,
  pick the factor per session from the prefix, by maximum source-model log-likelihood or minimum entropy over 5 factors.
- **Why.** Individual alpha and mu peaks vary from 8 to 13 Hz across people, but our filter banks are fixed. In my
  diagnostic, the label-free posterior alpha peak (IAF) correlates with person accuracy at Spearman −0.63 (p = 0.05,
  n = 10). Codex council item 4 proposes a parallel IAF band-power head. This is a different form: it warps the whole
  input for every member, and on the training side it builds invariance. Lane V tried only spatial transforms, and no
  frequency or time-warp augmentation appears in the ledger or the code.
- **Rules.** Allowed. Cost: test-time selection needs 5 forwards per window for the members it touches. Only the
  training-side form is cheap.
- **Size.** Absorption applies (member changes transfer about ×0.1), so ≤ +0.2 pp in-system. This is post-deadline
  unless the ceiling below is large.
- **Decisive ceiling (Galvani, 1–2 h).** Run cap250's raw FBC banks on held-new windows warped by each of 5 fixed
  factors. Pick the per-person best factor with labels as an oracle. If the oracle-best factor gains < +1 pp standalone
  on the weak five, stop.

### 5. Meta-learned "where to adapt" (LSTA, ICLR 2027; ARM-style)

- **What.** Meta-learn, from training people, a low-rank subspace of normalisation-parameter updates. The inner step is
  label-free entropy on ≤ 8 windows; the outer loss is supervised on the same person's later windows. At test time,
  adaptation can only move inside that subspace.
- **Why.** Our hand-designed TTT helps weak bases (+2.95 pp) and P04 (+1.77 heavy in lane T), but hurts under drift
  (−3.0 on Scherer). It is flat on average in cap250, with per-person spread −1.1 to +2.7. A learned subspace is the
  outside answer to "adapt only along directions that historically helped". I checked whether a physiological predictor
  could gate heavy TTT instead: the Blankertz SMR-peak predictor correlates with person accuracy (Spearman 0.64), but
  not with the per-person heavy-TTT gain (Spearman −0.21). A hand gate is not supported.
- **Size and timing.** Possibly +0.2 to +0.5 in-system, but it needs nested meta-training over people. **Not by
  Saturday.** Post-deadline, it is the most promising adaptation direction.

## 3. Checked and set aside (with the reason)

| Idea (field) | Why not, with evidence |
|---|---|
| Unsupervised labelling search, TURTLE (ICML 2024) / HUME-style (vision) | **My ceiling, non-causal, whole session.** Three views (EA-tangent 8–30 Hz, EA-tangent 4–8 Hz, log band power; PCA 20 each), balanced co-labelling with 12 restarts, labelling chosen by the label-free cross-view fit, classes then *named by an oracle* (Hungarian on the true labels). Selected 45.7 % mean over 14 sessions; best-of-12 56.5 %; own-label 5-fold CV in the same views 63.6 %; cap250 74–76 %. Even P09, the best case, is 58/71 % (selected/best). The class structure cannot be found label-free in generic views, consistent with team B (vMF), Lane D and Ward ARI 0.05. |
| TabPFN-3.5 / TS2TabPFN / TSPFN in-context classifiers (tabular FMs) | Its natural use is as a cross-person member on covariance features, and members are absorbed (×0.1). The simple-feature base is 50 %. Legality of shipping a labelled context is open (ledger TabICL note). The package is not in the worker image, and its licence would need checking. Expected ≈ 0. |
| Generic time-series FMs (MOMENT, Chronos, Mantis V2) | Controlled comparison 2606.30104: on MI, simple temporal extractors match frozen MOMENT. Our EEG-FM members already reach only 55–59 % and are absorbed. |
| Vision FM as EEG encoder (SeeEEG 2611.17031) | Frozen-encoder regime with ≤ 1,113 display parameters. It beats EEG FMs only when both are frozen. Already listed by research_codex. Same absorption. |
| Population relay, CHORUS 2611.24364 (shift + rotation to population anchors) | It is Procrustes to predicted anchors. Our §104 pseudo-label rotation was −2.6 to −3.3; causal true-label means gave only 64.6 → 68.7. |
| Label-shift-aware Riemannian SFDA (SPDIM, OSPDIM 2608.05315) | Under shuffled delivery the prefix is near-balanced; whole-session balancing is ≤ +0.6 non-causal. |
| Online Riemannian hypergraph TTA (MRieHy 2608.16134), MOPC, GeMoT, FUSED | These are cross-day or same-family forms of pseudo-label TTA and BN or prototype adaptation, which are saturated inside cap250 (TTT +0.08/+0.34). |
| Hyperalignment / shared response model (fMRI, MEG, ECoG) | Needs time-locked shared stimuli across people. The class-conditional version needs labels (RPA). |
| Source imaging, template lead fields, Laplacian (physics prior) | The person-level error is a different *strategy* (Friedrich), not head geometry. GRAFT is already in research_codex. |
| EOG/EMG as signal (artefact physiology) | Covered: §421 (EOG carries WG information in Shin), §413 (34–45 Hz regional contamination), and a 30–55 Hz band is already in use. The 0.1 Hz high-pass keeps slow eye drifts for every member. |
| Cluster or person experts, cross-family fusion | Already measured: M2 refuted; team C fusion +0.3 at best and a learned gate −1.2 to −1.8 (Fable council §1). |
| Gating heavy TTT by an SMR predictor (physiology) | My check: Spearman −0.21 between the SMR-peak predictor and the per-person heavy-TTT delta (low-SMR mean +0.30, n = 6; high-SMR −0.01, n = 4). Post hoc, n = 10, not a basis. |

## 4. Cheap guard from the 13 Nov paper (2611.17925, "What Balanced Accuracy Cannot Identify ... under TTA")

Under TTA, a class can stop being predicted in a subject cell. That happened in 26–47 of 112 cells against 7–9 for
controls, and an aggregate BA change hides it. Our metric is per-cell BA, so a dropped class costs 33 pp of recall in
that cell. Before the final upload, count the cells where any class has ≤ 2 predictions, for hardcb against cap250
on full0/within0 and returning. Use existing banks: 10 minutes on CPU. If FT or TTT raises the count, this is a
specific and fixable failure. (Handing this to the internal skeptic.)

## 5. Diagnostics run for this review (local CPU, not committed)

- Lateralisation: Welch log mu power, MI minus other tasks, right minus left over five homologous pairs, per-person
  t-test (numbers in item 2).
- Labelling-search ceiling: section 3 row 1. Per session: EA on the session mean (non-causal), OAS covariance, tangent
  space, PCA 20. LR C = 0.1 with 5-fold cross-fitted probabilities, balanced Hungarian reassignment, ≤ 8 iterations,
  12 restarts.
- Physiology: label-free 1/f-detrended spectral peak over C3/C4/CP3/CP4/C1/C2 (SMR) and posterior alpha, against
  Fable's per-person accuracy. SMR Spearman +0.64 (p = 0.048), IAF −0.63 (p = 0.050). Against lane T per-person
  heavy-TTT deltas: −0.21.

## 6. Sources (primary, read in this window)

- arXiv 2611.17925 (BA under TTA, 13 Nov); 2611.17031 SeeEEG; 2611.24364 CHORUS; 2611.20164 LSIM MAP priors;
  2608.05315 OSPDIM; 2411.07249 SPDIM; 2608.16134 MRieHy; 2610.30422 MOPC; 2605.00857 FUSED; 2606.30104 TSFM vs
  EEG-FM; 2602.17868 MantisV2; 2609.17895 TabPFN-3.5; 2608.31013 TSPFN; 2608.04174 TS2TabPFN; 2406.07236 TURTLE.
- OpenReview ICLR 2027 (search "EEG test-time adaptation cross-subject"): SCoRE (forum XwbL97Ct2Z), BN statistics
  for SF cross-subject MI (bEEFURhzwK), GeMoT (VUmnbLJlnT), SREA shared/routed experts (NgMQZYWPzL), Subject Identity
  as a Shortcut (kX8lIY8Hsz), LSTA (zqPPaYF6kt).
- Friedrich, Scherer & Neuper 2013 (long-term 4-class imagery BCI; person-specific task selection); Scherer et al. 2015
  PLOS ONE; Blankertz et al. 2010 SMR predictor; EEG biometrics cross-session and open-set FAR (Durham thesis; PMC5855894;
  PMC9963573); MathWorks i-vector score normalisation (S-norm, cohort).
- Competitor public write-ups (GitHub): nwaehner/proteus-track2-results (REVE + LoRA, 0.577 unseen),
  dpny518/neural-interfaces-2026-track02 (57.7 % LOSO). Both are far below us, with nothing to adopt.
- Hugging Face: EEGDash/nm000290 (13 Nov) is the same 10 PROTEUS people. No new same-task EEG corpus on HF or
  OpenNeuro in the last month (checked newest 100 OpenNeuro datasets).
