# Research 2 (Sonnet) · Fri 20 Nov 2026 evening

Scope: RESEARCH2_BRIEF.md. About 30 web searches (extended mode) and 14 primary-page fetches, 18:00-19:00 CET window.
Prioritised June-Nov 2026 arXiv, NeurIPS-2026-linked and ICLR-2026/27 pages, plus speech, recommender, medical imaging,
wearable and BCI transfers. No labelled data, no personal or project info in queries. The ledger was grepped before
ranking (Gromov, Wasserstein, hubness, CSLS, fMLLR, hypernetwork: zero hits; Procrustes only as a labelled routed
estimate, section 104; error-weighted fusion tried and negative, section 372b; pseudo-label stabilisation F-C3 undecided).

## Bottom line

- The recent literature mostly confirms our failures: gradient or memory-based online TTA on EEG is unstable
  (NeuroAdapt-Bench 2604.16926: "gradient-based approaches particularly prone to heavy degradation, optimisation-free
  more stable"; sEMG session paper 2607.27568: BN re-estimation collapses a model, feature-statistic alignment is the
  only method that helps every subject). I rate MOPC (2610.30422, +2.2/+3.7 on BCI-IV-IIb/OpenBMI, 2-class, gradient
  TTA), NeuroOnline (2607.03925, foundation model, no numbers) and BayaHAR MAP-EM (2606.04798, needs class set known,
  same as our EM/TSEM) as not worth a slot.
- The one genuinely new family, from three papers of 5 Aug to 18 Nov 2026 (SCORE, CHORUS, SPDAlign), is **label-free
  estimation of a rigid shift+rotation that maps a new person's space onto a population reference, using a few
  confident landmarks, hubness correction and an identity-regularised orthogonal fit**. It targets exactly
  bottleneck 1 ("the signal holds each person's own class map"). It is not the same as our EM (full Gaussian
  re-fit) or EA (mean/whitening only; the orthogonal ambiguity left after EA is what these methods estimate).
- Honest size: nothing here is a sure +1 pp on sealed. Best stack of ranks 1+2 might be +0.4 to +0.9 pp; each alone
  +0.2 to +0.5. I expect most dev results inside the +-0.5 noise: record "undecided", do not close.
- Both 200-way retrieval papers have dozens of landmarks. With 3 classes the rotation is only identified in a
  low-dimensional readout space, so every rank-2/5 test starts with an ORACLE ceiling that costs minutes.

## Ranked ideas

### 1. Label-free per-session accuracy estimate as the adapt / freeze / blend gate (bottleneck 4, 3)

- **Mechanism.** Returning people: keeping adaptation on after recognition is +3.7 pp on far-from-ceiling cross-day
  people and negative on near-ceiling P04/P10 (lane R4). Today that is one global choice. Speech and vision practice
  asks per domain "is this model already good here?" without labels: ensemble-disagreement and confidence-based
  accuracy estimators (agreement-on-the-line; AETTA 2404.01351 for TTA; gradient/prediction-matrix variants
  2401.08909, ICML 2023 Deng). We have a 16-slot family, the enrolled-session personal head and the adapted head,
  so three cheap per-session signals exist: (a) member disagreement rate of the prefix, (b) agreement between the
  frozen personal head and the adapted head, (c) mean max-posterior margin. Gate: blend weight
  w = sigmoid(a + b * est_acc) on keep-adapt vs frozen, fitted on source people only.
- **Why it can beat what failed.** It adds no new member (the mixer cannot absorb it) and no new gradient step; it
  chooses between two measured arms and captures a gain that exists on 17/19 people today but is cancelled by P04/P10.
  Bottleneck 2 caveat: 89.8 % shared errors mean disagreement under-states error; I only need a monotone
  relation across sessions (the "on-the-line" property), not calibration.
- **Rules.** Per session, prefix only, reset between sessions; fitted constants are fixed parameters. OK.
  Runtime negligible (statistics already computed).
- **Size.** If the gate captures half of R4's +3.7 on the ~38 % of sealed cells that are one-session returning
  people: about +0.7 pp optimistic; honest +0.2 to +0.5. P(>= +0.3 sealed-equivalent) about 30 %.
  Unverified weighting assumption applies (0.2 P04 + 0.8 P10; 57/43 split): report next to J, labelled so.
- **Decisive test (CPU, 2-3 h, existing outputs, done by Sat 10:00).** Both R4 arms already produce per-window decisions
  for every NETBCI person (19), P04, P10 and the Scherer visits. Per session compute the three proxies at
  ages 20/40/80 windows. Step 1 (10 min): Spearman between proxy and the arm difference delta (adapt minus frozen)
  across all sessions, people-nested bootstrap. Step 2: leave-one-person-out fit of the gate, score gated minus
  best fixed arm on NETBCI, P04, P10 separately. Pre-registered rule: Spearman >= 0.5 and gated NETBCI >= +1.5 pp with
  P04 and P10 each >= -0.1 pp -> build into a full-chain run; Spearman < 0.3 -> record closed for these proxies only;
  between -> undecided, add proxies.

### 2. Identity-regularised rigid class-map alignment on the readout space, with mutual-NN / CSLS landmarks (bottleneck 1, 3)

- **Mechanism (SCORE 2608.19134, 19 Aug; CHORUS 2611.24364, 18 Nov; PRISM 2608.15037, 15 Aug, CIKM 2026).**
  Per session, in a small readout space (3 logits, or the 2-D simplex coordinates of the top member, or a 6-16 D
  tangent projection), (1) match mean/scale to the source population; (2) score prefix windows against the 3
  population class anchors with CSLS (hubness correction: penalise anchors that are near to everything, which is
  how class collapse enters); (3) keep only mutual-nearest-neighbour windows as landmarks, weighted by the
  top-1/top-2 margin; (4) fit an orthogonal map with penalty rho*||R-I|| (rho = 0.1 in both papers, alternating
  Hungarian / polar factor) and refit as the prefix grows (CHORUS has a causal streaming mode). Ablation in CHORUS:
  removing the population refinement costs 4.2 to 6.6 pp; SCORE: identity regularisation +2.25 pp and
  hubness correction alone +9.6 pp over cosine.
- **Why it can beat TSEM/EM.** EM re-fits means and covariances (many parameters, confirms its own collapsed
  clusters); this fits 3 rotation angles plus a shift, strongly pulled to identity, from only confident mutual
  pairs, and the anchors are population fixed parameters (not self-updated means). Closest failed relative: label-free
  in-session methods recovering 0-3 pp, hard pair TTA; this is a different parameterisation, not a new loss.
- **Rules.** In-session only, causal, reset; anchors from public people. OK. Runtime: milliseconds per refit.
- **Size.** If it recovers 2 pp on new people on top of EM: 0.57 x 2 x (share of sessions where rigid
  rotation is the right model) about +0.3 pp; P(>= +0.3 people-nested) about 20 %. The second-stage stage-removal
  evidence (EM/TSEM worth 2.3 pp) says the room left after EM is small.
- **Decisive test (Galvani CPU, 1-2 h).** Step 0 ORACLE ceiling, 10 min: with true labels, fit the best
  rotation+shift in the readout space per held person on cached chain logits (8 pinned shuffled orders, ten held
  people). If the oracle gain is < 3 pp, stop: the class map is not rigid in that space. Step 1 if >= 3 pp:
  label-free version after the existing TSEM output, same cached logits, zero-change check (rho -> infinity must
  reproduce stock), report P04 and P10 separately and the unverified-weighted J. Predict +0.3 pp [80% -0.3,+0.9].

### 3. A member trained on the chain's own out-of-person errors, then scored in the chain (bottleneck 2)

- **Mechanism.** Lane Z says: gains right where the mix is wrong pass and even amplify; ordinary member gains are
  redundant. So aim training at the complement: weight source windows by the full chain's leave-one-person-out error
  (boosting residual stack, 2211.15621; learning to complement / defer in the human-AI literature), or add a
  diversity term against the chain's logits (expert-divergence line: 2603.00054, ICLR 2026; 2611.17472 reports
  explicit expert-diversity maximisation helped 3/4 tasks, router-diversity none). The new member sees
  chain-hard windows only.
- **Why different from what failed.** Section 372b tested error-weighted FUSION (negative, -0.33 pp); the member
  itself was never trained toward the mixer's blind spots. Section 142 says true labels on low-confidence windows
  recover the all-label ceiling, so the blind spots are learnable given labels, but for new people the chain's errors
  are person-level class-map errors (unlearnable label-free): my prior is that the member learns person noise.
  This is the highest-risk of the six.
- **Rules.** Training-side only. OK. Runtime: one more small member (FBLight scale).
- **Size.** +0.1 to +0.3 pp; P(>= +0.3) about 15 %.
- **Decisive test (Galvani GPU, about 6 h).** Train an FBLight member, 2 seeds, with chain-LOPO error weights vs
  same-seed unweighted. Gate 1 before any insertion (1 h): "fixed-where-wrong" rate = fraction of the chain's
  held-person errors the member gets right, versus an unweighted member (lane Z metric). Only if the ratio is
  >= 1.5x run full-chain insertion under 8 shuffled orders.

### 4. Cross-person class-matched averaging of training examples (CHORUS "SCA") (bottleneck 1, 5)

- **Mechanism.** CHORUS adds, beside real recordings, examples that are averages of 2 or 4 OTHER real people's
  responses to the same stimulus, keeping the label; no new loss. Alone it lifted pre-calibration retrieval
  22.7 to 28.2 % and 25.2 to 36.6 % on two backbones: the encoder learns the population-shared part. Here:
  average same-class windows of 2-4 source people (after EA) in covariance / log-band-power space, not raw waveform
  (induced mental-task activity is not phase locked, waveform averages cancel it).
- **Versus virtual subjects (failed).** Virtual subjects were synthetic people; this uses only real people and
  pushes toward the shared class structure rather than adding identity variety. Still close enough that I expect
  "undecided".
- **Rules.** Training-side only. OK. No runtime change.
- **Size.** +0.2 +- 0.4 pp on held-new; P(>= +0.3) about 20 %.
- **Decisive test (GPU, 4-5 h).** One FBLight-class member trained with 30 % averaged examples vs none, held-person
  standalone on the ten held people (8 orders), then in-chain insertion if standalone >= +0.5 pp. Include the
  pair-only and quad-only arms (CHORUS mixes 1/2/4).

### 5. Rotation of the covariance after EA (SPDAlign, 2610.06315, 5 Oct; sensor-shift REFIT 2610.08991) (bottleneck 1)

- **Mechanism.** Euclidean alignment whitens each person to identity mean; what remains is an unknown orthogonal
  channel mixing from head geometry and electrode placement. SPDAlign corrects it by Wasserstein-Procrustes after mean
  alignment; REFIT does the analogous "recognise, fix, test" for wearable placement shift without labels. Our members
  take EA inputs, so the remaining rotation is unaddressed. Estimate it by matching the person's whitened
  band-power / log-variance feature cloud to the pooled source cloud (OT plus polar fit), restricted to a near-identity
  rotation (Cayley, ||A|| penalised) so channel locality is preserved.
- **Risk.** The rotation is not identified by the mean alone, and a 41x41 rotation estimated label-free from few
  windows is high variance. The abstract gives no numbers; I could not read the method section (fetch gave the abstract only).
- **Rules.** Per-session, prefix-only. OK. Cost: one small eigendecomposition per refit.
- **Size.** +0.2 pp +- 0.4; P(>= +0.3) 15 %.
- **Decisive test (CPU 1 h, oracle first).** Oracle: with labels, per held person fit the rotation that maximises
  one member's accuracy (CSP-like, Cayley, penalty). If oracle gain < 2 pp stop; otherwise try the label-free OT
  estimate on the same member under pinned shuffled orders, zero-change check at penalty -> infinity.

### 6. Speaker-code amortised adaptation by multi-hypothesis entropy minimisation (bottleneck 1, 5; low odds)

- **Mechanism.** ASR: a compact per-speaker vector fitted on unlabeled audio with an entropy loss plus multi-hypothesis
  term gave 20 % relative WER on 1 min of data (2506.10653). ZeroMAG (2610.03546, 2 Oct) generates adapter weights
  from unlabeled recordings with no target labels, +7.2 pp balanced accuracy over EEG-only, within 0.5 of supervised
  adaptation, but on a foundation model with weak baselines. Here: a 16-d code that FiLM-modulates one member, trained
  on episodes in which each source person is the pseudo-target, fitted per session by closed-form or 3-5 gradient
  steps on the code only.
- **Overlap with a measured failure.** "Learned in-context adaptation (+2 on a weak base, <= prefix BN)" is this
  family. The only reasons to retry: the code is tiny (not an in-context transformer) and ZeroMAG claims the
  function-constrained latent space is what makes it work. I would give it one slot only if ranks 1-2 are queued.
- **Rules.** Source-person episodes only; code is session state. OK. Runtime: small.
- **Size.** +0.1 +- 0.4 pp; P(>= +0.3) 10 %.
- **Decisive test (GPU, 8 h, probably misses Sat 10:00).** Record "not run" unless a GPU is idle; if run, compare
  against prefix-BN on the same member, held-new, 8 orders.

## Considered and not ranked

- Speaker-verification score normalisation and sequential gate: already the Opus blind-spot rank 1; I agree with it
  and with its +0.35 pp size. It is complementary to my rank 1 (that one decides adapt/freeze after recognition).
- Gromov-Wasserstein unsupervised alignment: MGMCL (2608.08440) fits GW on sources only, no target use; GW needs
  many points per class, which three classes at prefix age < 50 do not give. Skip.
- Recommender cold start (TEA, MWUF, MeLU line): meta-learned warm-up of embeddings from a handful of
  interactions; all need interactions with labels. The unlabeled analogue is rank 6. Wearable HAR BayaHR/BayaHAR:
  prior-prototype MAP-EM requires the class set known, equals our EM/TSEM.
- Medical imaging site adaptation (PRISM-like closed-form affine bias regression; Shape moments TTA): the batch
  level affine correction is what EA + SessionBN already do; the new part is the class-anchored rotation, rank 2.
- Gradient online TTA (MOPC 2610.30422, NeuroOnline 2607.03925, Tent/SHOT on foundation models): measured here and in
  2604.16926 as degrading strong systems.
- Foundation-model-based test-time or channel adaptation (2604.23091, TSFT OJEMB 2026 pool of models): foundation
  models failed here.
- NeurIPS 2026 Brain & Body challenge Track 02 is the same task family on Dreyer 2023 (EEGNet 58.6, REVE 68.0 on warm-up);
  final evaluation ends 21 Nov, no solution write-ups yet. Nothing to import.
- NeurIPS-2026 workshop and ICLR-2027 pages: accepted lists / reviews were not indexed in search. Opus's cited "ICLR 2027 BN study" I could not re-find; my search returned only an ICLR 2026 EEG TTA paper stating BN recalibration is
  shallow. I did not verify it.

## Honest ordering for Sat 10:00

1. Rank 1 first: uses existing outputs, three hours, one-shot decision.
2. Rank 2 oracle ceiling and rank 5 oracle ceiling in parallel (minutes each, CPU): they either open a
   direction with a stated size or close the whole rigid-rotation family for this 3-class readout.
3. Ranks 3 and 4 as one GPU wave only if GPU is free; rank 6 last.
All six: pre-register in LEDGER, report P04 and P10 separately, undecided inside +-0.5 across people.

## Sources (fetched pages unless noted)

SCORE arXiv 2608.19134 (19 Aug 2026); CHORUS arXiv 2611.24364 (18 Nov 2026; two fetches, read 100k of 178k chars);
PRISM arXiv 2608.15037 (15 Aug 2026); SPDAlign arXiv 2610.06315 (5 Oct 2026, abstract only); sEMG session
recognition arXiv 2607.27568 (30 Jul 2026); MOPC arXiv 2610.30422 (21 Oct 2026); NeuroOnline arXiv 2607.03925
(4 Jul 2026); aligned subject-specific encoders arXiv 2606.16462 (v3 15 Sep 2026); ZeroMAG arXiv 2610.03546 (2 Oct 2026);
NeuroAdapt-Bench arXiv 2604.16926 (abstract); expert diversity MoE router ensembling arXiv 2611.17472.
Search-snippet only (not fetched): HyWA 2510.12947, BayaHAR 2606.04798, AETTA 2404.01351, 2506.10653 (ASR speaker codes),
2603.00054, REFIT 2610.08991, Brain & Body challenge pages.
