# Panel answer: distribution shift and test-time adaptation (14 Nov 2026)

Read-only review. Numbers marked "checked" are from seconds-long snippets on `data/proteus_windows.npz` and
`runs/replay_data/bases.npz` (B2_pool, `all_loso`, person-equal cell balanced accuracy, whole-session balancing, not
causal). The stored `final` array scores 77.3 in my scorer against the 80.2 the ledger quotes for the v3 chain, so
read my stage-level numbers as relative, not as the chain's level.

## 0. The formulation, and where the system sits in it

Source-free, online, transductive classification of one new domain per episode, with an exactly known class marginal,
final predictions, and (if the shuffle is full) an exchangeable stream. The known-best method class for that is not
gradient test-time adaptation. It is prototype or mixture refinement in a fixed discriminative embedding, with a
frozen source classifier as the prior and a balanced assignment step (transductive few-shot work: soft k-means
refinement, TIM, Sinkhorn/PT-MAP; on the adaptation side T3A and LAME). The team's chain is a hand-built member of
that class, so the class is right. What is weak is what surrounds it:

- the templates live in unsupervised spaces (128 PCs, ocular features), not in a space chosen for post-adaptation
  separability;
- the posteriors the stages exchange are under-confident (checked: confidence 0.5–0.7 is 80.5 % correct, 0.7–0.9 is
  96.8 %; the best single temperature is about 0.5, NLL 0.724 to 0.635);
- the members are trained as ordinary classifiers and adapted afterwards;
- every constant is tuned on ten people.

## 1. Diagnosis

- **Adaptation error is nearly spent.** The chain (80 %) sits at the within-session supervised level the record
  reports for this feature space (81–82 %). Output-level session adaptation is saturated (checked): label-free
  balancing reaches the oracle per-session bias (net mix 67.2 to 72.0, oracle 72.5; full mix 70.8 to 74.5, oracle
  74.7), and an oracle per-session 3×3 output map fitted in-sample adds only about 1 more.
- **The label marginal is used up.** It is exact (checked: every online run 10/10/10, every Graz cell 32/32/32), but
  a remaining-quota prior needs true counts. Soft counts are off by sd 3.4 per class at session end (hard counts
  8.2), while the hypergeometric signal at window 170 is about 2. Per-cell use needs the interface, and the team's
  oracle there is at most +0.1. I expect no more than +0.2 from any further use of the marginal.
- **What remains** is single-trial error in this representation, the online warm-up (first fifth at 73 %, about 1.3
  points of the session mean) and low-signal people (P04).
- **Structurally out of reach:** information about hard windows. Their §142 shows the whole fine-tuning ceiling sits
  in the low-confidence half and every label-free source is equally wrong there. The cluster assumption fails
  (clustering 47 %), so entropy, graph and pseudo-label methods have nothing to hold on to. Only a better
  representation moves this, which is why "more people" is the one lever with an undiminished slope.

## 2. Five proposals, ranked

**1. Train the members inside the adaptation they are deployed with.**
- Mechanism: episodic batches drawn from one session of one training person, in random order. Forward with session
  statistics (SessionBN), shift the batch logits so the mean prediction is uniform (two free parameters,
  differentiable), then take cross-entropy. Optional second step: refine class prototypes on the batch embeddings
  and classify against them. This is meta-learning with a closed-form inner loop, the only kind that survives nine
  tasks.
- Why here: the nets carry a 5-point session bias that the chain removes anyway (checked), so today they spend
  capacity on something that is free at test time. SessionBN, the feature-level version of this idea, was one of the
  largest gains in the project.
- Gain: new +0.3 to +1.0, returning +0.2 to +0.5; confidence low to medium.
- Experiment: retrain one SessionBN family (12 folds) this way; under 4 GPU hours; score through the chain under the
  three shuffle schemes.
- Kill: the family's post-balance accuracy rises less than 1 point alone, or the chain gains less than +0.3 on two
  seed sets without P10.
- Notes: no run-time cost, order-free, drop-in weights.

**2. An external-population harness for everything that is not the classifier.**
- Mechanism: the stages, the fine-tuning loop, the gate and the personal path are class-agnostic algorithms. Run the
  same code cross-subject on corpora with balanced classes (PhysionetMI 109 people, Lee2019 54×2 sessions, Cho2017,
  BNCI2014-001, Schirrmeister, Scherer) and, for the returning path, on multi-session corpora (Lee2019,
  BNCI2014-001, BNCI2015-001, Stieger2021). Use it first as a kill filter, then to choose constants.
- Why here: ten people cannot rank changes below ±0.3 to ±1; hundreds can. It is also the honest answer to "more
  people when none exist": none for the classifier, hundreds for the adaptation rule, which carries most of the
  score. The returning path, 43 % of the sealed set, is currently validated on two people.
- Gain: indirect; it stops shipping noise. Direct gain from re-chosen constants and returning-path design: new 0 to
  +0.4, returning +0.3 to +1.0 per session; confidence medium that it changes at least one decision.
- Experiment: CPU only, tangent features plus logistic regression plus the stages on two corpora; one day.
- Kill: PROTEUS-tuned constants are also optimal there and no recorded decision flips.

**3. One transductive objective instead of four stages.**
- Mechanism: for every delivered window, q_i ∝ p_source(y|x_i)^(1/T) × Π_views N(z_i; μ_c, Σ)^λ. Refit from scratch
  on the whole prefix every k windows: a balanced (Sinkhorn) E-step, an M-step with the population class template as
  prior mean, calibrated inputs (T about 0.5). Use an 8- to 16-dimensional supervised subspace (start from their
  PLS-DA view, +0.45 to +0.6 on two bases) instead of 128 PCs, where 40 seeds per class cannot estimate a mean.
- Why here: permutation-invariant by construction, so it cannot lose to the unknown shuffle.
- Gain: new +0.3 to +0.8 under full or run-level shuffle, about 0 under within-run shuffle (their measured
  non-causal headroom is +1.4, +0.9 and +0.1); confidence low to medium. Expect robustness more than points.
- Experiment: CPU replay on stored features with nested people halves.
- Kill: under +0.3 against the re-tuned chain on every scheme.
- Notes: a free-form learned stage on nine people fails (theirs lost 8 points). Learn only the few scalars, and on
  the harness of proposal 2.

**4. Fine-tuning: change the loss, stop chasing labels.**
- The +1.9 is not on the table. It measures the value of 190 labels, not the cost of noise (§142, and the
  confident-half-only arm hurts).
- What is left: targets from the stage posterior, sharpened to calibration and kept soft; an information-maximisation
  loss with the known marginal (KL of the mean prediction to uniform plus conditional entropy), since entropy alone
  collapses, which is what their HENT arm showed; an L2 anchor to the source weights, adapting normalisation and the
  last block only; predictions from the weight-averaged student.
- Gain: +0.1 to +0.4 over gC5 for new people; confidence low.
- Experiment: three arms on the existing gC5 script, a few GPU hours.
- Kill: under +0.3 without P10 on two draws.

**5. Adaptation strength gated on label-free session confidence.**
- Mechanism: one monotone function of the mean top posterior at about window 40 scales λ and the fine-tuning rate.
- Why here: their record says session confidence tracks session accuracy (r about 0.8), fine-tuning gains sit in
  weak sessions and strong sessions lose. No policy was ever tested.
- Gain: new +0.2 to +0.5; confidence low to medium.
- Experiment: CPU replay on stored outputs; fit on one people half, score on the other, confirm on the harness.
- Kill: sign differs between halves.

**Expected to fail here:**
- TENT and other entropy-only objectives; label propagation on kNN graphs.
- CoTTA-style restoring (there is no continual shift; state resets every session).
- MAML or Reptile on network weights with nine tasks.
- IRM, GroupDRO, CORAL, MixStyle, synthetic participants (their results of at most 0 match honest domain
  generalisation benchmarks).
- Foundation models, frozen (their random-initialised REVE probe equals the pretrained one) or fully fine-tuned, in
  seven days.
- Pooling other-task corpora into training.

## 3. Contrarian claim

"The tangent-space model is the strongest member, and member accuracy barely moves the score" is a measurement
artefact. Raw accuracy mixes session bias, which the chain removes for free, with single-trial skill, which it
cannot. After label-free balancing the net mix is 72.0 and core1 is 69.0 (checked); raw, they are 67.2 and 69.3.
The two team claims also contradict each other: accuracy from more people passes through the chain at 0.95, accuracy
from member changes at 0.1 to 0.2. So members have been selected and trained on the wrong metric.

Test, CPU, minutes: recompute post-balance accuracy for every stored member variant and regress the recorded chain
deltas on raw and on post-balance accuracy. If post-balance predicts and raw does not, re-open the closed member
variants under the new metric and adopt proposal 1.

## 4. One question

To the organisers: is the delivery order a uniform permutation of the whole session with both interfaces mixed, or
are runs kept as blocks? Under a full shuffle the stream is exchangeable and proposal 3 has +1.4 of headroom; under
within-run shuffle it has none and proposal 1 is the only stage-independent lever. If they will not say, the
equivalent data check is the cheap regression of section 3.

## 5. The two uploads

- **Upload A (16–17 Nov):** v5.1b plus the measured combination (gC5 fine-tuning, eye view, rest member) on
  unrecognised sessions, recognised sessions exactly as v5.1b. It is the only candidate replicated on every shuffle
  scheme, and it measures the new-people transfer ratio, which has never been observed: both earlier ratios (0.41,
  0.46) came from returning-person levers scored on P04. Plan with sealed ≈ 0.57 × Δnew × 0.6 to 0.8, about +0.3 to
  +0.5. Preconditions: a worker-image replay and a run-time margin at 380 windows.
- **Upload B (20–21 Nov):** A plus only what passed on PROTEUS and on the external harness. Retrained member families
  from proposal 1 are drop-in and the likeliest content; the confidence gate and a returning-path change are next. No
  redesign that has not been replayed on the worker image.
- **Honest outlook:** A lands at 0.822 to 0.825, B at 0.823 to 0.829; reaching 0.831 has perhaps a 10 to 15 % chance.

**Validation rules I would hold to:** one primary endpoint (new people without P10, mean over the three shuffle
schemes, person-paired interval); nested people halves for every chosen constant; replication on a second draw and
on the harness; halve any gain that was the best of many arms.
