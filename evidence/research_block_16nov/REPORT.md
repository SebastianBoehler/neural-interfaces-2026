# Research block: reopen the solution space

16 November 2026. Research ran from **16:54:42 to 17:24:48 CET**, an observed wall-clock duration of **30 min 6 s**.

**Decision:** expand the next research wave across different representations and learning algorithms. More variants of the current posterior ensemble alone do not cover the allowed space. DEFT-style task/person factorization, learned geometric alignment, and training a model for its causal adaptation are distinct, credible hypotheses. None currently has a measured PROTEUS full-system gain or establishes top-five performance.

This block produced research notes only. It did not change evaluation, model code, the official submodule, or pinned packages. It did not train, score, download datasets/weights, or submit new jobs. Implementation and experiment choices remain for the user's review. New compute, once agreed, belongs on Galvani.

## Reading guide

- [Allowed space and actual signal path](space_and_pipeline.md): current FAQ, captured Discord clarifications, code-path evidence, exact coverage and confounds.
- [Recent EEG methods](methods_recent.md): primary papers, dates, protocols, controls, release code, licenses and runtime limits.
- [Data and interdisciplinary transfer](data_and_transfer.md): corrected corpus inventory, what was already tried, task mismatch and controls.
- [Cross-field mechanisms](cross_field.md): train-for-adaptation, online geometric bias correction, tabular foundation models and adaptation monitoring.
- [Additional November primary checks](november_checks.md): trait/state hierarchy and scaled LaBraM, including normalization and selection caveats.

The review used current primary literature and official project sources. Discord evidence is the existing project capture, not a fresh independent reading of the private channel. Current FAQ and primary method sources were checked on 16 November. A 30-minute review cannot establish exhaustive coverage of every possible algorithm; this map makes the search boundaries and gaps explicit.

## 1. The actual problem, from the input upward

We must classify **motor imagery, subtraction, or word generation** from one 41-channel, 480-sample EEG window. The signal is four seconds at 120 Hz, beginning one second after the cue. The official pipeline filters continuous 500 Hz EEG, resamples it, then extracts the window. The competition supplies this tensor, not raw recordings, a separate evaluation-session rest recording, separate EOG channels, labels or interface metadata. Ocular activity can remain in the EEG. [Official FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md), [configuration](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/eeg/_motor_imagery_stream/config.yaml).

A useful statistical target is `P(class | current window, earlier delivered windows in this session, learned public-data parameters)`. The hidden causes include task, person, session, interface, recording setup and artifacts. Their effects need not separate cleanly. A model that removes every person or low-frequency signal can remove useful task information too.

**Architecture judgment:** the data motivates a spatial-and-temporal encoder that retains several frequency/time scales, followed by a small session conditioner and task classifier. Task and person/style information can have separate uses, rather than forcing a single identity-free representation. The session conditioner can summarize an unordered delivered prefix; it need not model recording-time drift. Covariance geometry and learned raw-signal encoders are two concrete realizations, not requirements to combine into one large model. This is a design hypothesis from the task and evidence, not a validated best architecture. The failed hand-selected regional hierarchy does not justify discarding the other electrodes or scales.

The causal boundary is **between delivered windows**, not inside the current four-second window. A bidirectional encoder within that window is allowed. A session model may use the delivered prefix, but must not assume delivery follows recording chronology. It cannot revise labels already emitted, inspect later windows, or carry learned session state into another session. Graz and BrainHero share one session state. Public labels and metadata can supervise offline training; inference adaptation remains label-free and session-local. Fixed labeled-example contexts such as TabICL are not explicitly adjudicated by the current clarification; do not assume they are permitted.

Balanced accuracy is computed separately for participant × session × interface cells and then averaged equally. Training loss, source sampling and model selection should reflect this objective. The hidden mixture of new and returning people is private. The old 57% mixture and P04/P10 weights are planning scenarios, not facts about the sealed cohort. Existing development selection has repeatedly used all ten public people; person-disjoint fits do not make the selected result a fresh independent estimate.

### What the processing preserves or removes

| Observation | Consequence for architecture and interpretation |
|---|---|
| Source filtering precedes segmentation. The inspected public NeuralBench/MNE 1.13.2 path gives a long, zero-phase low-frequency filter; the exact sealed-worker build is unverified. | Four-second crops may contain filtered context from outside their nominal interval. This is the supplied signal, not proof of label leakage. Neighboring crops/trials are not necessarily independent. Separate people, trials and filter support in diagnostic splits. |
| The installed target loader has no interpolation call; external exports do interpolate channels. | Do not explain a target result by an unverified target interpolation artifact. External and target preprocessing can still differ. |
| Covariance features demean each channel within a window. | They remove channel means. Slow/mean-preserving branches can contain information the covariance branch cannot recover. A historical mean-only baseline reached about 46%; that is diagnostic, not proof of neural origin. |
| CAR, centering, whitening and normalization change which amplitude and common-mode signals survive. | Treat them as competing modeling assumptions. Preserve matched raw/mean controls rather than treating more normalization as automatically better. |
| §222's tested alpha-band features beat its lower bands, despite a variance panel emphasizing below 8 Hz. | Low-frequency variance is not class discriminability. A low-frequency-only redesign is not supported. |
| The target is sampled at 120 Hz. | Upsampling to 200 Hz can match an encoder interface; it cannot restore frequencies above 60 Hz. Raw public recordings can support offline training or distillation, but are not an extra runtime input. |
| Prefix references mix delivered classes and nuisance variation. | A changing class mixture can move the reference without sensor drift. Posterior balancing cannot generally undo upstream feature distortion. This is a hypothesis to diagnose, not an established current bottleneck. |

## 2. Allowed space versus measured space

The incumbent is a comparison, not a challenge requirement. A whole alternative can replace its members, averaging, features, templates, routing and adaptation. The table distinguishes the tested forms from broader families; it does not assign universal closures.

| Allowed axis | Actual coverage | Material gap or limit |
|---|---|---|
| Static linear/nonlinear decoders | Strong tangent-space core; many compact CNN families; old compressed whole networks. | A jointly learned whole representation/classifier is not covered by testing its probabilities in the old ensemble. Compact nonlinear/table priors remain different classifier hypotheses. |
| Within-window representation | Dense band covariances and CNNs; named frozen and fine-tuned EEG foundation models. | Task/person factorization, cross-modal encoders and learned geometric references are not all represented. New slow, spectral and spatial views need matched signal-preserving controls. |
| Whole geometric systems | Strong causal centering; older Procrustes forms; simplified TSMNet as a base member. | Learned SPD bias/dispersion, anchors and complete geometry-plus-decoder systems differ. Offline target-batch methods require a new causal form. |
| Training objectives | Supervision, CORAL, adversaries, augmentation, distillation, SSL and private source heads all tried in named recipes. | Train-through-unlabeled-update is distinct. Explicit content/style factorization is not gradient reversal. Generic multi-head training is already tried. |
| Learned session algorithms | Many count/template/gradient stages; one old causal transformer; online grouping and change-point variants. | An exchangeable prefix model or meta-trained low-dimensional session controller is not exhausted. §98's intended small-rotation test is confounded and trained in recording order. |
| Training data | Ten target people; seven external sources; cognitive pooling; shifted task crops; pre-cue rest. | Separate baseline rest is not the same as pre-cue REST4. §277's second SSL form did include five task offsets, but no baseline-rest loader and only 26 channels. More crops are not more independent people. |
| Person/session conditioning | Strong returning-person inference; adverse source selection; confidence selectors tested. | Identity can help routing and hurt new-person classification simultaneously. A style branch need not be thrown away. Two returning people do not describe the hidden returning cohort. |
| Full-system co-design | §281 tested CORAL/B1 with downstream stages on new people; several older variants used posterior fusion only. | New representations must reach the stages that use them, or constitute complete alternative systems. The real v6.6 returning routed tail is still not fully validated by §281. |
| Deployment | Prepared v6.6 cap250 has recorded Mac and A100 checks. | No new literature method has worker-time, memory, reset, package or accuracy validation here. Current cost is a deployment constraint, not a scientific ceiling. |

### Corrections that change search decisions

1. **§98 does not cleanly reject learned session models.** Its QR construction can flip almost every feature sign while passing its absolute-diagonal guard. Training transforms features; evaluation does not. A numerical reproduction confirms this code property. Original traces were not inspected, so the effect on the recorded failure is not established. No fix or rerun occurred in this block.
2. **§290 rejects its hand-built head variants, not every hierarchy or geometry.** It was a core-only, recording-order screen. Learned hyperbolic structure, task/style separation and a new stage representation are different tests.
3. **Rest-based pretraining is still incompletely covered.** §277's expanded task crops are real coverage, not an unrun idea. Its negative paired result constrains that trunk, objective, channels and data. Separate baseline rest and full-montage learning remain distinct.
4. **A dataset name in an old inventory is not evidence it was unused.** CMx7-MM, EEGMAT and Shin were already on Galvani and in earlier teachers. Generic private heads and target/source co-training were also tried. Do not queue them again under a new name.
5. **A large oracle is not an attainable gain.** The +24.4-point Procrustes result fits with true full-session class means and scores the same windows. It is an in-sample diagnostic. The people-scaling slope near +4.7 points per doubling used a surrogate core with fixed stages, five subset draws, and a PC basis fitted on all nine training people even for smaller subsets. It is not a clean measurement of adding wholly unseen people or a scaling law for full v6.6.

## 3. What is current in the literature

| Source and verified date/status | Why it matters | What prevents copying the headline |
|---|---|---|
| [DEFT](https://arxiv.org/html/2610.14497), 10 Oct 2026, on official NeurIPS 2026 list | Explicit task/content and person/style factorization; subject-disjoint MI and arithmetic gains; matched-capacity controls. | Different tasks, folds and head capacity. Release sampler differs from the described main setup. Clarify rights before copying new adapter code; independent paper-based implementation is another route. No measured A100 inference. |
| [SeeEEG](https://arxiv.org/html/2611.17031), 12 Nov 2026 preprint | A frozen vision encoder with learned EEG rendering is a genuinely different representation prior. | Frozen EEG comparisons, many more training people in its PhysioMI comparison, no located implementation, large per-window token count. Small trainable parameter count does not mean cheap inference. |
| [Alignment De-Identifies Without Aligning](https://arxiv.org/html/2611.15204), 11 Nov 2026 preprint | Separates identity removal from alignment of class relationships. | Its theorem concerns congruence maps and a specific affine-invariant distance/statistic. It does not say alignment cannot improve accuracy, or directly cover our log-Euclidean pipeline. |
| [Backbone-Informed Adaptation](https://arxiv.org/html/2611.07192), 4 Nov 2026 preprint | Capacity-matched controls challenge claims that specialized adapters always help. | Its results are recipe/task-specific; they do not prove DEFT will fail or succeed here. |
| [LEAF](https://arxiv.org/html/2509.24302), original Sep 2025, revised 31 Oct 2026, NeurIPS 2026 | Task-text alignment; released tower and instruction weights; four-second interface. | Published held-out-subject normalization sees all trials. A causal replacement is a new experiment. Best supervised results are not zero-shot proof. |
| [BrainPro](https://arxiv.org/abs/2509.22050), original Sep 2025, revised 21 Oct 2026, NeurIPS 2026 | Task-process branches and montage-aware spatial retrieval. | Which branch should represent MI plus cognitive commands is unresolved. Channel retention, rights and runtime need checks. |
| [OSPDIM](https://arxiv.org/html/2608.05315v1), Aug 2026 | An explicitly online, source-free geometric correction for class-mix bias. | Binary MI/artificial imbalance; five updates per window. Unknown three-class transfer and cost; entropy/diversity can introduce their own bias. |
| [SPDAlign](https://arxiv.org/abs/2610.06315), Oct 2026 | Learns feature geometry and distribution-level rotations, beyond old pseudo-class-mean rotation. | Published target inference optimizes over the entire target domain. A prefix-only form is a new algorithm and runtime risk. |
| [HEEGNet](https://openreview.net/pdf?id=CNDNRjpVIL), ICLR 2026; LEAPNet, publisher 9 Oct 2026 | Whole hyperbolic model and per-window SPD anchors broaden the geometry family. | HEEGNet leaves online normalization to future work; LEAPNet was verified only from its abstract. Neither supplies a current deployable win. |
| [TabICLv2](https://proceedings.mlr.press/v306/qu26a.html), ICML 2026, original Feb 2026 | Synthetic tabular pretraining offers a different prior for compact existing features. | EEG subject shift unproven; 44k features unsuitable directly; dependencies, runtime and labeled-context rule need resolution. Distillation is a separate alternative. |

The detailed notes include negative benchmarks and source rights. Dates above distinguish original publication, revision, and conference status. A recent web crawl is not a recent paper. The newest method is not automatically the best use of compute.

## 4. Bottleneck hypotheses, ranked by evidence rather than novelty

**A. Cross-person task geometry and domain mismatch — strongest motivation, cause not isolated.** The surrogate people curve keeps improving, cognitive pooling often harms, and better mean matching partly reduces Shin transfer harm. This supports looking beyond more target windows and interchangeable source labels. It does not identify a single causal bottleneck or prove that more pretrained parameters solve it.

**B. Training/deployment mismatch in adaptation — a specific open gap with close precursors.** The deployed algorithm updates on an unlabeled shuffled prefix. §220 already trains through eight session bias updates, with older full-chain changes −0.10 to +0.16 points. §255/256 train auxiliary objectives before adaptation; §256's C10 full-chain fusion is −0.84 points. Meta-training through an actual label-free encoder/controller gradient update is distinct, but these nearby results lower the prior. Nine training people limit how flexibly we can learn it.

**C. Upstream alignment confounds task mix with nuisance — plausible, not measured here.** Class-dependent covariance can shift a pooled prefix reference. A reference correction before classification is different from correcting posterior counts. Offline label-informed diagnostics can measure whether this mechanism matters before building a complicated adapter.

**D. Information thrown away by representation and normalization — plausible and branch-dependent.** Means, slow potentials, spatial patterns and spectral relationships are not interchangeable. Blind artifact removal is risky because our objective is task prediction on this supplied tensor. Preserve information in parallel views and measure generalization by person/interface rather than declaring one physiological story correct.

**E. Evaluation uncertainty and pipeline coupling obscure small gains — established limitation.** §291 CORAL weight 10 improves its own standalone FB base by +0.559 points; its six-family addition interval crosses zero. §281 instead tests CORAL weight 1, B1 and both through the downstream stages: −0.069, −0.095 and −0.220 points, with intervals crossing zero. These are different forms, not a paired demonstration that the confirmed CORAL10 gain disappears in v6.6. They establish neither a full-system gain nor a direction-level refutation. Repeated selection on ten people and only two returning people prevents confident narrow rankings of small effects.

## 5. Proposed next portfolio — no code or jobs authorized by this report

| Priority | Scientific question and form | Controls and decisive evidence | Readiness |
|---|---|---|---|
| 1 | **Task/style factorization:** a DEFT-inspired CBraMod comparison, using existing weights, assessed as a complete decoder before any integration. | Same folds, normalization, sampler and backbone; existing 98k head and capacity-matched paper-style head; adapter on/off; equal budgets; assert same-person contrastive positives; person-disjoint inner selection. Forward parity is needed to claim the converted checkpoint matches the author's implementation. Preserve separate identity evidence if later routing uses it. | Concrete source and local backbone. Clarify rights before copying adapter code, or independently implement from the paper; published command is not turnkey. Training and inference cost need measurement. |
| 2 | **Learned geometry:** diagnose class-mix contamination, then compare a small learned SPD correction or per-window reference with ordinary prefix centering. | OSPDIM identity-bias control matches its standard centering baseline, not automatically v6.6; source-only reference; labeled balanced-reference diagnostic marked oracle; prefix-only updates; all pinned order schemes. Whole decoder/stages must use the new geometry. | OSPDIM is a concrete online mechanism. The diagnosis comes before a large grid; no gain is predicted from a binary-MI paper. |
| 3 | **Train for causal adaptation:** a small encoder/controller trained through its exact label-free prefix update, or a learned exchangeable prefix model. | Same architecture/compute with no update, ordinary auxiliary training, and train-through-update. Independent support/query episodes, session resets, shuffled training delivery and held-out people. | Largest algorithmic departure, higher engineering and overfit risk. Avoid a huge controller on nine people. §98 must not be reused unchanged. |
| 4 | **Recover unused in-domain training information:** full-montage representation learning with baseline rest and task crops, or the narrowly distinct WG auxiliary task. | Equal compute and window-count controls, full-montage random initialization, task-only pretraining, source-label permutation; exclude all held-out-person recordings. | Existing recordings help setup. Generic SSL and private heads are already negative/mixed; this needs a precise changed mechanism, not a renamed rerun. |
| Parallel review / reserve | **Cross-modal or tabular whole alternatives:** SeeEEG/LEAF/BrainPro or a compact tabular classifier. | Matched features/capacity; current-window or session-prefix preprocessing; no future batch statistics; rights, offline dependency and warmed A100 checks. | LEAF has released code and weights, so a bounded causal-preprocessing/rights review can proceed alongside the main wave. SeeEEG lacks an identified release; LEAPNet is abstract-only. Readiness and expected upside are separate judgments. |

This portfolio deliberately includes whole systems, representation changes and learning-algorithm changes. It is not restricted to preserving any incumbent component. No numeric gain promise or precise completion estimate is justified before the exact designs, existing fit times and Galvani queue are checked. The user asked to review implementation choices first; this report is that research input, not a preregistered run plan.

For any agreed run, register its mechanism, numeric prediction and refutation rule **before** scoring. Use discovery diagnostics to size the mechanism, then evaluate components in the current complete system or evaluate a complete alternative whole. Report person-nested uncertainty, order sensitivity, returning-person results separately, and real runtime. Use zero-change or equivalent baseline controls where meaningful; a new architecture need not reproduce v6.6. The v3 replay guard is necessary but does not validate a new shuffled scorer; preserve causal-prefix, reset and pinned-order checks. Do not rebuild evaluation during collection or interpret a screen as a package result.

## 6. Data search conclusion

No newly verified, public, licensed EEG corpus with all three matching tasks was found. Scherer is the known closest public match and is already tested. Banville's matching study is request-only in the inspected source, with availability and rights unresolved. Inner-speech and N400 corpora are different auxiliary tasks, not replacement WORD labels. MEG can match task semantics while differing in measurement physics. A longer recording or more crops adds observations, not new independent target-like people.

Public baseline rest is a real coverage gap, but it is training data. The challenge does not give target rest. Methods that require true labels after each target trial, a full held-out batch, or separate target recordings do not supply directly usable gains here. Dataset, checkpoint and code licenses are separate checks.

## Review record

- Source revision at start: `33af6f22a9de3d4537ef759f9f104d74a4949a46`; official submodule unchanged at `a66bc23d1a0e686b895c093a642b45c74970758b`.
- Three research readers independently checked current methods, bounds/pipeline, and data/transfer; the main reader checked cross-field transfers and synthesized the map.
- Final source/protocol review completed at **17:24:48 CET** (2026-11-16 16:24:48 UTC); elapsed **30 min 6 s**. Adversarial corrections distinguish CORAL10 from CORAL1, fixed priors from labeled-source adaptation, invertible alignment from information loss, and component tests from whole-system tests. Local note links and Git whitespace checks passed. Only Markdown research notes and the STATE link changed.
