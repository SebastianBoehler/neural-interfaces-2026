# Cross-field mechanisms: research, not experiment results

16 November 2026. No installations, model downloads, training, scoring, or job submissions were made for this note. The proposed transfers below have not been measured on PROTEUS. The owning evidence and negative controls are as important as the paper names.

## 1. Train the model for the update it will perform

Our deployment problem is classification after a label-free, session-local update. Ordinary supervised training optimizes the classifier before that update. Joint reconstruction or contrastive training does not automatically optimize classification after the update either.

[MT3, AISTATS 2022](https://proceedings.mlr.press/v151/bartler22a/bartler22a.pdf) supplies an older vision precedent: an outer supervised objective trains an initialization through a self-supervised inner update. [End-to-End Test-Time Training for Long Context](https://arxiv.org/html/2512.23675v2), first submitted December 2025, applies the broader train-for-adaptation principle to language models. Its language-model speed and accuracy claims do not transfer to EEG. [Meta-GC-TTT](https://openreview.net/pdf?id=LlDHSp7wPH) offers a 2026 reinforcement-learning example; the accessible paper says submitted to NeurIPS 2026, so acceptance is not established here.

**Transfer hypothesis:** meta-train a small encoder or geometric adapter on public training people. Each episode contains an unlabeled delivered prefix and a distinct labeled query. Differentiate through the exact prefix update and supervise the query only during training. At evaluation, adaptation never fits labeled public examples, receives target labels, reads future windows, or crosses a session. Fixed offline-trained parameters and priors remain allowed. Train with shuffled delivery; acquisition-order metadata is unavailable, and delivery neighbors must not be assumed to be recording neighbors. Within-window temporal dynamics remain available. Inner/outer separation and held-out people must be real, not just different correlated crops.

**Coverage:** §220 unrolls eight session bias updates, rather than an encoder gradient adaptation; its older full-chain changes were −0.10 to +0.16 points across bases, with returning-member losses of 3.3–4.1. §255 trains auxiliary objectives before later adaptation; adaptation improved its already harmful auxiliary fusions, but did not make them beat the chain. §256's C10 full-chain fusion was −0.84 [−1.41, −0.27]. §98 directly trains a causal transformer. The proposed gradient-based inner update has not been identified in these runs, but train-through-update in the broader sense is not wholly new here. These are close controls and negative precursors. Nine fit people make flexible meta-training vulnerable to overfitting; extra seeds do not add new people.

**Decision:** worth a separate small whole-system arm, not a claim that another entropy step will help the incumbent. Specify the inner loss, update variables, reset, episode distribution and cost before implementation. Compare ordinary supervised training, joint auxiliary training, and train-through-update with matched models and compute. A new model must have its own coherent session features rather than being rejected solely through a fixed posterior average.

## 2. Distinguish nuisance alignment from class proportions

[OSPDIM, 5 August 2026](https://arxiv.org/html/2608.05315v1), extends offline SPDIM with an online learned positive-definite bias on covariance features. It corrects the reference used for whitening, with a FIFO feature buffer of 32 and five update steps per sample. Its information-maximization objective includes conditional entropy and marginal diversity. The reported tests use binary motor imagery and controlled class imbalance. They do not establish three-class cognitive-task transfer or our runtime.

**Transfer hypothesis:** our running covariance reference mixes nuisance differences with whichever classes happened to arrive. Posterior class-count correction cannot generally undo the resulting change to a geometric decision boundary. Invertible whitening does not itself destroy information. OSPDIM applies a learned SPD bias after ordinary whitening; other learned-reference or class-conditional latent models are different designs. They act at a different point from the shipped count filter. Diversity losses can themselves bias predictions toward uniform counts; balanced-accuracy scoring does not establish that delivered class counts are uniform.

**Coverage:** §288's simplified TSMNet uses detached log-Euclidean prefix statistics, with no learned SPD bias or dispersion. SessionEM/TSEM estimate class templates downstream; §103/104 rotate estimated class means in a different feature representation. These constrain particular forms, not every upstream geometric model. The exact distinction and the noncommuting-matrix caveat are in [space_and_pipeline.md](space_and_pipeline.md).

**Decision:** first measure whether class-mix contamination matters using labeled offline diagnostics, clearly marked as oracles. Then test an actual label-free mechanism, keeping the source-only and unchanged-prefix controls. An oracle sizes its defined diagnostic; it does not predict attainable accuracy or a global ceiling. A per-window learned-anchor alternative is attractive because it need not aggregate future windows, but LEAPNet's accessible abstract alone is not enough to reproduce it.

## 3. A tabular foundation model as a different classifier

[TabICLv2, ICML 2026](https://proceedings.mlr.press/v306/qu26a.html), was first submitted 11 February 2026. Its prior comes from synthetic tabular datasets. It can make predictions using labeled training context without fitting a conventional head. The paper leaves out-of-distribution generalization as future work; random tabular benchmark splits are not leave-one-person-out EEG evidence. This is a cross-field classifier hypothesis, not evidence of an EEG improvement.

The [official documentation](https://tabicl.readthedocs.io/en/latest/) reports pretraining with 2–100 columns and 300–60,000 rows. Broader feature counts are extrapolation. Our roughly 44,000 core features are unsuitable as a direct input. Compressing them to 512 PCs previously lost about four points (§49), so cheap compression cannot be assumed harmless. A bounded selection of band powers, covariance coordinates, means, slow features or existing encoder features is a distinct representation choice, selected on training people only.

The [official repository](https://github.com/soda-inria/tabicl) supports a local checkpoint, disabling automatic downloads, and an immutable training-context cache for repeated inference. Default ensemble count is eight; auto offloading can write to disk. Worker-compatible packaging, cache memory, one-query-at-a-time behavior and A100 latency remain unmeasured. Its code is BSD-3-Clause; checkpoint provenance and terms still require a separate check.

**Contract caveat:** the organizer excludes fitting against labeled public anchors during evaluation adaptation. A fixed classifier with precomputed training-only context is conceptually different, but the clarification does not explicitly adjudicate in-context prediction from labeled source rows. Do not assume it is cleared. An offline distilled classifier is another design and would need its own test. In every case, target-session information must reset, and normalization must not inspect later queries.

**Decision:** potentially inexpensive representation/classifier diagnostic after rights and protocol checks. Compare the same compact features with logistic regression and a strong nonlinear head. A loss from an arbitrary PCA projection would not close the family. A gain on a standalone diagnostic is not a complete-system gain.

## 4. Current tabular releases are not all deployable

The official [Prior Labs technical-report page](https://priorlabs.ai/technical-reports) lists TabPFN-3.5 in September 2026. Its [28 October Thinking announcement](https://priorlabs.ai/blog-posts/tabpfn-3-5-thinking-upgrade?waitlist=open) and [4 November changelog](https://priorlabs.ai/blog-posts/the-prior-changelog-thinking-edition) describe a hosted upgrade. A hosted API does not provide an offline competition artifact; no target data was sent to any API. The TabPFN-2.5 report is dated November 2025, so a recent crawl is not a recent method release. TabICLv2 is a more concrete offline research lead than an API-only headline, subject to the checks above.

## 5. Detect harmful adaptation without inventing a guarantee

[Monitoring Risks in Test-Time Adaptation, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/746960ad49ddb47248970a0e1404230c-Abstract-Conference.html) studies monitoring with unlabeled loss proxies calibrated using source data. Its guarantees depend on an informative proxy assumption; confidence is not automatically a valid error estimate under arbitrary subject shift. Small session prefixes also make finite-sample bounds weak. This is a possible protection for a future adapter, not evidence that we can detect every bad session or certify a hidden score.

Our §236 confidence-based selector was negative on its settings and descriptors. Reusing confidence to claim a new safety guarantee would be unjustified. Any monitor needs a held-out-person test of proxy reliability and a useful decision threshold, with the unchanged model as a control. This is secondary to learning a genuinely better decoder.

## 6. General design consequences

- Prioritize different mechanisms, not many nearby configurations: representation factorization, learned geometry, train-for-update, and a different classifier prior.
- Architecture choice depends on the available signal. External text can encode training semantics; it is not an extra runtime sensor. Resampling 120 Hz to 200 Hz cannot restore frequencies above 60 Hz.
- A transformer within a four-second window is different from a transformer over delivered windows. The latter must not assume delivery chronology. A permutation-invariant summary of earlier windows is another allowed family.
- Shared/session-specific factorization is plausible, but eliminating identity is not the objective. Identity can support returning-person routing, and label-relevant geometry can survive identity removal without becoming aligned.
- Keep discovery screens and direction-level conclusions distinct. Whole alternatives need whole-system evaluation. Component proposals need the current full system, shuffled delivery and independent zero-change/order guards.
