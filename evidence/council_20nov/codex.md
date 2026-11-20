# Council review · Codex (OpenAI) · 20 November 2026

**Upload choice.** Use the existing `final_stack_hardcb.zip` if paired timing with active fine-tuning qualifies it by Saturday 10:00 CET. Otherwise choose the existing F-C25p hard ZIP. My
central sealed estimate for hardcb is **+0.1 to +0.2 pp** over 82.9873%, with a planning range of **−0.4 to +0.6 pp**. The gap to third is +1.71 pp and to first about +2.91 pp. I found no
tested or literature-backed change with a credible central estimate of that size by Saturday. [State](../../docs/STATE.md) · [Final stack](../wave_20nov/final_stack.md) ·
[Forecast record](../review_19nov/forecast_calibration.md)

## What the evidence changes

Track 2 predicts motor imagery (MI), mental subtraction or word generation from each 41-channel, four-second EEG window. The official metric averages balanced accuracy over
participant/session/interface cells. Reviewed upstream `main` at `8447e181`. Primary literature and web research ran from about 09:25 to 09:41 CET, alongside the comparison with local
evidence. Locally, the six recording-order checks printed `OK`, the hardcb archive passed CRC, and I read the ZIP configurations and hashes. The proposed comparisons remain unrun.
[Official contract](../../OFFICIAL_CONTRACT.md) · [Replay check](../../replay/check.py)

- **Coverage and complementarity are different.** B5 improved one raw family +4.60 pp standalone and moved the cap250 swap −0.39 pp, comparing five with nine training people. The experiment
  held every other family fixed. Virtual subjects added little or hurt, corrected Shin cost −0.19 pp, and B6's 81 additional MI participants gained +0.228 pp on one order at 5/6 of the raw
  slot and −0.077 at 1/6. This supports looking for independent errors at the actual consumer, with joint testing when two candidates use the same slot. It leaves a whole-ensemble same-task
  learning curve open. [B5](../wave_20nov/laneB5_curve.md) · [Virtual subjects](../wave_20nov/laneV_virtual.md) · [Shin](../wave_20nov/laneS_shin.md) · [B6](../wave_20nov/laneB6_mi84.md)
- **Adaptation gain depends on the starting system.** The clean reference is 44–46%, cap250 full0 is 73.74%, and Teams A/B/C reached about 58.8/65.3/69.7%. Team C's 62-recipe sweep moved
  66.8→69.7%. M1 gained +2.34 pp over blank history on a 59.44% base and scored 61.77%, below prefix BN's 62.93%. Inside hardcb, light/heavy TTT gained +0.079/+0.343 pp across two delivery
  types, both intervals spanning zero. Heavy TTT lost −3.03 pp externally on Team C. Default TTT gained +4.49 pp on two returning people on that weaker base. Neither result predicts the
  gain inside returning cap250. [Clean reference](../wave_20nov/CLEANSLATE.md) · [A](../wave_20nov/cleanslate_teamA.md) · [B](../wave_20nov/cleanslate_teamB.md) ·
  [T2](../wave_20nov/laneT2_sweep.md) · [M1](../wave_20nov/moonM1_incontext.md) · [Lane T](../wave_20nov/laneT_ttt_cap250.md) · [T4](../wave_20nov/laneT4_valid.md)
- **A correction to the Opus feedback claim.** I read the real stock, hardcb, hardcb-B and hardcb-light ZIPs. All set `v6_ft_labels=0`, `v6_ft_relabel=1` and `v6_routed=1`. `step62`
  computes the original v5.1b shadow, calls `ft.step` with that label and `route.tail.reapply`, then calls the altered `mix6` and RouteB. The changed warm predictions and slot members
  cannot feed back into the FT teacher in these ZIPs. The bank still approximates the full live downstream state and device arithmetic. The shadow is the right zero-change anchor. Exact
  replay can detect student drift under a fixed teacher. A feedback loop would be a separate algorithm. [Solver](../../submissions/lean_v66/submission.py#L1513) ·
  [Hardcb mixer](../../submissions/final_stack/stack_model.py.inc) · [Opus](opus.md)
- **The ceilings cover particular representations.** Whole-session Sinkhorn on stock scores was +0.59 pp full0 and −0.63 within0. Simple tangent-space plus Euclidean alignment reached
  49.71% causally. The 89.8% shared-error figure covers sibling arms. Team C independently got about a third of stock errors right in Fable's one-order exploratory comparison, yet its
  nested fusions ranged from −0.16 to +0.26 pp. A selector needs information about *which* error it can fix. Agreement, confidence and high standalone accuracy have each been weak proxies.
  [First-principles audit](../audit_20nov/first_principles_fable.md) · [Independent fusion diagnostic](fable.md)
- **Validation can move the estimate by more than the candidate.** Public held-new is ten reused people. Corrected replay has one full-session and one within-run shuffle. The official cell
  mean gives P04 and P10 three times the weight of a one-visit person in that fold. Scherer has 22 of 41 channels reconstructed and its enrolled package identifies 144/144 other-person
  streams as returning. Full and within-order reversal, reconstruction sensitivity and repeated selection all matter. The 57% new-cell share is unverified. Under the withdrawn design, about
  38% of all cells belong to eight returning people without later labelled public visits. Their ±1 pp moves the mixture ±0.38 pp before transfer assumptions.
  [Data-path audit](../audit_20nov/bugs_datapath_codex.md) · [Model audit](../audit_20nov/assumptions_model_codex.md) · [Composition](../audit_20nov/sealed_composition.md) ·
  [NI-V](../wave_19nov/NIV.md) · [One-session returning](../wave_20nov/laneR_onesession.md)

## Ranked work for Saturday 10:00 CET

The estimates below are incremental full-system development balanced-accuracy points unless marked sealed. Intervals for unrun arms are planning priors. Cluster times assume admission and
the pinned assets. Pre-register before scoring, smoke first, seal predictions before gold, and score every component through the consuming chain. Keep the official cell mean, paired
people/order intervals, and separate returning views. Prove stock zero, pinned order, activation, prefix equivalence and interleaved reset. [Ledger](../../docs/LEDGER.md) ·
[Rules audit](../audit_20nov/assumptions_rules_opus.md)

### 1. Resolve hardcb versus hard with the costly path active

- Hardcb is +0.514 full0 and +0.270 within0 versus stock, or +0.392 pp pooled [80% +0.280,+0.519]. Its paired point advantage over hard is **+0.183 / −0.035 pp**, or **+0.074 pp** pooled.
  The Scherer-free B variant adds +0.007 pp over hard on NI-V B-new, from two negative deltas within noise. The +0.23 pp role-weighted DEV figure is sensitive to unmeasured returning
  people. Historical sub-1 pp transfers also vary. Either method puts the marginal gain of CBraMod over hard at only a few hundredths of a sealed point centrally.
  [Final stack](../wave_20nov/final_stack.md) · [Calibration](../review_19nov/forecast_calibration.md)
- **Decisive test, ≤1 h on Galvani after allocation.** Stock, hard, hardcb in warm paired A/B/B/A on the same GPU/worker and identical inputs. Run recognised public, real naturally
  unrecognised Scherer and a pinned ~190-window real public fixture with recognition suppressed for timing. Use longer stress through the 250 cap. `laneT/time_ft.py` is a quick active-path
  smoke. Its synthetic stream measures cost and the script starts its timer *after* `fresh_model`. Add restore, session-copy/start, peak GPU memory and refit/step/CSP counts from the real
  workers. Project from the paired total ratio at 40/57/75% unrecognised windows, plus cold overhead sensitivity. At 2,554.9 s, a ratio of 1.409 already consumes the entire budget. My
  practical cutoff for hardcb is **≤3,350 s** on the upper plausible workload, with no package drift beyond stock A/A and no latched error path. Hard alone measured 1.06× unrecognised and
  1.09× recognised on an RTX 2080 Ti. Neither card is a PRO 4500 receipt. [Timing script](../../laneT/time_ft.py) · [Hard timing](../../friday/cap250_hard_package/README.md) ·
  [Worker qualification](../../friday/qualify_package.py)
- Use B1's registered seed, TTT and B6 slot arms in parallel. The current B1 card supersedes the earlier plan for two more corrected held-new full-shuffle orders. Those banks do not exist
  yet. NI-V can supply eight exact external full shuffles, with its montage limits. A one-seed CBraMod arm is a possible cheaper candidate only after it has a signed and timed ZIP. Its seed
  must be chosen by a preset rule, without picking the best of three on the same ten people. [B1](../wave_20nov/laneB1_final_choice.md) · [NI-V](../wave_19nov/NIV.md)
- **Failure path to observe.** `HardWaveEnsemble.step` requires `mix6` to consume `hard_window`. A v6 error before `mix6` leaves the frame unread and raises at that call, despite the
  parent's fallback. An error after consumption can produce the same failure on the next call after `dead` latches. This is a code-path finding. The qualification replay saw zero v6 errors.
  Check the counts on the active path, where real refits run. The enrolled public score alone scarcely exercises this path.
  [Hard wrapper](../../friday/cap250_hard_package/hard_model.py.inc#L48) · [Parent error handling](../../submissions/lean_v66/submission.py#L1556) ·
  [Qualification evidence](../wave_20nov/final_stack.md)

### 2. Test whether predictions within one window identify errors the whole-window posterior misses

- **New candidate inspired by BTTA-DG.** The ICLR paper reports 76.49% vs 74.52% on one-second, batch-1 MI windows. It summarizes the distribution of predictions *within* the delivered
  window as a Dirichlet vector and learns class densities from earlier confident trials. The released script at `5932d026` computes `EA(test_x)` across the full target set before iteration.
  It also defines `add_sample` but never calls it. Its live loop calls `update` and `predict_class` with empty buffers, so the Gaussian-mixture correction is inactive. The paper's execution
  path and gain remain unverified. The information in within-window dispersion is a testable idea independent of that result.
  [Paper §§2–3 and App. I](https://openreview.net/pdf?id=VDg6Pv4S3v) ·
  [Released loop](https://github.com/luo-huan-123/BTTA-DG/blob/5932d026bbd8a7de106d31a6d264f4f4924537e4/BTTA_DG.py#L258-L288) ·
  [Full-target EA call](https://github.com/luo-huan-123/BTTA-DG/blob/5932d026bbd8a7de106d31a6d264f4f4924537e4/BTTA_DG.py#L348-L352)
- **Mechanism.** Capture four existing segment logit contributions from one native FBC member's linear head. Summing the dot products and bias must recover its logits. Use each class's
  dispersion across the segments as a feature, conditioning on the same whole-window logits in both arms. A brief burst and sustained task evidence can produce the same whole-window margin.
  Use past, frozen shadow pseudo-labels to fit shrinkage diagonal densities with shared variance and 20 prior windows per class. Apply one fixed 0.10 bounded log-likelihood correction after
  RouteB and before the live count/CSP. Begin when every class has eight effective observations. This keeps the teacher fixed and costs no new encoder forward. Four segments favor
  regularized moments over estimating an eight-component mixture. [Native segment calculation](../../submissions/lean_v66/submission.py#L297) ·
  [Existing posterior and PC transduction](../audit_20nov/first_principles_fable.md)
- **Decisive test, 2–4 h on Galvani.** Compare the same hardcb trajectory with identity, whole-window features only and segment dispersion added. Replay genuine versus age-matched donor
  history with identical mass and current features. Capture real live models on Scherer and use full0/within0 fitted-member streams when available. Check that each class buffer fills and
  the correction changes eligible labels before gold. Forecast **+0.12 [−0.35,+0.60] pp** over hardcb, <2 ms/window engineering target. If the paired upper bound is ≤0 against either
  matched control, refute this form. If the signal exceeds +0.3 pp against both and holds direction on the external live path, qualify the exact ZIP. Otherwise leave it undecided. A useful
  result would show why the whole-posterior Gaussian and TTT missed those errors.
  [F-C19 disagreement ceiling](../../docs/LEDGER.md#f-c19--cross-trajectory-csp-evidence-at-complete-system-disagreements) · [Lane T](../wave_20nov/laneT_ttt_cap250.md)

### 3. Build diversity with different source people at the same inference cost

- B5's N5 mean beats its N9 swap by +0.39 pp, despite worse standalone. N5 ran the same five-person subset across the three seeds within each arm. Test three different subsets across seeds,
  averaging their probabilities **before** the nonlinear stages. This could improve coverage of the incumbent's errors without adding corpora, synthetic people or extra seeds. The leading
  competing explanation is that N5 merely supplies softer probabilities. Measure disagreement restricted to incumbent misses and match aggregate entropy to the N9 control in a diagnostic
  arm. [B5 design](../wave_20nov/laneB5_curve.md) · [Subset construction](../../laneB5/curve.py) · [EEG ensemble shift evidence, clinical task](https://pubmed.ncbi.nlm.nih.gov/41650580/) ·
  [Deep-ensemble shift caveat](https://papers.neurips.cc/paper_files/paper/2022/file/da18c47118a2d09926346f33bebde9f4-Paper-Conference.pdf)
- **Decisive test, 30–60 min on Galvani from existing B5 banks.** Pre-fix `(seed,subset)=(0,0),(1,1),(2,2)` for N5. Compare N5 shared subset 0, N9 three seeds, hard and hardcb. Replace the
  four hard shares in hardcb with the candidate mean, preserving its CBraMod share, shadow and warm delta, then rerun all consuming stages. Forecast **+0.05 [−0.45,+0.55] pp** versus
  hardcb. A gain that vanishes after entropy matching supports calibration and weakens the diversity claim. If there is a useful gain against hardcb and the matched shared-subset control,
  fit the same predetermined roster on full-source and fold-specific data on Leonardo and repeat within0 and external exact-package. Allow 2–4 h after the screen. B5 saved posteriors rather
  than streamable checkpoints, so additional orders need fits. Keep the same count of members and measure paired time. [Consumer](../../submissions/final_stack/chain/final22.py) ·
  [B4 seed count experiment](../wave_20nov/laneB4_seeds.md)

### 4. Align a measured spectral landmark on the delivered prefix

- Test an individual posterior alpha peak as a stable frequency reference for a parallel band-power head. Remove the aperiodic spectral slope, average class-blind spectra of earlier windows
  over posterior sensors and shrink a reliable 7–13 Hz peak toward the fold-source reference. Move only the peak-relative bands by a bounded ±1 Hz, retaining the fixed bands as a control.
  This uses a physical cue independent of pseudo-labels. F37 moved source spectral amplitudes by a fixed dose, and Lane V simulated heterogeneous subjects. Neither adapted to a repeatable
  peak measured on the current stream. The relevant MI frequency-aware study uses subject-specific decoding. The same-task MEG study found that cortical beta contrasts differ by task and
  person, which motivates a check and limits any universal frequency-shift claim. [Spectral parameterization](https://doi.org/10.1038/s41593-020-00744-x) ·
  [AFFormer](https://www.sciencedirect.com/science/article/pii/S1746809426016988) · [Same-task MEG](https://pmc.ncbi.nlm.nih.gov/articles/PMC10171552/) ·
  [F37](../../friday/fbc323_spectral_shift/prereg-19nov.md)
- **Decisive test, 1–3 h Galvani CPU plus replay.** First estimate peak agreement on disjoint earlier-window halves at ages 40 and 80, without labels. If real history cannot beat a
  matched-age donor on peak replication, there is little reason to fit the adaptation. Otherwise fit source-only compact heads at fixed and peak-relative bands, and evaluate fixed, genuine
  and donor at one fixed 1/6 raw-family share in the complete consumer. Report word/subtraction as well as MI recall, first-40 versus later change and the full metric. Forecast **+0.08
  [−0.40,+0.55] pp** versus the fixed-band insertion, <2 ms/window target. A nonpositive paired upper bound against fixed or donor refutes this form. Inspect existing F37c full-system and
  all-source receipts before spending effort on a second spectral package. F37 beta's +0.554/+1.086 pp are standalone and were selected after seeing them.
  [F37c protocol](../../friday/cap250_f37/prereg-19nov.md) · [F37 results](../../friday/fbc323_spectral_shift/result-19nov.md)

## Other literature and competition evidence

| Lead | Transfer judgment for this stream |
|---|---|
| **Wimpff et al. and MOPC** | Wimpff uses a 32-trial buffer, source shrinkage and entropy updates on two MI datasets. Its 2a 67.31% from 57.38% source includes alignment and BN. MOPC uses five models, 24 prior trials and BN-affine optimizer steps for one model per new trial. Its four-class run attributes +12.59 of +15.08 pp to dynamic alignment. The remaining +2.49 was against its own alignment-only arm with one seed. The released four-class driver keeps one model and buffer across sessions T and E, which Track 2 forbids. Cap250 already runs alignment/normalization and confidence/consistency refits. Carry over only cheap reliability evidence within the existing refit dose. MOPC's 40–60 ms per trial would be 800–1,200 s across ~20k trials if added whole. [Wimpff §§II–III](https://arxiv.org/html/2311.18520) · [MOPC Tables VIII–IX](https://arxiv.org/html/2610.30422v1) · [Four-class driver](https://github.com/KunKun-Zhang/MOPC/blob/main/bci2a/run_2a_4class_mopc_final.py#L281-L287) |
| **Riemannian alignment and OSPDIM** | OSPDIM adjusts the tangent reference with an information-maximization loss, five updates per call on a 32-sample buffer. The foreground benchmark induces severe 0.1–0.4 minority/majority ratios in binary MI. Public Track 2 class counts are close to balanced, and marginal whitening can leave class geometry unchanged. F-C29 recolored source-class covariances at actual native CSP refits and gave +0.330/−0.187 pp full/within, −0.093 exactly on Scherer. First test geometric distortion at early shuffled ages under genuine prefix versus a fixed reference. Full manifold optimization has low priority for this deadline. [OSPDIM protocol](https://arxiv.org/html/2608.05315) · [Geometry](https://arxiv.org/html/2611.15204v1) · [F-C29](../../friday/cap250c29/result-19nov.md) |
| **Prefix transduction and class priors** | SCALAR stabilizes a logit queue with Sinkhorn on image corruption. Its small-batch appendix reaches batch 2 with a 1,024-item window. LAME's per-batch affinity needs other examples. Local posterior balancing reversed across orders, and Team C's C5 fit 2,583 tangent features to ≤190 pseudo-labelled trials and lost 0.33 to 3.67 pp on the two primary TTT fusion forms. The conditional-dispersion test uses three statistics from inside the current tensor, keeps a strict prefix and emits once. [SCALAR App. E.2](https://arxiv.org/html/2611.12242) · [LAME](https://openaccess.thecvf.com/content/CVPR2022/papers/Boudiaf_Parameter-Free_Online_Test-Time_Adaptation_CVPR_2022_paper.pdf) · [Local prefix lane](../wave_19nov/laneD_prefix.md) · [Team C](../wave_20nov/cleanslate_teamC.md#c5-result-leonardo-64922136-held-new-64922150-returning-orders-01) |
| **NeurIPS 2021 BEETL and BCI Competition IV** | BEETL winner Cogitat combined latent statistical alignment and Deep Sets. Its motor-imagery result used labelled target calibration with PhysioNet source data and 30 common channels. The second-place approach used class alignment with per-target calibration, and the third used SPDNet. BCI IV 2a winner used subject-specific labelled FBCSP, one-vs-rest and Parzen classification. Borrow matching montage and class contrasts in source training. The official sealed stream has no target calibration. M1 and B6 already probe important legal subsets. [BEETL §§4.1–4.3](https://proceedings.mlr.press/v176/wei22a/wei22a.pdf) · [BCI IV standings](https://www.bbci.de/competition/iv/results/) · [Winner entry](https://www.bbci.de/competition/iv/results/ds2a/KaiKengAng_desc.pdf) |
| **NeurIPS 2025 and foundation models** | KUL_EEG's official 0.88668 win was response-time prediction with pretraining on over eight million segments. MBZUAI's public Neurosned code describes the original combined metric. Official Challenge 2 ranks MBZUAI second and flags contiguous-subject information in that evaluation. Within-trial temporal features transfer as a hypothesis. Between-trial chronology, subject grouping and regression leaderboard scores provide weak evidence here. A real-world EEG FM benchmark also finds variable gains and gradient drift. Local clean-slate FM and in-mixer evidence already justify cheap complementarity checks ahead of a fresh FM training lane. [Official board](https://eeg2025.github.io/leaderboard/) · [KUL author report](https://www.vscentrum.be/post/scaling-eeg-foundation-models-on-vsc-an-iclr-2026-benchmark-and-a-neurips-2025-eeg-challenge-win) · [Neurosned](https://github.com/sneddy/neurosned) · [NeuroAdapt-Bench](https://proceedings.mlr.press/v340/lee26a.html) |
| **Kaggle Grasp-and-Lift and HMS** | Grasp-and-Lift winner used subject-trained covariance/other views, a global calibration layer, and an ensemble chosen for AUC. Their inputs were chronological executed movements and their rules allowed earlier series into later series. HMS winner Sony's raw EEG and spectral CNN processing addresses a clinical soft-label task. The actionable overlap is source-held calibration of genuinely different views. Equal per-cell recall and cross-person error complementarity are the comparison here. [Grasp-and-Lift winner](https://github.com/alexandrebarachant/Grasp-and-lift-EEG-challenge) · [Sony write-up](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/492560) |

## Release call

1. Freeze the exact, unmodified hardcb ZIP by Saturday 10:00 if active-path timing clears and further exact-package evidence shows no recurrent failure. Hardcb SHA-256
   `958fd3f3662102f7935da0ab56c3c9e62304d151f8e68f9dbf797a2025782705`. The exact external hardcb-B ZIP used different CBraMod weights to exclude Scherer. A file-level CRC and the existing
   replay receipts support different checks. [Final stack](../wave_20nov/final_stack.md)
2. Use F-C25p hard if hardcb timing is unresolved or exceeds the practical cutoff. Hard SHA-256 `6a569180d343c038558fe7eccb7903a36e2bdf694af0df3687bff3b17049fbcc`. Preserve known stock A/A
   variation context. I would leave the time guard at the qualified setting. An elapsed-time trigger changes predictions with host speed and does not cap the entire evaluation. Consider a
   new arm only when it beats the exact chosen parent with matched consumption, per-order evidence and worker qualification by the cutoff.
   [Hard receipts](../../friday/cap250_hard_package/README.md) · [Model audit](../audit_20nov/assumptions_model_codex.md)
3. Prefer the early end of the brief's **Saturday 14:00–16:00 CET** decision window. Only the best eligible sealed score counts. Treat a repeat attempt as available only after Codabench
   reports `Failed` and the live slot reopens. A queued or running attempt reserves the slot. Qualify and seek explicit user authorization for the exact SHA before any upload. The budget is
   3,600 s on one RTX PRO 4500 and the deadline is 23:59 CET. Each prediction uses the current tensor and delivered session prefix only. Reset all mutable state between sessions and declare
   external data and compute. [Live upstream terms](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/terms.md) ·
   [Track FAQ](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/scientific_faq_bci.md) · [Contract](../../OFFICIAL_CONTRACT.md)
