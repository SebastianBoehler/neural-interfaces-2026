# Strategy review: path to the top three (20 October 2026)

Sources: git history `3de267e`…`7234467`, `OPTIMIZATION*.md`, error analysis of
the saved r6 probabilities, a code review of everything since `6807043`, and a
literature survey (links below). Scores are cell-mean balanced accuracy (%);
dl = `dev_loso`, ds = `dev_session`, al = `all_loso`, ss = `seen_session`.

## 1. What produced the breakthroughs

| Lever (round) | Kind | Δ dl | Note |
|---|---|---:|---|
| Causal session recentering (R1) | domain alignment | ≈ +7 | +14.6 ds, +9.6 official |
| 1–4 Hz band + centred < 1 Hz waveform (R1, v2) | signal discovery | +6.8 | slow signal is spread over the scalp |
| Probability ensemble v2 + aligned/raw ShallowConvNets (R2) | ensembling | +6.8 | |
| Unaligned FBCNet member (R6) | input-stream diversity | +1.5 | +1.9 ds |
| Causal class rebalancing (R3) | exploits balanced runs | +1.3 | +6.3 ds, +5.8 official |
| Windowed rebalancing W 30, α 1 (R4) | exploits run length | +1.0 | +3.0 ss |
| FBCNet member (R3) | architecture | +0.8 | |
| Scherer 2015 members (R5) | external data | +0.5 | within noise, licence question |

Only structural changes (alignment, a new signal, ensembling) ever gave more
than 2 points; everything since round 3 is ≤ 1.5. The next step change has to
come from a **new source of information**, not from tuning what exists.

Fresh signal (20 Oct): one seed of a **fine-tuned CBraMod** (pretrained EEG
foundation model, aligned input) scores dl 60.0 alone (≈ best single net) and
adds **+1.5 dl / +2.2 ds** to r6, where an extra same-size FBCNet adds
+0.1 / +0.9. First new-family gain since round 6; unconfirmed (1 seed).

## 2. Challenge bounds and the solution space

Hard bounds: input `(1, 41, 480)` µV at 120 Hz, 4 s from cue + 1 s; no
labels or metadata at test; one window at a time, in order, state reset per
session; ≤ 60 min on one A100 (GPU available, we use CPU); only
`submission.py` + data files; public redistributable pretraining data and
models allowed if declared; sealed = later sessions (up to "session 6") of
the 10 public participants + 10 unseen participants; metric = mean of
participant × session × interface cells; top three audited by re-training.

| Region of the solution space | Status |
|---|---|
| Input alignment (EA, Riemannian recentering, priors) | **exhausted** (k, stretch, train-causal) |
| Hand-crafted features (filter-bank tangent, slow waveform, ERP covariances, REVE frozen) | mostly exhausted |
| Supervised convnets from scratch (8 architectures, aligned/unaligned, augmentation) | **exhausted** |
| Ensemble combination (weights, stacking, geometric mean, seeds) | **exhausted** |
| Output-level adaptation (rebalancing, windowing, Sinkhorn) | **exhausted** |
| Feature-level TTA (AdaBN) | redundant with rebalancing |
| Self-training with all pseudo-labels | rejected (50–75% correct labels) |
| **Fine-tuned foundation models** (CBraMod, REVE, LaBraM, EEGPT, ST-EEGFormer) | CBraMod: +2 dl / +1.5 ds but all_loso +0.1, seen_session −2.2 (R8, not adopted); LaBraM base recipe no gain; REVE/EEGPT open |
| Label-shift-aware alignment (class-balanced running reference) | closed: oracle with true labels +0.05 dl (R8) |
| Selective TTA (T3A prototypes, confidence ≥ 0.6/0.7) | rejected: adaptive below frozen control (R8) |
| Calibrated fusion (per-member temperature) | untested; matters once FM members join |
| Multi-dataset pretraining with ≥ 2 classes per session (Shin 2017, BCI III-V, BNCI 004) | untested |
| GPU at inference (larger models, test-time ensembles) | unused |
| Task-period crops (cue + 2…4 s, same pipeline) | rejected: −2.8 dl / −6.2 ds for FBCNet (R8) |
| Per-session member weighting | rejected: oracle selection below fixed mix (R8) |
| Resting-state alignment priors | proposed R1, never tested |

## 3. Where the errors are (r6, saved probabilities)

- Classes are confused evenly (recall MI 65 / SUB 67 / WORD 65 in al).
- Participants dominate: al from P04 49 and P10 61 to P07 83. P10 recalls
  MI 26% when unseen but scores 84% on later sessions when session 1 is in
  training; P04 is weak everywhere (49 al, 58 ss).
- **Early windows**: positions 0–30 of a session 63.8% vs 30–60 69.8% (al):
  the adaptation warm-up costs ≈ 6 points on 16% of each session.
- **Confidence is calibrated**: max-probability ≥ 0.6 on 39% of windows at
  85% accuracy (≥ 0.8: 8% at 95%). Usable for selective adaptation.
- Members disagree: some member is right on 86% of windows, the ensemble 66%.
- Interfaces: Graz 66 vs BrainHero 66 (al); ss BrainHero 76 vs Graz 65.

## 4. Evaluation honesty

`all_loso` tests the 8 `dev_loso` participants + P04/P09; `seen_session`'s
P10 half is `dev_session`'s test data. Only P04/P09 in `all_loso` are
untouched by selection (`mix_probs.py` prints them). `dev_session` has 4
cells and swings 2–6 points. Rule: adopt only if dl gains > 1 on a plateau
(seeds/weights), ds does not drop, and al/ss do not drop; report P04/P09.
EEGNeX (+1.2 dl, ±0 al) is the cautionary example.

## 5. Code status (reviews 19 and 20 October)

No correctness bug in the shipped package or the new numerics (FFT bank
bit-identical, causal BN causal, ERP leak-free, `mix_probs` exact). Fixed
20 Oct: AdaBN "k = 0" rows relabelled "AdaBN off"; CBraMod weights pinned
(rev `584cdc41…`) and loaded strictly (only `final_layer.*` missing);
trained-net cache only for `*_bn*` configs, on CPU; `build_sealed.py`
validates `train`; `replay.sbatch` strips a trailing slash. Open:
- rebuild evidence comes from a dirty tree (`5f9b9ab+uncommitted`); re-run
  `build_sealed` from a clean commit and record source hashes;
- GPU training is not deterministic (no deterministic flags); state the
  audit tolerance, or set flags and show A100 repeatability;
- `bank_key` concatenates bytes without lengths (theoretical collision),
  `Solver.name` still "TangentShallowEnsemble" (cosmetic);
- `mix_probs` does not print which files (directories/devices) it used.

## 6. Literature that transfers (survey 20 Oct)

- Foundation models fine-tuned (not probed) are competitive on cross-subject
  MI: REVE > CBraMod on BCIC-IV-2a (0.64 vs 0.51 balanced accuracy) and
  mental arithmetic (0.77 vs 0.73) [REVE, arXiv 2510.21585]; CBraMod best
  in LOSO on BNCI2014-001 [EEG-FM-Compass, 2601.17883]. Recipes: linear probe
  → full fine-tune, layer-wise LR decay 0.65–0.9, label smoothing 0.1,
  cosine LR, LoRA, Mixup, averaging ≥ 5 fine-tuned weights (+≈1.5)
  [REVE; CBraMod 2412.07236; MI audit 2609.23924]. ST-EEGFormer (MIT,
  braindecode hub) won NeurIPS 2025 EEG Challenge 1.
- Label shift breaks Riemannian recentering; SPDIM/OSPDIM learn a per-domain
  SPD bias by information maximisation [ICLR 2025, 2411.07249; 2608.05315].
  Our early-window deficit is consistent with this.
- TTA for EEG: T-TIME (entropy + adaptive marginal regulariser, ensemble)
  [2412.07228]; online alignment + AdaBN + entropy minimisation
  [Wimpff 2024, 2311.18520]; backprop-free augmentation averaging
  (+1.3–1.5) [2601.07556].
- TSMNet with SPD domain-specific batch norm [Kobler, NeurIPS 2022].
- Winners (BEETL 2021, NeurIPS 2025 HBN, Kaggle HMS): alignment +
  heterogeneous ensemble + pretraining.
- Multi-class-per-session public data: BNCI 004-2015 (used), BCI Comp III-V
  (MI + word generation), Shin 2017 hBCI (MI or arithmetic vs rest).

## 7. Plan to the top three (ranked by expected value per day)

1. **Foundation-model members (highest).** Confirm CBraMod on 3–5 seeds
   (Galvani 2982784: 8 configs × 3 seeds, aligned/unaligned, lr, epochs,
   scratch control). Then the recipe: label smoothing, cosine, layer decay,
   LP→FT, weight averaging over seeds. Then REVE fine-tuned (licence check,
   MOABB in its pretraining must be declared) and ST-EEGFormer for diversity.
   Package: plain-torch port in `submission.py` (pattern: FBC class) or a
   verified braindecode path; GPU inference is allowed; keep ≤ 30 min.
2. **Label-shift-aware causal reference.** Weight each window's covariance
   in the running mean by rebalanced ensemble posteriors so the reference is
   class-balanced (v2 and the EA input streams). Target: windows 0–30.
3. **Selective information-maximising TTA.** Per-session bias/head update
   from windows with confidence ≥ 0.6 only, with a marginal-entropy term
   (T-TIME/SPDIM), reset per session.
4. **Calibrated fusion** (per-member temperature on out-of-fold
   probabilities) once foundation-model members join.
5. Multi-dataset pretraining of the convnets on 2-class-per-session data.

Operational: worker-verify every candidate in warm-up before 25 Oct 02:00
CEST; sealed opens 28 Oct (1/day, ≤ 5 uploads planned, best of last five).
