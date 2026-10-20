# Optimization round 8: foundation-model members, bounds, task-period crops

Run 20 October 2026 (Europe/Berlin), in progress. Reference: `ensemble_all_r6nf`
(`dev_loso` 67.2, `dev_session` 80.5, `all_loso` 64.9, `seen_session` 71.6).
Selection on `dev_loso`/`dev_session` only; mixes from `scripts/mix_probs.py`
(rebalancing α 1, k 5, W 30). Galvani array 2982784 (CBraMod, 3 seeds) waited
all round behind the per-user job limit; everything below ran on the Mac (MPS).

## Closed by an oracle bound or a negative result

| Idea | Test | Result | Verdict |
|---|---|---|---|
| Trial-order prior | class transitions within runs (84 runs) | P(same class next) 0.29, others 0.32–0.39: random permutation of exactly balanced runs | nothing beyond rebalancing |
| Label-shift-aware causal reference (v2) | reference balanced with **true** labels; whole-session reference | dl 56.17 → 56.22 (oracle), 56.53 (non-causal); windows 0–30: 52.3 → 52.3 / 52.7 | early-window deficit is not reference error; ≤ 0.05 reachable |
| Per-session member weighting | best family per stream (oracle), causal agreement weighting | dl 69.1 (fixed mix, r6nf + CBraMod s0) vs 67.1 oracle selection, 65.6–60.8 agreement | mixing beats selection; rejected |
| Selective prototype TTA (T3A) on v2 features | causal class prototypes in a 20/50-dim training PCA, updated from windows with rebalanced ensemble confidence ≥ 0.6/0.7, mixed with β 0.15/0.3 (48 settings) | dl 67.18 base; frozen control ≤ 67.19; every adaptive setting 62.6–66.9; ds 80.5 vs 78.2–81.1 | rejected: confident pseudo-labels still reinforce errors (as self-training in R4); gradient-based selective TTA relies on the same labels and is deprioritised |

## CBraMod (fine-tuned, aligned input) as a member

Plain-torch port (`submission.CBra`) matches braindecode 1.8.1 to 5e-8 in
probability. Cost ≈ 1.6 GFLOP per window and net: 186 ms on one loaded Mac
CPU thread, so the member runs on the harness device (CUDA on the worker,
TF32 off).

| Mix | dev_loso | dev_session |
|---|---:|---:|
| r6nf | 67.2 | 80.5 |
| + CBraMod EA s0, weight 0.5 / 1 / 1.5 / 2 / 3 (family units) | 68.2 / 69.1 / 69.7 / 69.3 / 67.4 | 82.5 / 81.7 / 80.9 / 79.3 / 75.4 |
| + CBraMod unaligned s0 | 66.8 | 82.8 |

| Seed (alone, dl / ds) | + to r6nf at weight 1 (dl / ds) |
|---|---|
| 0: 60.0 / 63.1 | 69.1 / 81.7 |
| 2: 58.3 / 62.3 | 69.3 / 82.7 |
| 0 + 2 averaged, weight 0.5 / 1 / 1.5 / 2 | 68.3 / 69.5 / 69.9 / 69.1 dl; 83.2 / 81.7 / 80.9 / 79.4 ds |

| 1: 58.3 / – | 68.8 / – |
| 0 + 1 + 2 averaged, weight 0.5 / 1 / 1.5 | 68.2 / 69.4 / 70.2 dl |

With all three seeds at weight 1: dl 69.35 (+2.2), ds 81.95 (+1.5).

**Seen-participant weakness.** On `seen_session`, seed 0 lowers the mix
71.6 → 69.9 (P04 59.7 → 56.3, P10 83.5 → 83.5). Alone, CBraMod reaches P04
38.4 / P10 63.7 where ShallowConvNet-EA and FBCNet-EA reach 55.6 / 76.1 and
53.4 / 75.9: the from-scratch nets gain ~20 points when the participant's
first session is in training, CBraMod gains little (the same gap shows on
`dev_session`: 60–63 alone vs 78 for FBCNet-EA). Hypothesis: lr 1e-4 /
wd 0.05 / 30 epochs keeps it near its pretrained features, so it does not fit
participant-specific structure. About half of the sealed cells are seen
participants, so this halves the expected gain. Next: fine-tuning variants
selected on `dev_session` (the seen regime) and `dev_loso`.

| CBraMod variant, seed 0 | alone ds | r6nf + variant (w 1) ds |
|---|---:|---:|
| lr 1e-4, 30 epochs (baseline) | 63.1 | 81.65 |
| lr 3e-4 | 60.0 | 80.35 |
| 60 epochs | 61.3 | 81.64 |
| layer decay 0.75, top lr 3e-4 | 62.6 | 81.68 |
| label smoothing 0.1 | 58.2 | 80.71 |

No variant fits the seen regime better; all mixes are within noise of the
baseline, which stays. Seed weight averaging was not tried: heads start
from different random inits, so a soup needs LP-FT with a shared head.

### Evaluation (choice fixed: CBraMod family, weight 1)

| Mix | all_loso (P04 / P09) | seen_session |
|---|---:|---:|
| r6nf | 64.92 (48.5 / 69.6) | 71.6 (P04 59.7, P10 83.5) |
| + CBraMod s0 | 65.11 (46.9 / 67.6) | 69.9 (P04 56.3, P10 83.5) |
| + CBraMod s5 | 64.96 (47.5 / 68.1) | – |
| + CBraMod s0 + s5 (all_loso), s0 + s1 (seen_session) | 65.03 (47.5 / 67.6) | 69.4 |

**Not adopted.** Selection gains (+1.9–2.2 dl, +1.5 ds on five seeds) do
not transfer: `all_loso` +0.1 with the two untouched participants −1.0 / −2.0,
`seen_session` −2.2. CBraMod alone scores 51.0 / 50.4 in `all_loso` (seeds 0,
5) against 59.9 (seed mean) in `dev_loso`; on the eight shared participants
it is 3.6 lower when P04 and P09 join the training set, while FBCNet and
ShallowConvNet are equal or better with more participants. P04/P09 data are
unremarkable (RMS 17–27 µV, aligned 99.9th percentile 5.4–6.1). Sealed
training uses all ten participants, so this is the regime that matters.
Paired check running: seed 5 on `dev_loso` (same seed, without P04/P09);
Galvani 2984514 adds seeds 1–4 on both evaluation protocols.

### Galvani grid (array 2982784, seeds 0–1, A100, alone)

| Config | dev_loso | dev_session |
|---|---:|---:|
| aligned, lr 1e-4, 30 epochs (selected) | 60.1 / 58.4 | 64.6 / 62.8 |
| aligned, lr 1e-4, 60 epochs | 59.4 / 57.3 | 62.9 / 61.4 |
| aligned, lr 3e-4, 30 epochs | 56.8 / 57.2 | 63.5 / 61.7 |
| aligned, **from scratch** | 46.6 / 48.0 | 46.6 / 50.6 |
| unaligned, lr 1e-4 (30 / 60 epochs) | 50.0 / 47.2, 47.9 / 47.3 | 62.2 / 62.6, 65.7 / 65.6 |
| unaligned, lr 3e-4 / from scratch | 47.9 / 48.1, 45.8 / 47.6 | 58.5 / 58.6, 58.5 / 56.0 |

Pretraining is the source of the gain (+12 dl over the same architecture
from scratch); the A100 reproduces MPS (seed 0: 60.1 vs 60.0) at 6.6 min per
`dev_loso` seed. The two A100 seeds are independent realisations: with all
five seeds (MPS 0–2, A100 0–1) at family weight 1 the mix is dl 69.1 (+1.9),
ds 81.95 (+1.5); the A100 seeds alone give 68.6 / 82.5.

### LaBraM (BSD-3-Clause, `braindecode/labram-pretrained` rev 0563b6c6)

All 41 channels map to LaBraM's position vocabulary (forward with
`ch_names`); the pretrained 16-position temporal embedding is cut to CLS + 4
patches. With the CBraMod recipe (seed 0): alone dl 48.5, ds 49.1; added to
r6nf + CBraMod (dl 69.35 / ds 81.95) at weight 1 / 0.5: 68.7 / 69.4 dl,
82.7 / 82.1 ds: no gain. LaBraM's own recipe (lr 5e-4, layer decay 0.65) is
queued on Galvani (2984618) with the baseline's seeds 1–2 (2984515).

### Warm-up package with the GPU path (`warmup8_r6nf_cbramod_gpu.zip`)

`scripts/warmup8_cbramod.json` (official train split; r6nf members + CBraMod
seeds 0–2, weights 1/6 per family) built on MPS: agreement with the harness
1.0 on the 755 official test windows; official ingestion + scoring replay
(CPU) 0.4586 cell mean, files unchanged, ZIP 60 MB, sha256 `a7a4f313…`.
On the worker the CBraMod nets run on CUDA; the reported duration
distinguishes the GPU path (≲ 2 min) from a CPU fallback (≳ 4 min).

The official score is below `warmup7` (r6f, 0.5243). Decomposition on the
official test windows (diagnosis only, no selection): every family is weak on
P04 (6 of 8 cells; FBCNets 33.6–34.5, chance), CBraMod alone 41.8 beats both
FBCNet families, and the r6nf mix goes 47.3 → 45.9 with CBraMod (P04 −2.3,
P09 +1.1). The gap to `warmup7` is r6f vs r6nf (Scherer members), not
CBraMod; one participant's six cells are not evidence either way.

## Task-period crops (new training data)

The official window is cue + 1 → 5 s, but the task event lasts 8 s (PROTEUS
events: `word`/`sub` duration 8.0; NeuralBench config: "cue at 0 s,
feedback from 3 s, trial end at 8 s"). `scripts/export_windows.py START`
exports windows from the same official pipeline at another offset; the
+2 s export equals the official window's last 3 s exactly (max |Δ| 0.0 µV,
2,655/2,655 rows aligned). `AlignedNet(crops=...)` adds the crops of the
training rows only (no leakage across LOSO folds), whitened with the
official windows' session statistics; test windows are unchanged.

| FBCNet-EA, seed 0, alone | dev_loso | dev_session |
|---|---:|---:|
| official windows only | 59.1 | 70.5 |
| + crops at +2 and +3 s (3× rows) | 56.3 | 64.3 |

In the r6nf mix the crop-trained seed replaces seed 0 at 67.08 vs 67.18
(66.97 without it). Rejected; the ShallowConvNet run was stopped. Likely
mechanism: the classes carry time-locked content (the < 1 Hz waveform of
R1), and later crops add feedback-period signal, so shifted windows are a
distribution shift, not more of the same data. The export option and
`AlignedNet(crops=...)` stay for the record (default off).
