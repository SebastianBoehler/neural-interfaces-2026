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
With all seeds (Galvani 2984514 seeds 1–4, MPS 0 and 5): `all_loso` 65.07
(P04 46.8, P09 68.1) vs 64.92 (48.5 / 69.6); `seen_session` 70.2 vs 71.6.
Paired seed 5 (same seed, eight dev participants): 59.3 trained without
P04/P09 (`dev_loso`), 56.2 with them (`all_loso` folds).

**New selection protocol `dev_loso_all`** (`mix_probs.py`): the `all_loso`
folds of the eight dev participants only. Training includes P04/P09, as in
the sealed models; they are never scored, so they stay untouched for
evaluation. r6nf 69.37 → + CBraMod 70.25 (+0.9, vs +2.2 on `dev_loso`).
From now on new members are selected on `dev_loso_all` and `dev_session`
(with `dev_loso` reported), which requires their `all_loso` runs.

LaBraM baseline recipe, seeds 0–1, added to r6nf: dl 67.76 (+0.6), ds 81.74
(+1.2): within noise.

### Clean rebuild (step 5)

Galvani 2984724 rebuilt r6nf from commit `7f22d21` (`NI26_SOURCE_REV`; the
workspace is an rsync copy, so `git_dirty` reads true there): agreement with
the harness 1.0; against the staged r6nf 3 / 755 official-test and 5 / 2,655
windows differ (GPU training is not bit-deterministic; the earlier rebuild
differed on 1 / 2,655); official replay on CUDA passes
(`runs/sealed/ensemble_all_r6nf_clean_7f22d21*` on Galvani).

### Queued overnight (20 → 21 Oct)

| Job | Configs | Protocols |
|---|---|---|
| 2984802 / 2984803 | FBCNet aligned / unaligned variants: stride 2, stride 8, 16 or 64 spatial filters (all packageable), seeds 0–2 | dl, ds, all_loso |
| 2984808 / 2984809 | REVE fine-tuned (aligned, z-like input clamped ±15; lr 1e-4, and lr 3e-4 with layer decay 0.8), seeds 0–2 | dl + ds / all_loso |
| 2984618 | LaBraM own recipe (lr 5e-4, layer decay 0.65), seeds 0–2 | dl, ds |

REVE: `brain-bzh/reve-base` rev dc2a075c (69.6 M parameters), REVE
Responsible Use License v1.0: research and commercial use, BCI research
explicitly allowed; a shipped fine-tune must include the licence, state the
modification and identify the checkpoint (§5.1). Weights copied to Galvani
from the local cache of 19 Oct (sha256 8ecc6506…), run with `HF_HUB_OFFLINE=1`.

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

### Overnight results (21 Oct, partial: the job cap blocked most of the queue)

LaBraM, own recipe (2984618, seeds 0–2) added to r6nf: dl 65.68 (−1.5), ds
81.2: rejected; LaBraM closed.

FBCNet variants, aligned, seed 0 (2984802_0), each added to r6nf as one
extra family (weight 1/6):

| Added to r6nf | dev_loso | dev_session | dev_loso_all |
|---|---:|---:|---:|
| none | 67.18 | 80.49 | 69.37 |
| control: FBCNet-EA weight ×2 / FBCNet-raw ×2 | 67.23 / 67.38 | 81.16 / 79.73 | 69.95 / 70.42 |
| stride 2 (alone 56.0 / 64.3) | 68.12 | 82.10 | 71.58 |
| 64 spatial filters (alone 57.7 / 69.6) | 68.53 | 81.14 | 71.31 |
| stride 8 (alone 59.9 / 69.8) | 68.27 | 81.63 | 70.73 |
| 16 spatial filters (alone 55.3 / 63.0) | 68.10 | 80.32 | 70.01 |
| stride 2, seeds 0–2 (dl/ds), seed 0 (all) | 68.39 | 81.32 | 71.58 |

Stride 2 and 64 filters beat the upweighting control by ≈ 1 on `dev_loso`
and `dev_loso_all`: candidate diversity, not yet a decision (one seed on
`dev_loso_all`). Waiting for seeds 1–2 (2984802_1/2, running) and the
unaligned variants (2984803); REVE (2984808/2984809) has not started.

### Why participants fail (21 Oct; r6nf, `all_loso`)

Own-data ceiling: shrinkage LDA on per-channel log band power (1–4, 4–8,
8–13, 13–30 Hz), 5-fold CV within each session. Class-contrast similarity:
correlation of a participant's z-scored class-mean band-power contrasts
with each other participant's.

| | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 | P09 | P10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| own-data ceiling | .72 | .70 | .66 | **.58** | .67 | .71 | .82 | .84 | .81 | .72 |
| r6nf LOSO accuracy | .66 | .80 | .64 | .48 | .67 | .75 | .79 | .80 | .69 | .60 |
| contrast corr. with others | .36 | .31 | .37 | .24 | −.08 | .22 | .36 | .40 | .34 | **−.40** |

P04 is a weak-signal participant (lowest own-data ceiling; every single
feature ≤ 0.54), not a transfer failure: no structural fix expected there.
P10 (and P05) are transfer failures: class patterns anti-correlated with the
group; r6nf recalls MI 24% on P10 and predicts MI on 16% of its windows even
after rebalancing. Unsupervised per-participant boundary adaptation is the
remedy class, but T3A and self-training failed because pseudo-labels are
wrong exactly there. Open problem.

### Overnight 21 → 22 Oct: all queued jobs finished

FBCNet variants, seeds 0–2, each added to r6nf as one family (weight 1/6):

| Variant | aligned dl / ds / dl_all | unaligned dl / ds / dl_all |
|---|---|---|
| stride 2 | 68.39 / 81.32 / 71.26 | 68.54 / 81.67 / 70.96 |
| stride 8 | 68.53 / 81.91 / 70.85 | 68.96 / 81.48 / 71.16 |
| 16 spatial filters | 68.67 / 80.29 / 71.05 | 68.50 / 80.41 / 71.19 |
| 64 spatial filters | 68.69 / 81.91 / 71.52 | 68.91 / 81.48 / 71.06 |

r6nf: 67.18 / 80.49 / 69.37. All eight add +1.2…+1.8 dl and +1.5…+2.2
`dev_loso_all`, ds never drops: a plateau. Pooled, one family of the four
aligned (or unaligned) variants gives 68.6 / 81.4 / 71.4 (68.9 / 80.4 /
71.4); a second pooled family adds nothing; all eight at full weight hurt
(66.6 / 80.7 / 67.3). The upweighting control (existing FBCNet families ×2)
gives at most 67.4 dl / 70.4 dl_all, but a fair control is new seeds of the
standard FBCNets: seeds 5–7 aligned/unaligned queued (Galvani 2988708, MPS
`probs_ctrl`).

REVE fine-tuned (2984808/2984809, seeds 0–2), alone dl 53.8–55.7, ds
55.2–57.7, all_loso 51.2–52.3 (P04 0.40–0.45, P09 0.55–0.61). Added to r6nf:
lr 1e-4 67.71 / 79.57 / 70.81, layer decay 67.42 / 82.46 / 69.75; on top of
the pooled variant families it lowers everything (67.9 / 81.1 / 70.9). Not
better than a cheap FBCNet variant at 70 M parameters: dropped.

### Warm-up board, read honestly

Training on train + validation (8 participants, allowed) instead of train
only lifts the r6nf mix on the warm-up test from 47.3 to 50.5 (P04 42.5 →
45.7, P09 61.6 → 64.8); every member gains 1.5–3. P04 (6 of 8 cells) has an
own-data ceiling of ≈ 0.58, so a cell mean above ≈ 0.62 needs cross-participant
P04 accuracy at or above what a model trained on P04 itself reaches: entries
≥ 0.62 very likely trained on P04/P09. Only the ~5 entries at 0.57–0.60 are
plausibly honest and ahead, by ≈ 7–10 points on P04.

### Control with fresh standard seeds (Galvani 2988708, seeds 5–7)

| Added to r6nf | dev_loso | dev_session | dev_loso_all |
|---|---:|---:|---:|
| none | 67.18 | 80.49 | 69.37 |
| 3 more seeds inside the existing FBCNet families (same weight) | 68.34 | 80.88 | 70.85 |
| one extra family of fresh standard FBCNets (aligned + unaligned) | 68.48 | 80.90 | 71.23 |
| one extra family of FBCNet variants (r8v) | 69.00 | 80.88 | 71.55 |

Most of the gain is more fresh FBCNets (+1.2 dl, +1.5 dl_all), some is the
extra family weight, and the variants add +0.3–0.5 (within noise, same sign
on both). r8v stays the candidate. Sealed build from `967d2b2`: Galvani
2988950; seen_session for the eight variants: running locally (capped).

### Correction (22 Oct): FBCNet filter banks were trainable since `fce02e9`

`param_groups` (layer-wise decay helper, added 20 Oct) called
`requires_grad_(True)` on every parameter, which unfroze FBCNet's FIR filter
bank (frozen at build in braindecode 1.8.1). Every FBCNet trained since then
learned its filters. Found because r8v ran 2.1× slower than r6nf (66.6 vs
31.6 ms/window, same Mac): its 18 FBCNets had 18 distinct filter banks, so
the shared FFT bank (R7) no longer applied; the staged r6nf has one.

Affected and invalid until rerun: the FBCNet variants and the r8v selection
and evaluation numbers above (their FBCNets had trained filters, the r6nf
FBCNets in the same mixes did not), the fresh-seed control (seeds 5–7), the
task-period crop experiment (crop FBCNet trainable, baseline frozen: the
rejection is confounded), the r8v build (2988950) and the "clean" r6nf
rebuild (2984724; its 2,650/2,655 agreement did not reveal it because sealed
packages fit every window). Not affected: CBraMod, LaBraM, REVE (only an
unused mask buffer was unfrozen), ShallowConvNets, v2, the staged packages.

Fix: `param_groups` records the build-time `requires_grad` flags and never
unfreezes a frozen parameter (checked: FIR unchanged after training; CBraMod
212/213 trainable). Affected probabilities moved to
`probs_buggyfir_2026-10-22`. Reruns: Galvani 2989010/2989011 (variants, dl,
ds, all_loso), 2989016 (variants, seen_session), 2989017 (standard seeds
5–7). Lesson for the build check: also compare per-member weights against
the reference package, not only predictions on fitted windows.

### After the fix (22–25 Oct)

Frozen-filter variants add ≈ 0 (dl 66.8–67.3, dl_all 69.6–70.1): the r8v
gain came from **learned FIR filter banks**, now an explicit option
(`AlignedNet(train_fir=True)`, configs `*_tf`). The pre-fix probabilities are
exactly that and were relabelled (`probs_tf`). r8t = r6nf + eight
learned-filter FBCNets: dl 69.0 / ds 80.9 / dl_all 71.6, all_loso 66.6 (P04
48.7, P09 71.2), seen_session 70.6 (−1.0, all P04; a frozen-filter family and
plain FBCNet upweighting drop P04 alike). Expected sealed effect ≈ +0.5.
`scripts/sealed_r8v.json` (built with the bug) is replaced by
`scripts/sealed_r8t.json`. Package: input spectra shared across nets and
filter spectra cached (bit-identical; r8v-style package 204 → 166 ms/window
on a loaded Mac, r6nf 97).

Clean r6nf rebuild from `1631413` (Galvani 2989027): FIR banks identical to
the staged package (90/90), v2 max |Δw| 2e-11, 1 / 2,655 windows differ,
mean |ΔP| 0.003, 64 ms/window: audit-grade.

### Ceiling: our members trained within each session (`within_session`)

Five contiguous blocks per session, train on four: v2 78.8, FBCNet-EA 74.6,
FBCNet-raw 71.8, ShallowConvNet raw 60.6 / EA 42.6 (too few windows). Mix
v2 + FBCNets 81.9 vs r6nf all_loso 64.9: P10 88.8 vs 60.6, P09 92.7 vs 69.6,
P04 66.9 vs 48.5, P07 88.5 vs 79.3. ≈ 17 points sit in participant-specific
decision boundaries.

Label-free route by clustering (`experiments/cluster_bound.py`): per
session, k-means/GMM on 5–20 PCA dims of session-centred v2 features: 0.37–0.39
balanced accuracy even with the oracle cluster→class map (chance 0.33;
global model 0.65). Classes are not the session's main structure; closed.

### 25 Oct: crops retested, fresh Shallow family, personal models

Crops with frozen filters (Galvani 3002612, seeds 0–2): alone dl 51.4 (EA) /
57.0 (raw), added to r6nf as one family dl 67.28 / ds 80.17 / dl_all 69.59:
rejected (now unconfounded). Fresh ShallowConvNet family (3002618, seeds
5–7): 67.15 / 79.41 / 70.01: no gain; the fresh-family gain is specific to
learned-filter FBCNets.

Personal models (`experiments/personal_bound.py`, oracle participant): a v2
trained on P10's session 1, mixed into r6nf on `dev_session`: 80.5 → 83.0 /
83.5 / 81.2 at weight 0.15 / 0.3 / 0.5 (personal alone 74.2).

Recognition (`experiments/identify.py`; 7-band tangent at the global
reference, PCA 100, multinomial LR over participants; causal mean of window
log-probabilities): seen P10 sessions 2–3 correct from 5 windows on
(posterior 0.79–0.94); P04 sessions 2–3 wrong (s3 confidently, 0.90 after 5
windows); unseen participants reach max posteriors 0.25–0.91 (P02 0.81,
P04 s1 0.79). No threshold separates them: hard routing is unsafe.

Soft routing (`experiments/soft_route.py`): P = (1 − λ) r6nf + λ Σ_p q_t(p)
personal_p. dev_session 80.49 / 81.71 / 84.55 / 84.04 and dev_loso_all
69.37 / 69.38 / 68.02 / 67.22 at λ 0 / 0.1 / 0.2 / 0.3; the selection
objective (mean of seen and unseen proxies) peaks at λ 0.2 (+1.4). λ fixed
at 0.2 before evaluation; evaluation running.

Soft-routing evaluation (λ 0.2, fixed before): `seen_session` 71.62 → 72.64
(P04 59.72 → 59.98, P10 83.51 → 85.29); `all_loso` 64.92 → 63.90 (P04
48.54 → 47.60, P09 69.58 → 71.66). Selection overestimated the seen gain (P10
is the easiest participant to recognise); with the sealed set about half or
more unseen sessions the net is ≈ 0 and `all_loso` drops: not adopted.
Recognition across sessions is the limit, not the personal models.

### Worker replica (Apptainer, Galvani)

`scripts/cloud/worker_image.def` rebuilds the official worker image recipe
(python:3.12-slim, torch from the cu130 index, the official unpinned
requirements; resolved 25 Oct: torch 2.14.1+cu130, braindecode 1.8.1, numpy
2.5.2, scipy 1.18.1, benchopt 1.10.0, neuroai 0.4.0); `worker_replay.sbatch`
runs the official ingestion + scoring inside it on an A100 with CUDA
visible. 3002631: staged r6nf 0.9973 (66.7 s), staged r6f 0.9987 (66.8 s),
`warmup8_cbramod` 0.4586 on CUDA (121.3 s), files unchanged, scores equal to
the local replays: the substitute for the missed warm-up uploads.

r8t built from `c625ecc` (3002611): harness agreement 1.0, the ten r6nf
FBCNets' filter banks identical to the staged package, the eight new ones
learned; 100 ms/window on that node's CPU (r6nf rebuild 64 ms on another);
CUDA replay passes (ZIP `46be1133`); worker replica 3002634.
r8t in the worker replica (3002634): 0.9987, files unchanged, 107.4 s for
755 windows (r6nf 66.7 s).

### Learned filters complement frozen ones; a 4-net family suffices

Learned-filter standard FBCNets (3002616, seeds 0–3): added to r6nf as one
family 68.9 / 80.6 / 71.4 (dl / ds / dl_all), like r8t (69.0 / 80.9 / 71.6);
*replacing* r6nf's frozen FBCNets with them loses (66.4 / 78.3 / 69.2): the
gain is the combination. r8t's complete `seen_session` (8 nets, 3002617):
70.46. Four-net families all match the eight-net one (aligned variants,
seeds 0/1/2: dl 68.4–68.6, dl_all 71.1–71.3; unaligned 68.7–68.9 / 70.9–71.5;
two aligned + two unaligned learned-filter standard FBCNets, seeds 0–1:
69.11 / 80.91 / 71.43). Fixed as **r8t4** (simplest, half r8t's extra cost):
all_loso 66.31 (P04 47.77, P09 70.67), seen_session 70.93 (r6nf 64.92 /
48.54 / 69.58 and 71.62). Built as Galvani 3002636 (config
`scripts/sealed_r8t4.json`), worker replica 3002637.

Fresh seeds of the learned-filter family (never used for a decision): eight
variants, seed 3 / 4 / 5: dl 69.43 / 68.96 / 68.95, ds 81.14 / 81.39 / 81.37,
dl_all 71.60 / 71.60 / 71.44; standard seeds 2–3 (r8t4 layout): 68.95 /
80.35 / 71.69. The gain reproduces.

GPU path for all nets (`submission.py`, `scripts/device_agreement.py`): CPU
path bit-identical to the staged code; on CUDA in the worker replica labels
agree on all 2,655 windows (r8t4: max |ΔP| 1.2e-7, 31 vs 83 ms/window; r6nf:
9.5e-8, 25 vs 64 ms/window).

### Participant-aware objectives (25 Oct, running)

Sealed composition (official description: later sessions of the ten public
participants + ten new participants, six sessions each): ≈ 46 seen later
sessions vs ≈ 60 unseen ones, i.e. ≈ 57% unseen. Expected effects: r8t4
0.57 × 1.4 − 0.43 × 0.7 ≈ +0.5; soft routing 0.43 × 1.0 − 0.57 × 1.0 ≈ −0.1
(stays rejected).

v2 with participant-balanced weights or GroupDRO-style reweighting
(`TangentSpace(group=...)`): alone dl 56.6–56.8 (plain 56.2), ds 68.3; in
r8t4 dl 69.0–69.05 / ds 80.91 / dl_all 71.1–71.2 vs 69.11 / 80.91 / 71.43:
no gain (C 0.1 leaves little to reweight). Existing members as additions to
r8t4 (v1, v2_C1, EEGNeX, IFNet, TIDNet): all lower dl and dl_all.
FBCNet/ShallowConvNet with `pbal` / online GroupDRO (η 0.01, 0.1), seeds
0–4, all protocols: Galvani 3004101–3004105, 3004131 (pending behind the
partition queue); FBCNet η 0.1 and pbal seed 0 on the Mac.

FBCNet with participant objectives, seed 0 (Mac): GroupDRO η 0.1 alone dl
57.7 / 57.1, ds 68.4 / 67.6, all_loso 53.1 / 53.2 (aligned / unaligned;
standard seed 0: 59.1 / 60.4 dl); balanced loss 59.3 / 60.9, 70.5 / 69.7,
54.8 / 56.4. In r8t4 (dl / ds / dl_all; staged 69.11 / 80.91 / 71.43):
replacing standard seed 0 by DRO 69.00 / 80.65 / 71.23, by balanced 68.90 /
81.17 / 71.44, control without seed 0 68.80 / 81.14 / 71.28; as an extra
family DRO 68.68 / 81.94 / 70.90, balanced 68.78 / 81.16 / 71.70. Neutral on
one seed; the five-seed Galvani arrays (3004101–3004105, 3004131) decide.

### 26 Oct: five-seed objectives, family weights, LDA head, calibration

Five seeds (Galvani 3004101–3004105, 3004131), in r8t4 (dl / ds / dl_all,
selection score = mean of ds and dl_all; r8t4 69.11 / 80.91 / 71.43, 76.17):
balanced loss replacing the FBCNets 69.00 / 80.91 / 71.54 (76.22), the
ShallowConvNets 76.00, both 75.94, as an extra FBCNet family 76.34; GroupDRO
η 0.01 / 0.1 replacing the FBCNets 75.78 / 75.66, the ShallowConvNets 75.67 /
74.74, both 75.16 / 73.86, extra family 76.63 / 76.38 (ds only). Closed:
weighting participants does not improve generalisation to new ones here.

Family weights re-checked on the new selection protocols (v2 0.5–1.5,
ShallowConvNets 0.5–1, fixed FBCNets 1–1.5, learned family 1–2): selection
score 75.97–76.64, no consistent direction; current weights on the plateau.

v2 with a shrinkage-LDA head: alone dl 57.1 (LR 56.2), ds 67.7, all_loso
53.6; added to r8t4 72.94 (weight 1) / 74.72 (0.5): its posteriors are far
more peaked. Calibrated fusion (`experiments/calibrate.py`: one temperature
per family by log-loss on dev_loso_all, applied unchanged): temperatures
1.9–3.6 for every family (LDA 9.2), r8t4 −0.26 dl / +0.90 ds / −0.76
dl_all; with LDA still below r8t4. All members are overconfident across
participants by similar factors, so calibration barely changes the mix.
Closed.
