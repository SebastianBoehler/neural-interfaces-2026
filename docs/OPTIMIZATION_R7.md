# Optimization round 7: review, diversity, test-time BN, runtime

Run 19–20 October 2026 (Europe/Berlin). Reference: `ensemble_all_r6`
(`dev_loso` 67.8, `dev_session` 80.5, `all_loso` 66.0, `seen_session`
70.8). Selection on `dev_loso`/`dev_session` only. Offline mixes now come
from one tracked script, `scripts/mix_probs.py` (seed-averaged members,
windowed rebalancing α = 1, k = 5, W = 30), which reproduces the r6 row
exactly (67.81 / 80.49 / 65.97 / 70.82). `all_loso` rows also report P04
and P09, the only participants no selection round has used (see the
protocol caveat in `HANDOFF.md`).

## Unaligned architectures as extra members (Galvani 2981487, seeds 0–4)

| Member (alone, 5 seeds, no rebalancing) | dev_loso | dev_session |
|---|---:|---:|
| FBCNet unaligned (r6 member) | 60.2 | 73.1 |
| IFNet unaligned | 57.0 | 64.6 |
| TIDNet unaligned | 52.4 | 56.6 |
| EEGNeX unaligned | 50.4 | 58.3 |

| Mix | dev_loso | dev_session | all_loso (P04 / P09) | seen_session |
|---|---:|---:|---:|---:|
| r6 | 67.8 | 80.5 | 66.0 (49.4 / 72.3) | 70.8 |
| + IFNet | 67.7 | 81.5 | – | – |
| + TIDNet | 67.8 | 80.4 | – | – |
| **+ EEGNeX** (selected on dev) | **69.0** | **81.1** | 66.0 (49.5 / 71.1) | 70.3 |
| + IFNet + EEGNeX | 68.2 | 81.9 | – | – |
| + all three | 67.8 | 79.3 | – | – |

EEGNeX was a dev plateau: every single seed adds +0.6 to +1.2 `dev_loso`
and +0.1 to +0.9 `dev_session`; weights 0.5–1.5 all add +0.5 to +1.2. The
evaluation does not confirm it (±0 `all_loso`, −0.5 `seen_session`; in the
no-Scherer mix +1.4 / −1.6), so the dev gain is treated as selection noise.
**Not adopted**: it would also need a plain-torch port and more inference
time.

## Causal test-time batch-norm adaptation (AdaBN)

`AlignedNet(adabn=k)`: at test time every batch-norm layer normalises
window t with the per-channel mean and variance of windows 0..t of the
session, blended with the training running statistics as k pseudo-windows
(label-free, reset per session; checked causal and equal to plain eval BN
for k → ∞). The trained nets are reused across k (`AlignedNet._fitted`).
Configs `{shallow,fbcnet}_{ea,raw}_bn{0,5,20,100}`; Galvani 2981495, seeds
0–2 (single seed 0 on MPS agrees).

| Single model, k = 0 → 5 | dev_loso | dev_session |
|---|---:|---:|
| FBCNet aligned (MPS s0) | 59.1 → **64.0** | 70.5 → 75.6 |
| FBCNet unaligned (MPS s0) | 60.4 → 61.5 | 71.6 → 75.1 |
| ShallowConvNet aligned (MPS s0) | 52.3 → 53.0 | 58.9 → 65.0 |
| ShallowConvNet unaligned (MPS s0) | 54.3 → 53.4 | 58.1 → 64.4 |

| Ensemble, all four nets PROTEUS-only, 3 seeds | dev_loso | dev_session |
|---|---:|---:|
| k = 0, rebalanced | 67.0 | 80.7 |
| k = 5 / 20 / 100, rebalanced | 66.0 / 66.4 / 67.5 | 80.4 / 80.7 / 81.5 |
| k = 0, **no** rebalancing | 65.0 | 71.0 |
| k = 5, **no** rebalancing | 64.7 | 78.6 |
| AdaBN on the FBCNets only, k = 5 / 20 | 67.0 / 66.9 | 80.9 / 80.9 |

AdaBN gives large single-model gains, but in the rebalanced ensemble it is
within noise. Without rebalancing it recovers most of what rebalancing
does (`dev_session` 71.0 → 78.6): both correct the same session-level
class bias, so they are redundant. **Not adopted.**

## Scherer for the unaligned FBCNet, and a no-Scherer fallback

| Mix (5 seeds per member) | dev_loso | dev_session | all_loso (P04 / P09) | seen_session |
|---|---:|---:|---:|---:|
| r6 | 67.8 | 80.5 | 66.0 (49.4 / 72.3) | 70.8 |
| r6, unaligned FBCNet → + Scherer | 67.6 | 79.2 | – | – |
| r6 + Scherer unaligned FBCNet as 6th member | 67.8 | 80.2 | – | – |
| **r6n**: r6 without any Scherer member | 67.2 | 80.5 | 64.9 (48.5 / 69.6) | **71.6** |
| r3w (previous fallback) | 66.1 | 77.8 | 64.2 (48.7 / 70.1) | 69.6 |

Scherer does not help the unaligned FBCNet. r6n is the new licence
fallback: equal to r6 on `dev_session` and `seen_session`, −0.6 `dev_loso`,
−1.1 `all_loso`; averaged over the seen/unseen halves 68.3 vs 68.4.

## Faster inference (`submission.py`, same weights, bit-identical)

The ten FBCNets recomputed the same FIR filter bank (9 filters of 45–199
taps) per net. `Ensemble.mixed_proba` now filters each input stream once
per distinct bank (`_fft_bank`, one input FFT per filter length) and
non-finite input samples are zeroed before any state update. Against the
previous code on all 755 official-test windows: max |ΔP| = 0.0, 100%
prediction agreement (`scripts/time_package.py`). Mac CPU, same load:
r6 51 → **35 ms/window**, r6n 50 → 35 ms. A first version used the first
filter's length for every band and changed 0.4% of predictions; the
agreement check caught it.

## Packages (official ingestion + scoring replay, Mac CPU, files unchanged)

| Package | Weights | Replay cell mean | Ingestion | ZIP (`runs/upload/`) |
|---|---|---:|---:|---|
| `ensemble_official_r6f` | `ensemble_official_r6` | 0.524299 | 36 s | `warmup7_ensemble_official_r6f.zip` (`96cfc098…`) |
| `ensemble_all_r6f` | `ensemble_all_r6` (Scherer) | 0.998656 (leakage) | 33 s | `sealed_candidate_ensemble_all_r6f_scherer.zip` (`accc74df…`) |
| `ensemble_all_r6nf` | r3w + 5 unaligned FBCNets (`add_fbcnet.py --raw --share 0.2`, MPS; harness agreement 100%) | 0.997312 (leakage) | 32 s | `sealed_candidate_ensemble_all_r6nf_noscherer.zip` (`1d347187…`) |

## A100-node runtime

The official replay of the old `ensemble_all_r6` ZIP on a Galvani A100
node (job 2981644, EPYC 7302, load ≈ 22) took 423 s, but that run rebuilt
the NeuralBench caches under `/scratch_local` (no `~/.neuralbench` config),
so it is not an inference time. Worker reference: upload 1000615 (round 3)
took 49.8 s on the worker vs 35 s ingestion on the Mac, i.e. the worker is
roughly 1.4–1.9× the Mac. Inference only, `scripts/time_package.py` on
an A100 node (job 2982352, EPYC 7302, 4 threads, node load ≈ 26):

| Package | ms / window | ≈ 20,000 windows |
|---|---:|---:|
| `ensemble_all_r6` (old code) | 98.8 | 33 min |
| `ensemble_all_r6f` | 63.7 | 21 min |
| `ensemble_all_r6nf` | 61.8 | 21 min |

Old and new code agree on all 755 windows on Linux as well. The fast code
keeps the projected sealed time under the 30-minute budget; the old code
did not.
