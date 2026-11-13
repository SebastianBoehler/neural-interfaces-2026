# Inventory: what `cand_v51b_own` executes, and what the lean tree drops

Subject: `submissions/ensemble_v5b/submission.py` (sha256 `0259d1ea…`, 1,081 lines) with the data files of
`runs/sealed/cand_v51b_own` (`weights.npz`, 996 arrays; `teacher.pt`; 30 `sbn_*.pt`).

Two sources: reading the code against the keys in `weights.npz`, and line coverage of the original while it
streamed 23 parity cases (4,559 windows: every public session, external people, a person switch, a long
session, corrupted windows, two other montages; `lean/evidence/coverage_original.txt`). 72 of 774 statements never
ran. Line numbers below are the original's.

## 1. What is live

| Part | Data | State in this package |
|---|---|---|
| core1 (`AlignedTangent`, `ts_`) | 17 bands, `car` 1, `nsub` 2, real | live, weight 0.125 of 1.875 |
| Hilbert core (`ts2_`) | 7 bands, `car` 1, `nsub` 2, `hilbert` 1 | live, weight 0.125 |
| ShallowConvNet exports | nets 0–4 whitened input, 5–9 scaled input | live, 0.025 each |
| FBCNet exports | nets 20–21 whitened, 22–23 scaled; two distinct FIR banks per group | live, 0.03125 each |
| Teacher (`Jit`) | net 34, `teacher.pt` | live, 0.625 |
| SessionBN traces | nets 35–64: 24 with the shared FIR bank (`SharedFIRSJit`: 18 FBCNet, 6 FBLightConvNet), 6 SincShallowNet (`SJit`) | live, 0.0208 each |
| FBCNet exports 10–19, SincShallowNet exports 24–28, FBLightConvNet exports 29–33 | weight 0 | **never loaded** |
| Input streams | one whitening stream (all 65 nets carry the same `G`, `k` 5, `shrink` 1e-3), one scaling stream | live, two groups |
| Rebalancing, SessionEM, TSEM, gated early boost | `rb_alpha` 1.15, `em_boost` 2, `tsem_boost` 4 | live |
| Routing v3 + identity gate + own-space stage | `route_*`, `bio_*`, `own_*`, 10 public people | live |
| Montage mapping (`channel_map`) | — | live; the public montage equals the training order, so the mapping itself only ran in the forced cases |

## 2. Removed, and why it is safe

"Safe" means: with these data files the removed code cannot run, or it computes the same values another way that
the parity run confirms bit for bit.

**Code for other packages (configuration decides; never executed here)**

| Original | Lines | Why dead |
|---|---|---|
| `_log`, `_upper`, `AlignedTangent._cov`, the `nsub == 0` body of `_features` | 68–74, 155–181 | both cores have `nsub` 2; this is the pre-6-Nov core |
| `car` as an option | 110, 162 | both cores set it; `check_package` now requires it |
| classes `FBL`, `Sinc` | 407–450 | only nets 24–33 use them and their weight is 0; `build_net` raises on such an export with non-zero weight |
| `_sjit` fallback to the generic `SJit` | 329–336 | all 24 fingerprinted traces pass the check; a failing check now raises instead of silently taking the slower path |
| the shape / order / dtype / layer-name parts of the trace fingerprint | 301–308 | the class name and the SHA-256 of the FIR coefficients are kept; the rest only chose between the two paths above |
| `None` handling for every stage and for the second core in `Ensemble` and `Solver.load_model` (`rb`, `em`, `tsem`, `route`, `tangent2`, `"bio_V" in p`, `"own_beta" in p`, `em_boost` / `tsem_boost` absent, `tsem_lam` 0) | 470–472, 557–559, 571, 771–774, 901–912, 1018–1025, 1049–1065 | every stage is present; `check_package` raises if a weights file says otherwise |
| `assert int(p["route_v3"]) == 1`, the `Rebalance` assert, shape asserts | 605, 764, 772–774, 908 | moved into `check_package` |
| `return []` for an ensemble without nets | 984 | there are 44 live nets |

**Batch handling.** The evaluator calls `predict` with one window (`benchmark_utils/streaming.py`). The original
collected per-window lists for a batch, stacked them, ran the nets on the batch and looped the stages over it
(940–1027). The lean `Ensemble.step` handles one window; `predict` loops over the batch. For one window the
arithmetic is the same; for a larger batch (never sent) the nets would now see the windows one by one, which is
also what the SessionBN members need.

**Guards that cannot trigger**

| Original | Lines | Argument |
|---|---|---|
| `try / except (LinAlgError, FloatingPointError, ValueError)` around the class-direction gate | 839–846 | the block holds sorting, medians, products and norms on arrays of fixed shape; nothing in it raises these (NumPy does not raise on floating-point errors unless asked) |
| the same around the personal head mix | 855–864 | as above; the non-finite check on the result stays |
| the same around the identity gate | 848–851 | its only failure is a singular matrix in `solve`; the matrices depend on the window count alone, and B, W, S are positive definite (smallest eigenvalues 3.96, 1.09, 4.72), so B + W + S/n and the joint matrix are positive definite for every n |
| `try / finally: wait(tangent)` around the nets | 993–996 | it only ordered state after an exception in the nets; such an exception ends the evaluation either way |

**Duplicates folded**

- Three softmax implementations (`_softmax`, `_softmax_rows`, the tail of `SessionEM._lik`) → `softmax`; `_lik` →
  `lda_posterior`. Same operations in the same order.
- `TSEM.view` and `Route3.view` (each subtracting `tsem_m` again) → the 256-PC view is computed once in
  `Ensemble._tangents`, still as two separate projections.
- The montage mapping and `nan_to_num` ran twice per window (1007–1016) → once, in `to_windows`.
- `Route3.bind` (deep copy of the three stages) and `TSEM.gate_src` (a reference from TSEM to "its" SessionEM) →
  `Tail`, a class holding the three stages; the model builds two. The gate is passed as an argument.
- `SessionEM.gates` kept every window's gate; only the current one is read → one float.
- `Route3` held both gates' logic and the stage; now `ContrastGate`, `IdentityGate`, `OwnStage`, `Route`.
- Unused attributes (`Shallow.dev`, `Jit.dev`, `FBC.dev`, `Bio.last`, `Route3.G`) and the 39-line history
  docstring.

## 3. Kept although it never ran on the public data, and why

| Kept | Why |
|---|---|
| Non-finite mix → uniform posterior | depends on the data; it did not occur, but cannot be excluded for sealed windows |
| `LinAlgError` and non-finite fallbacks in SessionEM and TSEM | data-dependent (TSEM's non-finite fallback did run in the stress case) |
| `ContrastGate`: no decision when a class has no window or the scores are not finite | data-dependent |
| `IdentityGate`: no decision on non-finite scores | one line; not reached with finite input, kept as the only barrier between a bad spectrum and the routing |
| `OwnStage` degenerate-window test, the `try / except Exception` around the stage and its counters | data-dependent (`eigh` can fail); the stage is optional by design |
| The `[own-stage]` stderr line per session and `__del__` | it is the only evidence in a sealed log that the stage ran; the evaluator drops a session's copy without a call, hence `__del__` |
| `channel_map` and its `TypeError` fallback | the sealed montage order is not known; `meta["ch_names"]` may be absent |
| `threadpoolctl` `ImportError` fallback | the worker image cannot be inspected from here |
| Worker thread for the cores, with the process-id check | run time on the worker (cores on CPU while the nets use the GPU); the check costs two lines and a wrong guess would hang a sealed run |
| FIR fingerprint of the shared-bank traces | sharing one filter-bank output is only valid for traces with identical filters |
| FFT caches of the FBCNet exports | run time; values unchanged |
| `Jit.__deepcopy__` (sessions share the teacher) and `SJit.__deepcopy__` | per-session copy cost and state isolation |

## 4. Not touched: data

The data files are copied unchanged (the builder checks the digests). `weights.npz` carries arrays no code reads:
the 20 zero-weight net exports (2.7 MB), `route_pop`, `route_b`, `route_names`, `route_slot_a`, and 64 redundant
copies of the whitening reference `G` (0.6 MB), together 3.3 of 62.7 MB. A rebuilt `weights.npz` could drop them;
that changes a pinned data file and is left for the next package build.

## 5. Line counts

See [PARITY.md](PARITY.md), last section.
