# Parity of the lean package with `cand_v51b_own` (13 Nov 2026)

**Result: exact.** On 5,155 windows in 26 streams the lean package gives the same labels and the same posteriors
as the original, bit for bit, together with every internal value that was recorded. The official warm-up replay
gives the same score to the last digit. Run time is equal within 0.1 %.

| | Original | Lean |
|---|---|---|
| Code | `submissions/ensemble_v5b/submission.py`, sha256 `0259d1ea…f498534` | `submissions/lean_v5/submission.py`, sha256 `315febb8…c45b1c11`, bundled from `lean/src` |
| Data files | `runs/sealed/cand_v51b_own` (main checkout): `weights.npz`, `teacher.pt`, 30 `sbn_*.pt` | the same 32 files, copied by `lean/tools/build.py` (digests equal; `lean/evidence/package_files.sha256`) |
| Package ZIP as the replay builds it | sha256 `4a0bc427…9933bef` (the ZIP uploaded on 12 Nov) | sha256 `f78f9756…62de2bf` |

Environment: Mac (arm64, macOS 27.2), CPU, Python 3.12.13, numpy 2.5.4, scipy 1.18.1, torch 2.14.1, one thread
per library as the package sets it. All jobs ran one at a time through `scripts/run_capped.sh`.

## 1. Method

`lean/tools/record.py` loads a package as the harness does (`Solver.load_model` with the dataset's channel names)
and streams each case the way `benchmark_utils/streaming.py` does: a `copy.deepcopy` of the loaded model,
`reset_state()`, then one window per `predict` call as a float32 tensor of shape (1, 41, 480). After every call
it stores the label and, from the model's internals, the final posterior, the member mix, core1's posterior, the
256-PC view, the unrouted posterior, both gates' (person, score, ramp) and the own-space stage's (person, ramp,
log members). `lean/tools/parity.py` compares two recordings as raw bytes, field by field.

**Tolerance: none.** The target was bitwise equality on CPU and it is met, so no tolerance is applied to labels,
posteriors or internals. This is the appropriate bar here because the lean code performs the same floating-point
operations in the same order through the same library calls; a difference of even one bit would point to a
changed operation. It is a meaningful bar because both packages are deterministic: a second recording of each
equals its first bit for bit (§4).

## 2. Cases and results (`lean/evidence/parity.json`, `parity_guards.json`)

Columns: windows; windows on which the identity gate / the class-direction gate gave a positive ramp; windows on
which the own-space stage acted; public people routed to (index in `route_names`); changes of the routed person /
times the stage switched on or off.

| Case | Windows | Identity | Contrast | Staged | People | Changes | Labels | All recorded values |
|---|---:|---:|---:|---:|---|---|---|---|
| 14 public sessions (`public/P01|1` … `P09|1`), each | 186–199 | all from window 10 | all from window 60 | all from window 10 | the session's person | 0 / 1 | equal | bitwise equal |
| 4 sessions of external people (`unknown/A|1`, `A|2`, `C|1`, `C|2`; Scherer 2015) | 120 each | 0 | 0 | 0 | none | 0 / 0 | equal | bitwise equal |
| `switch`: 95 windows of P04 session 2, then P10 session 2, no reset | 281 | 271 | 221 | 271 | 3, 9 | 3 / 1 | equal | bitwise equal |
| `lapse`: 60 external windows, 100 of P05, 60 external | 220 | 0 | 141 | 141 | 4 | 0 / 1 | equal | bitwise equal |
| `long`: P10 sessions 1–3 as one stream | 577 | 567 | 517 | 567 | 9 | 0 / 1 | equal | bitwise equal |
| `stress`: P04 session 3 with NaN, ±inf, all-zero, flat-channel, 10⁶-fold and constant windows | 188 | 178 | 45 | 174 | 3 | 0 / 7 | equal | bitwise equal |
| `perm`: P02 session 1 delivered in reversed montage order | 189 | 179 | 129 | 179 | 1 | 0 / 1 | equal | bitwise equal |
| `missing`: P02 session 1 with two channels absent from the montage | 189 | 179 | 129 | 0 | none | 0 / 0 | equal | bitwise equal |
| **Total of the 24 cases above** | **4,779** | | | | | | **equal** | **bitwise equal** |
| `extreme`: P04 session 3 with windows scaled 10³⁰-fold, set to ±3·10³⁸, scaled 10⁻³⁰-fold | 188 | 178 | 0 | 178 | 3 | 0 / 1 | equal | bitwise equal |
| `faults`: P04 session 3 with injected failures (every 5th matrix inversion raises; the own-space members raise on every 7th routed window) | 188 | 178 | 128 | 153 | 3 | 0 / 51 | equal | bitwise equal |

What the cases show:

- **A returning person is recognised** in all 14 public sessions (all ten public people are in the galleries).
- **Nobody is recognised** in the four external sessions: both gates stay at zero for all 480 windows, so the
  output is the unrouted chain.
- **Recognition changes mid-session** in `switch` (the routed person goes from P04 to P10, with three changes
  around the transition) and in `lapse` (the identity gate never opens after an unknown start; the class-direction
  gate alone recognises P05, so the routing rests on the other gate).
- **The fallback paths agree**: non-finite and degenerate windows (`stress`, `extreme`, `missing`: all 189 windows
  degenerate for the own-space stage), failures inside SessionEM, TSEM and the own-space stage (`faults`: 25 stage
  errors, counted and reported identically by both packages).
- `perm` equals the unpermuted session, as it should: both packages map the montage by name.

## 3. Official replay (`lean/evidence/replay_original.json`, `replay_lean.json`)

`scripts/replay_package.py PKG OUT cpu`: the warm-up phase's ingestion and scoring programs of the pinned official
code (`a66bc23d`) on the package ZIP, extracted read-only.

| | Original | Lean |
|---|---|---|
| `balanced_accuracy_cell_mean` | 0.9932767697132616 | 0.9932767697132616 |
| `balanced_accuracy` / `accuracy` | 0.9933862433862434 / 0.9933774834437086 | the same |
| Ingestion wall time | 89.4 s | 86.7 s |
| Package files unchanged by the run | yes | yes |
| `[own-stage]` lines | 4 sessions: 189 / 189 / 188 / 189 windows, 179 / 179 / 178 / 179 staged, 0 errors | identical |

The score is in-sample (the package is trained on these people) and equals the replays of 12 Nov. The two
single replays are not a timing comparison; §5 is.

## 4. Determinism (`lean/evidence/determinism_original.json`, `determinism_lean.json`)

- Original: the recording made under line coverage and a later plain recording are bitwise equal (23 common
  cases, 4,559 windows).
- Lean: a recording of an earlier bundle (before a formatting-only change of the sources) and the recording of
  the final bundle are bitwise equal (the same 23 cases).

## 5. Run time, paired and warm (`lean/evidence/timing.json`)

`lean/tools/timing.py`: both packages loaded in one process; one untimed session each; then six public sessions,
two rounds, each session streamed by both packages back to back with alternating order; the time includes the
per-session copy and reset.

| | Original | Lean |
|---|---|---|
| Total for 2,272 windows | 244.64 s | 244.82 s |
| Per window | 107.67 ms | 107.76 ms |
| Lean / original | | 1.0008 (sum); per session: median 0.998, range 0.994–1.018 |

The lean code is not faster and was not meant to be: it does the same arithmetic. Python overhead is a small part
of the 108 ms; the eigendecompositions, filters and nets dominate.

## 6. What the comparison does not cover

- **GPU and the worker image.** Everything here ran on the Mac CPU. On CUDA the nets run on the GPU; an earlier
  package of this line was seen to differ from itself by up to 1.5e-8 between a cold and a warm first stream
  there (`docs/SEALED_UPLOADS.md`), so on the GPU the expectation is equal labels and posteriors within that
  spread, not bitwise equality. A replay in the worker image is owed before the lean package is uploaded.
- **Batches of more than one window.** The evaluator never sends them; the lean code would process them window
  by window, the original ran the nets on the batch.
- **Lean statements that never ran** in the coverage pass (`lean/evidence/coverage_lean.txt`, 13 of 612): the
  three loud refusals (weights that do not match the code, an unsupported net export, a wrong SessionBN trace),
  SessionEM's non-finite fallback, the class-direction gate's two "no decision" returns, the identity gate's
  non-finite return, the fallbacks for missing channel names and a missing `threadpoolctl`, and the `except` in
  `__del__`. The refusals were tested separately: changed flags (`ts_car` 0, `tsem_lam` 0, no `own_beta`), a
  SincShallowNet or FBLightConvNet export given weight, and another trace under a fingerprinted name are all
  refused at load (the last with an `AttributeError` from the missing filter bank instead of the intended message).
  The other lines are copies of the original's and are argued by reading only.

## 7. Line counts

| | Lines | Code lines | Statements | Docstring lines | Bytes |
|---|---:|---:|---:|---:|---:|
| Original `submissions/ensemble_v5b/submission.py` | 1,081 | 811 | 759 | 119 | 50,643 |
| Lean bundle `submissions/lean_v5/submission.py` | 915 | 634 | 606 | 90 | 43,282 |
| Lean sources `lean/src/*.py` (10 modules, 49–185 lines each; with their import lines) | 922 | 666 | 638 | 98 | 42,007 |
| Tools `lean/tools/*.py` (bundle, build, record, parity, timing) | 450 | 332 | — | — | — |

Code lines exclude blank lines, comments and docstrings. The reduction is 22 % of the code lines and 20 % of the
statements. It is no larger because 91 % of the original's statements are live for this package
(`lean/evidence/coverage_original.txt`); the gain is in structure (one path per function, ten modules by
responsibility, configuration checked at load) more than in size.
