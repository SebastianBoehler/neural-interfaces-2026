# Panel answer: performance engineering (14 Nov 2026)

Read: `lean/src62g` and the `src62h` diff, `lean/src`, `docs/V6_BUILD.md`, `lean/PARITY.md`, `OFFICIAL_CONTRACT.md`,
`2026-competition/benchmark_utils/streaming.py`, ledger §106–§112. Nothing was run from the package and no weights were
loaded. Two CPU micro-benchmarks on synthetic data are in `evidence/review_14nov/checks/` (`perf_microbench.py`,
`perf_chunk_bitwise.py`); the Mac was busy, so those numbers are rough. Every "ms" below is a Leonardo-A100 millisecond
per window unless marked "Mac". Conversion: v5.1b is 82 ms per window there and 1,287 s sealed, so **1 ms per window ≈
0.26 sealed minutes**.

## 1. Diagnosis from my field

- The run time is not GPU compute. It is single-thread CPU LAPACK on the tangent path, about 54 TorchScript forwards
  at batch 1 that cost launch overhead, and interpreter/autograd overhead in the refits. The A100 is close to idle.
- Compute is not what limits accuracy. By the team's own sensitivities (member +1 gives +0.1 to +0.2; extra members at
  fixed weight hurt) most of what spare time can buy is worth little. The exceptions are things that improve the labels
  or views the in-session learning uses.
- What the system cannot do today is in-session learning at scale: fine-tuning is confined to 4 of 30 nets and a short
  schedule only because of its implementation cost.

### Budget, v6.2 on an unrecognised 190-window session (158 ms per window = 1.92 × v5.1b, measured)

| Part | ms per window | Source |
|---|---|---|
| Tangent thread (the critical path): 153 real + 63 complex 41×41 eigh, 24 band filters, covariances, heads | 55–58 | eig 40 and filters 11 measured (11 Nov profile); rest estimated |
| 44 net forwards on the main thread, hidden behind the tangent thread | 45 | measured; about 1 ms per net |
| Shadow tail | about 3 | estimated (Mac: SessionEM + TSEM adapt under 1 ms even at 380 windows) |
| `Route.step` | 19 | measured (§112); mostly own-space filters, gate medians, identity solves, not the EM |
| v6 additions without fine-tuning (eye, rest, third tail, RouteB) | +25 | measured ratio 1.26–1.32 |
| Fine-tuning | +54 at 190 windows, +76 at 378 | measured ratios 1.81 / 2.08 against 1.15 |

Fine-tuning in detail, from the two measured lengths: a refit costs about 0.55 s at a mean prefix of 100 windows and
0.75 s at 195. A two-point fit gives **about 0.35 s fixed per refit plus 2.1 ms per prefix window** (node noise ±5 %,
so the intercept is ±0.1 s). The fixed part is about 15 ms per net-step.

Floor with good engineering: tangent 14–18 (4–6 threads), nets 8–12, tails and routing 10–12, fine-tuning 6–10. That
is 40–50 ms per window, or 10–13 sealed minutes with fine-tuning on every session, against 41 expected and 49 worst
case now.

### The five largest savings

| # | Saving | Sealed min saved (all sessions unrecognised) | Outputs | Risk |
|---|---|---|---|---|
| 1 | Fine-tuning refit path | 14–20 → 2–4 | within the package's own GPU run-to-run spread | low to medium |
| 2 | Duplicate stage work in v6.2 | 3–4 | bitwise | low |
| 3 | core1 and core2 on two threads | about 2.5 | bitwise (chunking unchanged) | low; needs 3 cores |
| 4 | Band-chunked tangent threads plus family-batched nets | about 8 more, only if both land | numerically close | highest; touches the strongest member |
| 5 | Per-session overheads | 0.5–1 | bitwise | low |

**1. Fine-tuning refit path** (`finetune.py::_refit`).
- Keep one preallocated GPU buffer per distinct input stream and write each window once. Today every refit runs
  `np.stack` on the whole prefix and uploads it again, per slot.
- Three of the four slots read the same whitened stream and their generators carry the same seed, so their augmented
  batches are identical. Compute `strong(X)` once per stream.
- The FIR bank is frozen. Cache its output for the clean prefix (append per window) and feed the clean forward and the
  StreamBN rebuild from the cache, split as `SharedFIRSJit.proba` already does. Two slots share the fingerprinted bank.
- `zero()` and `params()` walk `named_buffers()` of a ScriptModule on every step. Collect those tensors once per session.
- `dtheta` forces about 80 GPU syncs per refit for a log line nobody can read on the worker. Compute it once per session.
- Bitwise is not the right bar here: GPU fine-tuning already differs from itself by 0.011–0.034 run to run. Use that
  spread, plus exact agreement on CPU for the buffer and bookkeeping changes.

**2. Duplicate stage work** (`v6chain.py::step62`, `routing.py`).
- Four tails (shadow, routed, `tail6r`, RouteB's) each recompute `SessionEM.features(x)` and the same standardised
  128-PC matrix. Compute once per window and share.
- `RouteB` inherits its own `OwnStage`, so `observe(x)` (band filters on the whole window) runs twice with identical
  input and state. Share one instance.
- On sessions never recognised, `observe` is pure overhead. It can store windows and catch up at first recognition.

**3. Two tangent threads** (`ensemble.py::_tangents`). Submit core2 as a second future. The critical path becomes
max(core1 about 37, core2 about 20, nets 45) instead of 55–58. Three lines.

**4. Chunked tangent threads plus batched nets.** Split the bands over 4–6 threads (LAPACK and `sosfilt` release the
GIL); Mac: one eigh pass for both cores 8.6 ms → 1.9 ms on 6 threads. Without batched nets this stops at 45 ms, so the
two belong together. In my check a chunked `torch.linalg.eigh` was **not** bitwise equal to the unchunked call
(deterministic at fixed chunking), so this is a 1e-13-level change to core1. I would not do it this week.

**5. Per-session overheads.** `FineTune.reset_state` runs in `__deepcopy__` and again when the evaluator calls
`reset_state`: eight `torch.jit.load` calls per session, four wasted, and the template model keeps a set forever. Load
lazily at the first refit. The deep copy also duplicates two 44k × 128 float64 view matrices (90 MB) per session.

Not worth doing: mixed precision, channels-last, `cudnn.benchmark`, freezing, `torch.compile`. Nothing is compute-bound.

## 2. Five proposals, ranked

**P1. Guard rails and the bitwise savings (2, 3, 5 and the bitwise parts of 1) in the next package.**
- Mechanism: same outputs, less time; a budget that cannot be fooled.
- Gain: none in accuracy. It keeps fine-tuning on the last sessions (the budget trips in v6.2's expected case) and
  removes the tail risk of losing one of two uploads.
- Experiment: `lean/tools/record.py` + `parity.py` on CPU, then one paired A100 timing. Under 1 GPU hour.
- Kill: any parity difference on CPU, or less than 15 % off the paired time.

**P2. Make fine-tuning cheap and linear, then fine-tune all seeds.**
- Mechanism: saving 1 in full. Then stack the six seeds of each SessionBN FBCNet family in a plain-tensor forward: the
  `FBC` class in `nets.py` is already a functional FBCNet, so add StreamBN and a seed axis to its einsum. Autograd runs
  through it directly, and 18 nets fine-tune in three backward passes. FBLight and SincShallow stay as they are.
- Why: the refit is overhead-bound, so six stacked seeds cost about what one costs.
- Gain, team's dev numbers: all6 +0.36 to +0.46 against delta1 +0.26; cum10 +0.16 over gC5. Together +0.2 to +0.35 dev
  on new people, about +0.1 to +0.15 sealed. Returning people about 0. Confidence: medium on cost, medium-low on gain.
- Experiment: stacked forward against each trace (≤ 1e-5), then time one 378-window session. 1–2 GPU hours.
- Kill: stacked refit of 18 nets costs more than 2× the engineered delta1 refit.

**P3. Relabel the prefix at every refit, and even out how often each window is used.**
- Mechanism: labels are the causal outputs from when each window arrived, so the earliest windows carry the worst
  labels. With refits at 20, 30, … on the whole prefix, window 0 enters about 85 steps in a 190-window session and
  window 175 enters 5. At each refit, apply the current templates to all earlier windows (one batched LDA pass, about
  1 ms) and sample a fixed batch with equal expected visits.
- Why: true-label fine-tuning is measured at +1.9 over chain labels; this moves towards it at no cost.
- Gain: +0.1 to +0.5 dev on new people; confidence low to medium. I did not check whether §145's label arms cover it.
- Experiment: the §178 rerun with relabelled labels. Under 4 GPU hours.
- Kill: under +0.15 over v6.1's labels on the within orders, or negative on full.

**P4. Bagged stages.** K perturbed or bootstrapped copies of the tail (K = 16 costs about 30 ms per window before
vectorising), outputs averaged. The stages carry the score and fit 51- and 128-dimensional templates on tens of
windows. Gain +0 to +0.3, low confidence. Experiment: CPU only, in `replay/` on stored member outputs. Kill: under
+0.15 averaged over the three shuffle schemes.

**P5. Deterministic deferred refit.** Start the refit at window t on a second thread and join it at t + 5. Latency
becomes smooth and the wall time overlaps with the CPU-bound tangent path. Caveat: `torch.jit.optimized_execution(False)`
is a global switch and would de-optimise the main thread's nets, so this needs P2's plain-tensor forward first. Zero
accuracy gain; only worth it if P2 leaves fine-tuning above 10 ms per window.

What I would not buy with spare time: more members (measured negative), test-time augmentation of the nets (at most
+0.1, and StreamBN treats the batch axis as time, so it needs a stateless path), two whole systems (no label-free
selection rule exists). A large pretrained member fits easily: batch-1 latency scales with depth, not width, so my
estimate is 5–8 ms per call for an 86 M-parameter encoder in fp16 and 12–18 ms for 300 M, which is 2–7 sealed minutes.
The limits are ZIP size and its measured value (none so far), not time.

## 3. Contrarian claims

- **"Fine-tuning is quadratic, so only delta1 fits and all6 at 142 s per session is infeasible."** The team's own two
  timings say 46–63 % of a refit is fixed overhead that does not depend on the prefix. That is implementation, not
  arithmetic. Test: time `_refit` against t on one A100 and read the intercept; watch GPU utilisation.
- **"GPU eigh is the safe speed-up" (11 Nov note).** For 41×41 float64 batches I expect CUDA `eigh` to be slower than
  CPU LAPACK; small-matrix GPU eigensolvers usually are. I could not test it on torch 2.14. A five-line benchmark on
  Leonardo decides it before anyone builds on it.

## 4. Question for the organisers, and reliability findings

**Question:** how many CPU cores and how much RAM does the worker container get, is the GPU exclusive, and does the
60 minutes include container start and data loading? Savings 3 and 4 depend on the core count. Logs are hidden, so the
package should also calibrate itself: read `len(os.sched_getaffinity(0))` and time a fixed kernel in `load_model`.

**Reliability (v6.2h tree):**
- **Budget clock.** `FT_CLOCK` starts at `load_model`; setup before it is unseen. Use process age (`/proc/self/stat`),
  with the current clock as fallback.
- **Guard shape.** The absolute budget removes fine-tuning from whichever sessions come last. Add a pace rule below it:
  the package times its inner v5.1b per window, so allow additions only while total time ≤ about 2.3 × inner time, and
  lengthen `every` (10 → 20 → 40) before stopping.
- **Sticky CUDA errors.** v6.2h catches exceptions in the additions, but a poisoned CUDA context (the
  `cudaErrorContained` seen on Galvani) also kills v5.1b's nets. No cheap fix; know that the guard does not cover it.
- **Out of memory.** The exception is now caught, but nothing calls `torch.cuda.empty_cache()`; add it in the `except`
  and at session reset. Refit batches grow from 20 to 400 windows, so the allocator caches many block sizes.
- **Determinism.** Set `torch.backends.cudnn.deterministic = True` and `benchmark = False` explicitly, then repeat the
  run-twice test. An audit rerun of a fine-tuning package will not reproduce labels exactly today.
- **After the cap.** `self.X` keeps growing past `v6_ft_cap` although no refit will read it.
- **No timeout on `cores.result()`.** A hung worker thread hangs the run. Low probability.
- I found no GPU memory growth across sessions: copies are released before the next deep copy, and I saw no reference
  cycles. Repeated `torch.jit.load` (about 800 calls per run) may leak a little host memory; bounded.

## 5. The two uploads

- **Upload 1:** v6.2h with P1 only, after CPU parity and one paired A100 timing. Same accuracy candidate, expected
  time down by roughly a quarter to a third, budget not tripping. Its observed run time against the prediction is the
  only worker calibration available.
- **Upload 2:** upload 1 plus whichever of P2 and P3 passed dev under all three shuffle schemes, at ≤ 40 minutes on the
  calibrated worker. If neither passes, do not spend it on speed alone.

## Round 2

Withdrawn: the process-level budget and my process-age clock (round 1, section 4). Both cross the session boundary.

### (a) A run-time guard that lives inside one session

- **Primary, deterministic:** bound the work, not the time. Each refit uses a fixed batch (at most 128 windows: the
  newest plus an even sample of the prefix), every 10 windows. Cost per window is then constant, and the total depends
  only on the window count, not on session lengths or order.
- **Backstop, session-local clock** started in `reset_state`: additions run only while
  `elapsed ≤ A × windows_so_far + one refit of credit`. Over it: lengthen `every` (10 → 20 → 40), then freeze for the
  rest of that session. Nothing is kept afterwards.
- **A = 149 ms**: (3,300 s target − 300 s allowed for setup) / 20,083 windows. v5.1b's sealed pace was 64 ms per
  window; v6.2 today is about 123 (190 windows) to 141 (378).
- **It bounds:** total predict time ≤ 149 ms × 20,083 ≈ 49.9 min, plus one refit of credit per session (106 × ≤ 0.8 s
  ≈ 1.4 min), for any mix of session lengths and with the same treatment of every session.
- **It cannot bound:** time outside `predict` (setup, loader, about 0.3 s of copy and load per session); a worker more
  than about 1.8× slower than on 12 Nov, where the frozen path alone exceeds A; a single hung call; a sealed set that
  is not 20,083 windows.
- **Trap:** the first session's one-off 65 s warm-up would freeze that session. Run a synthetic refit in `load_model`.
- **Check:** with the guard inert the package must equal the unguarded v6.2 bitwise on CPU.

### (b) Cost and traps of the others' top proposals

| Proposal | ms per window | Trap |
|---|---|---|
| Run-level count filter (1) | about 0.1 (169-state forward step, log space) | An exact urn zeroes a class after one wrong boundary or lost window: keep the soft floor. Gates and fine-tuning labels must keep reading the unfiltered posterior. Latch the detector; default off |
| Run-offset removal (2) | 0.1–0.2 (recompute offsets over stored views; change-point from core1's band variances, already computed) | The gates and personal heads read the same 256-PC view: subtract only on the copy the template stages get, once, upstream of all four tails |
| Eye regression out of members (4) | about 0.05 | Not a run-time problem. Every member, PC basis, gallery and personal head must be rebuilt, and two versions of the window are in flight (eye net and SessionEM need the raw one) |
| Episodic retraining (7) | 0 | Exports must keep the fingerprinted FIR bank and the StreamBN prior; the family weights sit on a sharp optimum, so match the mix's sharpness or re-tune |
| Transductive objective (8) | 1–2 at 16 dimensions, refit every window | Hard Sinkhorn balance on a short prefix forces errors: relax the marginal by about √t and work in the log domain. Gates must keep v5.1b's shadow |

### (c) The freed 30 minutes

None of these needs them: all cost under 2 ms per window (0.5 sealed minutes). Spend 3–6 minutes on all-seed
fine-tuning if it passes; keep the rest as slack. With the process clock gone, slack is the only protection against
a slower worker.

### The six points

1. **Attack: eye regression out of the members (4).** It fails on integration, not mechanism: no parity anchor
   survives, and it cannot be replayed on the worker image by 20 Nov.
2. **Support: the run-level count filter (1), above my own.** A last stage with a bitwise off-switch, +0.68 with true
   run ids. It works if boundaries come from position plus change points and the items in the table hold.
3. **Conflicts.**
   - A: proposals 1 and 2 need the same run-boundary estimator; build it once. Settle with true run ids in the real
     chain, each alone and together (CPU, 1–2 hours).
   - C: statistics found one SessionBN seed set equals three in the chain. That is about 20 forwards to drop.
   - D: not my field; ship it as a weights switch so it can be removed without a rebuild.
   - E: upload now, after the clock is replaced. Its run time is the only calibration.
4. **Merged ranking** (dev, new people unless stated; tests in this order):
   1. Run-count filter with detector: +0.3 to +0.6 if runs are contiguous.
   2. Run-offset removal: +0.2 to +0.6; same harness.
   3. Person-specific route strength (10): returning +0.3 to +1.0; CPU.
   4. Episodic retraining (7): +0.2 to +0.6; start training today, it is the long pole.
   5. Cheap fine-tuning and all seeds (11): +0.2 to +0.35. I cut my relabelling to 0 to +0.2 after the ML expert's
      reading of §142.
5. **Combination.** 1 + 2 + the scheme detector share one estimator; expect about 1.3× the larger gain, not the sum.
   Do not combine 4 with anything this week: it invalidates the stored outputs the other tests use.
6. **Odds of 0.831:** next upload about 5 %; last upload about 15 % if the filter passes and runs are contiguous,
   otherwise about 8 %.
