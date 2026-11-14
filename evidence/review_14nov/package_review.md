# Package review, 14 Nov: v6.2g → v6.2h (candidate `cand_v62_r2hL`), with notes on the v6.2i working tree

Read-only review. Nothing was run on data and no net was loaded; only scalar keys and array headers of `weights.npz`
were read. Isolated checks and their logs are in `evidence/review_14nov/checks/` (`check_guards.py` + `.log` per tree,
`bench_parts.py` + `.log`, `read_keys.py`, `cmp_G.py`, `zip_vs_dir.py`).

| Reviewed | Identity |
|---|---|
| v6.2g | `lean/src62g` → `submissions/lean_v62g/submission.py` `37da0689…` (bundle current) |
| **v6.2h, the candidate** | `lean/src62h` → `submissions/lean_v62h/submission.py` `ccf16a94…` (bundle current). `runs/sealed/cand_v62_r2hL`: `v6_routed` 2, `v6_ft_labels` 0, cap 400, budget 2700 s, t0 20, every 10, 5 steps, lr 1e-3. ZIP `v62_r2hL.zip` `e22f67e7…`: its 39 members equal the build's `.sha256` list and its `submission.py` is the current bundle |
| v6.2i, in flux | `lean/src62i` **working tree, uncommitted edits at review time** (`finetune.py` `ab6122eb…`, `v6chain.py` `ccaceff8…`, `solver.py` `863289d8…`; bundle current against that tree; no package built yet) |

Confidence labels: **read** = from reading the code; **reproduced** = shown in an isolated run.

---

## Part 1. Correctness and robustness, most dangerous first

### 1.0 Verdict on the v6.2h fix

Finding 1 of the draft (no error protection in `step62`) is fixed for everything that runs per window.
- **Inside the guard** (`lean/src62h/v6chain.py:149-169`): `ft.relabel`, `eye.step`, `rest.step`, `tail6r.step`,
  `ft.step` with every refit, both `mix6` calls, `routeB.stepx`, and both non-finite checks. The AST check confirms one
  `try` in `step62` with both `raise` statements inside it, in the tree and in the bundle (reproduced; v6.2g: none).
- **The fallback is the baseline output.** `out_v5` is computed at line 140, before any addition. Nothing in the
  additions writes to the baseline chain: the fine-tuned nets are separate modules loaded from bytes, `RouteB` owns its
  tail, own-space stage and gate stubs, and `torch.jit.optimized_execution` restores its flag in a `finally` (read).
  The team's forced-exception run shows windows after the error bitwise equal to the v5.1b package (27 of 27, CPU).
- **The fallback is cheap.** A dead session skips eye, rest, fine-tuning and `RouteB` (line 147) and costs v5.1b's
  time per window.
- **Before the `try`:** only v5.1b's own chain plus list and counter updates. No new risk.

What the fix leaves open is items 1 and 2.

### 1. Session start is outside the guard (v6.2h) — crash path, low probability
- **Where:** `lean/src62h/finetune.py:59-62` (`__deepcopy__` calls `reset_state`) and `:64-70` (`reset_state`: four
  `torch.jit.load(..., map_location=dev)`, parameter clones, a device `torch.Generator`). The evaluator calls
  `copy.deepcopy(model)` and then `reset_state()`, so this runs twice per session, 212 times in a sealed run.
- **Scenario:** any exception there escapes. Inside the deep copy it becomes the evaluator's `TypeError`; in
  `reset_state` it propagates directly. The ingestion sets `BENCHOPT_DEBUG=true`, so the run ends. The plausible
  trigger is a CUDA allocation failure at a session boundary: the 800-window test logged the allocator at 67 of 68 GB
  with out-of-memory-and-retry in normal operation. The allocations here are small and the retry path normally
  succeeds, so the probability is low, but this is the only new code that runs unguarded in every session.
- **Confidence:** read.
- **Smallest fix:** load lazily inside `step` (first refit), which is inside the guard. Outputs stay bitwise identical.
  The v6.2i working tree already does this (`finetune.py:58-73`, `load()`).

### 2. The handler can raise — crash path, low probability
- **v6.2h, `v6chain.py:167`:** the `print(..., file=sys.stderr, flush=True)` sits in the `except` block. A failing
  stderr write (closed pipe, full log volume) propagates out of `predict`. v5.1b has the same exposure in
  `Route.report`, so the kind is not new; the place is.
- **v6.2i working tree, `v6chain.py:167-168`:** `torch.cuda.empty_cache()` was added inside the handler, before the
  print. Two problems:
  - if the caught error was a CUDA runtime error, this call can raise again, now outside any protection;
  - it does not free what it is meant to free. The exception's traceback still holds the failed refit's frames (the
    prefix batch, the graph, the optimiser) until the `except` block ends, and the dead session keeps its fine-tuned
    nets and input buffers. `FineTune.reset_state` at the next session start already empties the cache.
- **Confidence:** read.
- **Smallest fix:** wrap the handler's body after `self.dead = True` in its own `try/except Exception: pass`. In
  v6.2i, drop the handler's `empty_cache`, or drop the session's fine-tuning state first (`self.ft.nets = None`,
  `self.ft.X = {}`) and call it after the `except` block.

### 3. v6.2i: the per-session limit does not bound the run, and the first session trips it — timeout and silent loss
- **Where:** `lean/src62i/finetune.py:141-146` (`v6_ft_session_s`, replacing v6.2g/h's process budget).
- **Scenario A, timeout protection.** The limit caps refit seconds per session, so the run-level bound is
  `sessions × (S + one refit)`. With 106 sessions, v5.1b at 21.45 min, non-refit cost at about 1.4 × v5.1b (assumed:
  1.30 measured without fine-tuning, plus the four per-window predictions) and 1.1 min warm-up, the largest S that
  still finishes in 60 min when every session reaches it:

  | Worker slowdown | Largest safe S |
  |---|---|
  | 1.0× | about 15 s |
  | 1.25× | about 11 s |
  | 1.5× | about 6 s |

  Refit seconds per session at A100 speed, from the published timings (my fit, about 0.3 s per refit plus 2.2 ms per
  prefix window): 150 windows about 6 s, 190 about 9 s, 378 about 27 s. So a limit that is inert at 378 windows
  (S ≥ 27 s) allows up to 50 min of refits and protects nothing; a limit that protects at 1.25× (S ≈ 10 s) stops
  refits in every session longer than about 200 windows at normal speed. v6.2h's process budget held to 1.49×
  whatever the session lengths. No v6.2i package exists yet, so S is not chosen; choose it from this table and state
  which side was taken.
- **Scenario B, first session.** The one-off warm-up of fine-tuning in a fresh process (about 65 s with fine-tuning
  against 10 s without, `docs/V6_BUILD.md`) falls mostly inside the first refits. `ft_sec` counts it, so with any
  S below about 50 s the first sealed session stops refitting after its first one or two. Every single-session check
  in a fresh process would show the same.
- **Scenario C, timings do not carry over.** `lean/src62i/solver.py:56` sets `cudnn.deterministic = True` for the
  whole process, which also changes algorithm choice for v5.1b's own convolutions. All A100 ratios (×1.85 / ×2.13) were
  measured without it.
- **Confidence:** read (A is arithmetic on published numbers; B is inferred, not measured).
- **Smallest fix:** B: count `ft_sec` from the session's second refit (`if self.refits: self.ft_sec += dt`); the
  first refit has the smallest prefix. C: one warm, paired A100 timing of the i bundle before any estimate is quoted.
  A: pick S from the table.

### 4. v6.2h: the budget clock is silently off when `load_model` was not the entry point
- **Where:** `lean/src62h/finetune.py:130` (`FT_CLOCK["t0"] is not None and …`).
- **Scenario:** with `t0` None the guard never trips and prints nothing. Reproduced: budget 1e-9 s, clock never
  started, 4 refits ran, 0 frozen windows. In the official harness `Solver.load_model` always runs in the predicting
  process (`base_solver.py:75`), so the sealed run is not affected; any tool that builds the model another way is.
  This already voided one wave of checks (`docs/V6_BUILD.md`, the `--fresh-clock` row).
- **Confidence:** reproduced.
- **Smallest fix:** treat None as "start now", or raise. (Moot for v6.2i.)

### 5. GPU memory: the allocator fills the device; long prefixes were never run on a 40 GB card
- **Where:** `finetune.py` `_refit` (prefix batches of 20 to 400 windows, four nets).
- **Scenario:** live peak 7.2 GB at 400 windows, but reserved memory reached 67 of 68 GB with retry warnings on
  Leonardo. The worker-image GPU replay on Galvani used sessions of at most 199 windows. A 300- to 400-window session
  on a 40 GB device therefore runs the retry path earlier and more often; its time cost is unmeasured. With the
  guard, a real out-of-memory error now costs that session's additions, not the run.
- **Confidence:** read.
- **Smallest fix:** one A100 run of the 378- and 800-window timing with
  `torch.cuda.set_per_process_memory_fraction(40/68)`. v6.2i's `empty_cache` in `reset_state` limits the build-up to
  one session.

### 6. The stress cases were never run on any v6 package
- **Where:** `lean/tools/record.py` cases `stress`, `extreme`, `faults`, `long` (577 windows, past the cap), `switch`,
  `lapse`. `docs/V6_BUILD.md` and the ledger record them only for lean v5.
- **By reading, on the candidate:**
  - zero and NaN windows stay finite: `to_windows` zeroes non-finite samples, StreamBN has a prior and `eps`
    (`experiments/session_bn.py:195-225`), and the eye traces are finite for zero, constant, 3e38 and 1e-30 windows
    (reproduced);
  - one window that overflows float32 inside a net (the `extreme` case) ends the additions for the session: tuned
    member 47 sits on the scaled stream (`x / 24.28`), its posterior goes NaN, and the mix check raises. If the window
    arrives before window 20 it sits in the prefix, the first refit produces NaN weights, and the session is dead from
    window 20. After v6.2h this is a fallback to v5.1b, not a crash;
  - late or flipping recognition: with `v6_ft_labels` 0 the label is always v5.1b's output, so `relabel` is a no-op
    and recognition timing does not touch fine-tuning. `RouteB` follows v5.1b's gate decisions window by window.
- **Confidence:** read.
- **Smallest fix:** run those six cases on `cand_v62_r2hL` on a CPU node and assert: no exception, and the `[v6]`
  error count per case as predicted (0 except `extreme`).

### 7. The guard can hide a permanent failure
- **Scenario:** an error that recurs in every session (a worker-specific failure in a refit, as on 14 Nov with the
  unusable driver) gives v5.1b's output for all sessions from the failing window on, at v5.1b's cost. The only trace
  is stderr: one `[v6] error at window …` line and `errors 1` in each session's `[v6]` line. There is no process-level
  count.
- **Confidence:** read.
- **Smallest fix:** none in code. Read the sealed log for `errors` and `swapped` before interpreting the score.

### 8. Keys: presence is checked, values are not
- **Where:** `lean/src62h/solver.py:29-34`, `v6chain.py:82`, `lean/tools/build6.py:61`.
- **Scenarios:**
  - `v6_ft_cap` 0 or `v6_ft_budget_s` 0 switch the guards off without a message;
  - `v6_routed` outside {0, 1, 2} runs as 1;
  - `v6_ft_every` 0 raises `ZeroDivisionError` at window 20 of every session (reproduced), which the guard now turns
    into a silent v5.1b;
  - a cap below t0 would predict from nets that never saw a refit;
  - `build6.py` defaults `--budget` to 3000 while the documents and all four h packages say 2700 (verified in the
    packages); a rebuild without the flag changes the guard;
  - `v6chain.py:82` keeps a default (`… if "v6_routed" in z else 0`) that `check_package` makes unreachable.
- **Confidence:** read; the `every` case reproduced.
- **Smallest fix:** range asserts in `check_package` (routed in {0, 1, 2}, every > 0, cap ≥ t0, the time guard > 0),
  and no default in `build6.py` for the time guard.

### 9. Log and dead code (no effect on outputs)
- Two `[own-stage]` lines per session (`route` and `routeB`), indistinguishable in a sealed log.
- `self.tail6` is built and reset every session but only `step61` uses it; `self.active` is unused in `step62`;
  `FineTune.step`'s `members` argument is unused; the bundle header says "from lean/src" for every tree.

### Not new: inputs with another shape
A different number of samples or channels fails in v5.1b's own code first (`IdentityGate.feat`, the nets). The eye
net adds one more failure below 25 samples (`savgol_filter`, reproduced), which the guard catches.

---

## Part 2. Efficiency, by estimated saving

Scale: v5.1b is 82 ms per window on the A100 and 1,287 s sealed, so 1 ms per window is about 16 s sealed. The
candidate measures ×1.85 (190-window sessions) and ×2.13 (378), about 41 and 47 min. Fine-tuning is 12 to 18 min of
that. GPU figures below are estimates from the published timings; the CPU figures were measured in isolation on the
Mac (one thread, busy machine, synthetic data). `panel_performance.md` in this folder reaches items 1, 2 and 4
independently.

| # | Waste | Where | Saving (sealed) | Outputs |
|---|---|---|---|---|
| 1 | Refit recomputes the frozen FIR bank on the clean prefix | `finetune.py` `_refit` | about 1.2 to 2.7 min | not bitwise on GPU |
| 2 | Own-space stage runs twice per window | `routing.py:70-71` via `RouteB` | about 1.3 to 2.3 min | bitwise identical |
| 3 | Main thread waits for the tangent thread, then does GPU work | `v6chain.py` `step62` | up to about 1.5 min | bitwise identical |
| 4 | Four fine-tuning nets loaded twice per session, also when unused | `src62h/finetune.py:59-70` | about 10 to 30 s | bitwise identical |
| 5 | Small per-refit and per-window overheads | `finetune.py` | about 0.3 to 0.8 min | bitwise identical |
| 6 | v5.1b's routed tail repeats the shadow tail until the first routed window | `routing.py:65` | about 15 to 25 s | bitwise identical |
| 7 | Per-session deep copy | evaluator | under 15 s | bitwise identical |

1. **FIR bank in the refits.** Each refit runs, per net, five clean forwards and one rebuild pass over the same
   prefix, and the next refit repeats them on a prefix that grew by ten windows. The FIR bank is frozen, and the
   bank output of every window already exists per input group in `NetBank` (`nets.py:129-131`, `cache["sbn_bank"]`).
   Keeping those outputs on the GPU (about 0.7 MB per window and group, 570 MB at the cap) lets the clean and rebuild
   passes start after the bank, as `SharedFIRSJit.proba` does; only the augmented pass needs a new bank. Three of
   the four tuned nets qualify. If the bank is 60 to 80 % of a forward pass, this removes 10 to 15 % of refit time.
   The bank share is not measured: one `torch.profiler` run of a refit at 200 windows settles it, and also says where
   the fixed 0.3 s per refit goes. Results change at float32 level (batch shape), inside the package's own GPU
   run-to-run spread; the reference checks need a rerun.
2. **Own-space stage twice.** `route.own` and `routeB.own` see the same windows from the same reset, so their state
   is identical, and `RouteB` gets v5.1b's own (person, ramp). Measured: `observe` 4.3 to 4.9 ms (every window),
   `log_members` 2.5 to 2.8 ms (routed windows; 18 of the 36 eigendecompositions of a routed window are duplicates).
   A wrapper in `v6chain.py` that reads `route.own.cur` and reuses the window's log members removes both without
   touching `personal.py`. A tail costs only 0.7 to 1.1 ms, so of the "baseline beside a second routed chain" the
   own-space stage is the expensive duplicate, not the tails.
3. **Thread balance.** By the 11 Nov profile the tangent thread (about 51 ms) finishes after the nets (about 45 ms),
   and everything else then runs serially on the main thread. `eye.step` and `rest.step` depend only on the window and
   could run before `cores.result()`; the refit and the fine-tuned predictions need only earlier labels and could too.
   The saving is at most the main thread's idle time, which is not measured (log
   `t(cores.result()) − t(nets done)` once). Moving them means a second guard around them.
4. **Loading twice.** Fixed in the v6.2i working tree by `load()`; that also closes Part 1 item 1.
5. **Small overheads, all bitwise-safe:** the prefix is stacked and uploaded once per net although two streams serve
   four nets (`finetune.py:94`); `dtheta` costs one GPU sync per parameter tensor per net per refit and is only logged
   (`:114`); `zero()` and `params()` walk `named_buffers()` about ten times per step; the per-window predictions run
   three more FIR banks at batch 1 and four separate `.cpu()` syncs (`:150`).
6. **Routed tail before recognition.** Exact duplicate of the shadow tail, but removing it edits v5.1b's chain and
   needs the parity suite. Not worth it at 1 ms.
7. **Deep copy.** About 0.1 s per session (measured for v5.2); v6 adds three rest nets and `RouteB`'s arrays.

**Eigendecompositions.** The tangent cores' 216 per window are all needed and I found no duplicate among them. The
rest member's input stream has its own reference matrix (`v6_rest_G` differs from the nets' `G`, checked), so its one
decomposition is not a duplicate either. On "GPU eigh is the safe speed-up" (memory note of 11 Nov): from my memory of
torch's cuSOLVER backend, the batched Jacobi path is used only up to 32×32; at 41×41 it loops over matrices and is
likely slower than LAPACK on the CPU. Test with a one-minute A100 micro-benchmark before investing.

**Larger, not output-preserving:** fewer or lighter refits (every 20 windows after window 100, a prefix cap of 200).
These need dev scoring.

---

## Checked and found sound

- **Bundles are generated:** `bundle.py --check` passes for v62g, v62h and the v62i working tree; the four h package
  directories and the staged `v62_r2hL.zip` carry the current bundle.
- **Budget clock (v6.2h) in the official harness:** started in `load_model`, which `set_objective` calls in the
  predicting process; monotonic; untouched by the per-session deep copy. The 60-minute arithmetic holds whatever the
  time before `load_model`, because v5.1b's 1,287 s contains the same lead time.
- **After the budget trips:** `ft.step` returns at once; the remaining cost is the measured no-fine-tuning path.
  Label and weight lists stay in step (reproduced: 60 frozen windows, then `relabel` passes).
- **Cap and schedule:** refits at 20, 30, …, 400; the largest prefix is exactly 400; none after (reproduced).
- **Sessions of 1 window or fewer than 20:** no refit, no fine-tuned prediction; every other addition is defined
  from window 0.
- **State between sessions:** eye nets are stateless and shared; rest nets and fine-tuned nets are per session; the
  augmentation generators are reseeded per session; no global RNG is used; the session copy is dropped before the
  next is made, and no reference cycle keeps GPU tensors alive.
- **v6.2i's lazy loading and the buffer change:** `X` is kept for windows 0 to 399, which is what the refit at 400
  reads; predictions use the window's input directly.
