# Native EEG Stack Implementation Plan

> **For agentic workers:** Execute the user's approved fifth lane with parallel
> backend, reference-validation and cluster workers. Track completed checks in
> `evidence/next_wave_16nov/track5.md`; do not repeat completed checks after compaction.

**Goal:** Run native training and causal inference on Galvani, validate a Python
reference, then compare complete new EEG models and learning rules.

**Architecture:** C++/LibTorch shared library with a C ABI and thin Python ctypes
bridge. Native code owns preprocessing, normalization, models, training,
adaptation and session state. Python selects disjoint folds and pinned delivery
orders, validates artifacts, and calls existing scoring functions.

**Tech Stack:** C++17, installed LibTorch CPU/CUDA libraries, ctypes, existing
NumPy experiment I/O and official-metric helpers.

**Spec:** `evidence/next_wave_16nov/native_stack.md`, authorized by the user;
exact scientific definitions and predictions are in `docs/LEDGER.md` §293.

## Global constraints

- New heavy work runs on Galvani, at most32 running user jobs and300 submitted.
- Preserve the four earlier lanes, pinned packages and official submodule.
- Register before scoring. Never fit the held person's rows or use future windows.
- Every mutable native session state belongs to its handle and resets on copy/reset.
- Source files stay below approximately300 lines. Fail loudly on malformed input.
- No generated binaries, checkpoints, datasets or environments enter Git.
- The Python wrapper does not implement scientific preprocessing/training/adaptation.

## Review focus

- Check double-precision FFT/mean semantics and float32 feature output against NumPy.
- Supply identical masks/weights/minibatches for gradient and AdamW parity.
- Clone interleaved sessions, reset, and prove future-prefix independence.
- Ensure every new encoder and loss changes gradients; reject silently ignored arms.
- Verify source, binary, fit-person, data, order and checkpoint provenance before scores.

## Task1: Native backend and explicit interfaces

Files: `studies/16nov/lane5/native/` and `build.sh`.
Interface: C ABI declared in `native/api.h`; opaque handles own model and session.

- [ ] Pin ABI jointly with the Python bridge before implementing callers.
- [ ] Implement the exact Lane2 FFT/MLP reference plus explicit forward, gradient
  and one-AdamW-step entry points accepting shared inputs/masks/initial parameters.
- [ ] Implement native fold-only normalization, four registered production arms,
  training, causal step, reset/clone and save/load.
- [ ] Compile using installed Torch libraries in a CPU Slurm allocation.
  Expected: loadable shared library, exported ABI, CPU and CUDA smoke results.

## Task2: Reference and state validation

Files: `studies/16nov/lane5/bridge.py`, parity/reference helpers and tests.
Consumes: `native/api.h`. Produces: non-scientific parity evidence and strict
pass/fail status required before model experiments.

- [ ] Test features, fixed-weight logits/loss, input and parameter gradients and
  AdamW update. Expected tolerances: features2e-5 abs/rel; logits/loss/gradient
  and update2e-5 abs/rel, with maximum errors recorded rather than rounded away.
- [ ] Test checkpoints, independent clones, reset, early identity, current-only
  input, adaptation displacement/active updates and future-prefix invariance.
- [ ] Compare the reference on actual public EEG without scoring held labels.
- [ ] Smoke all four new arms with real disjoint-person training, active extra
  loss and finite normalized predictions. Expected: all guards pass.

## Task3: Registered whole-model experiment workflow

Files: `studies/16nov/lane5/run.py`, scoring helpers, Slurm files;
evidence `evidence/next_wave_16nov/track5.md`.

- [ ] Stage immutable private source with input/source hashes and exclusive outputs.
- [ ] Queue the80-fit matrix only after the reference and real-data smokes pass.
  Expected: four training arms ×ten held people ×two seeds, eight pinned orders,
  static/adaptive outputs, complete native checkpoints and provenance.
- [ ] Queue complete-matrix validation/scoring as a successful dependency.
  Expected: existing official equal-cell metric and people/order intervals,
  matched-seed contrasts, zero comparison, pinned orders and incumbent reference.
- [ ] Record true timings, warm paired latency, transfer/memory limits and worker
  compatibility status. Commit and push source and compact evidence.

Implementation choices above are the agent's engineering decisions within the
approved scope. A faithful port is the scientific control, not the search boundary.
