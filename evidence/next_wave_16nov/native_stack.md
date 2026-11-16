# Native-stack assessment and proposal, 16 November 2026

Status: the user authorized implementation as a fifth Galvani lane on16Nov,
including training and inference, reference validation, then new architectures
and learning mechanisms. The four earlier lanes remain separate. No competition
upload is authorized. Execution is registered in `docs/LEDGER.md` §293 and
tracked in `track5.md`.

## User objective

The user wants control over representations, data layout, activations, losses,
learning algorithms and execution, as well as speed. Coding agents reduce the
cost of translation and implementation. The incumbent architecture is not a
boundary for the proposed stack. Numerical verification remains necessary:
generated source code alone does not establish equivalent EEG processing or
learning behavior.

## Rules evidence

The user supplied a Discord screenshot on 16 Nov. On 20 Oct at10:49, Myrmecoman
described a Python-bound C++ processing pipeline for expensive functional
connectivity, followed by LDA rather than deep learning. The question explicitly
asked whether bundled wheels were allowed and said their submission worked.
At11:50–11:51 Bruno Aristimunha answered yes and required Python importability.
[Message and adjacent replies](https://discord.com/channels/1405198616934481961/1547897829253451877/1562024877135896619).

This resolves the permission question for a native pipeline behind the Python
entry point. It does not prove their method outperforms ours, that it replaces
every dependency, or that our own binary is worker-compatible. Build artifacts
must match the worker's Python, CPU, shared-library and CUDA interfaces.
Session-dependent native buffers must reset and remain independent when the
evaluator copies a model; native static caches cannot carry adaptation across
sessions. The model-facing input and official metric remain fixed.

## Current implementation and measured limits

- `lean/src66/ensemble.py` already overlaps CPU tangent cores with GPU networks.
  Further parallelism should follow actual data dependencies, not launch every
  operation concurrently. Inference receives one window per call; future
  windows cannot be batched before delivery.
- NumPy linear algebra already calls BLAS/LAPACK. PyTorch provides a C++ frontend
  and custom C++/CUDA operators. We can own layouts, scheduling and selected
  operators while retaining numerical libraries where they are useful.
  [NumPy](https://numpy.org/doc/stable/reference/routines.linalg.html),
  [PyTorch C++](https://docs.pytorch.org/tutorials/advanced/cpp_frontend.html),
  [custom operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html).
- Existing `evidence/v66_14nov/rprof-3103933.out` profiles one v6.5b refit on a
  2080Ti, not full v6.6 worker time. At190 prefix windows it measured1284.86ms:
  clean forward325.6, augmented forward321.7, backward/Adam518.6, staging29.4,
  augmentation9.9, final StreamBN64.2 and parameter-delta extraction1.8ms.
  An isolated FIR took9.18ms against17.71ms for one member forward. These
  measurements justify investigating neural/refit execution as well as CPU
  features. They establish no end-to-end native speedup.

## Authorized scope

Prefer C++/CUDA for direct integration with the current GPU stack. Rust is also
credible for CPU signal-processing kernels; choosing it would require a clear
benefit at the GPU interface. This is an engineering recommendation, not a
language restriction imposed by the challenge.

1. Define explicit window, feature, model and per-session-state interfaces.
   Record units, axes, dtype, filter padding, normalization and covariance
   regularization. Expose intermediate tensors for comparison and inspection.
2. Build one complete reference path: preprocessing, features or network,
   prediction, causal adaptation, loss, optimizer and reset. Include training
   for that selected model, rather than only a disconnected fast kernel.
   General-purpose autodiff, optimizers and GPU kernels may use existing native
   libraries; an entirely independent training framework is a larger optional
   scope, not assumed necessary for architectural freedom.
3. Validate fixed weights and inputs against the current implementation, then
   validate a training update and session isolation. Report floating-point
   tolerances, gradient differences and label changes explicitly. Reuse the
   existing scorer and pinned delivery orders.
4. After the reference is faithful, explore genuinely different feature
   layouts, connectivity models, activations, losses and adaptation mechanisms.
   Register scientific changes separately so language-port effects and method
   effects remain distinguishable. Whole new systems are evaluated whole.
5. Measure full warm/paired execution, initialization, memory and transfers on
   Galvani and test the binary in the exact worker image. Benchmark190/378/800
   delivered windows. Report developer control and full-system accuracy as
   outcomes alongside runtime; speed alone is not the objective.

Success check: a worker-loadable, session-isolated end-to-end native reference
with inspectable data semantics and verified numerical behavior, followed by
independently measured method variants. Galvani build and numerical validation
do not alone establish exact competition-worker compatibility.
