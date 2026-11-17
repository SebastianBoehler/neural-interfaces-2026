# Matched CPU logistic solver (§343)

The current implementation is `FullReference` in `reference.py`, backed by
`reference.cpp`. It uses the existing SciPy L-BFGS-B optimizer with exactly the
sklearn reference's zero initialization, C=.3, unweighted three-class mean
cross-entropy, weight-only L2/(2*C*N), unpenalized bias, float64, gtol=1e-5,
ftol=64*eps, maxiter=2000 and maxls=50. It returns ordinary `[3,D+1]` coefficients.
The reference external scaler must already have been applied to x.

The C++ context borrows x and validates it once. Each objective callback uses
two matrix products and fused scalar loops for softmax, cross-entropy, L2 and
finite checks. Logit and gradient buffers persist for that fit only. Parameters
and gradients keep sklearn's feature-major/class-minor optimizer layout.
SciPy applies its ordinary original-coordinate stopping rule. The implementation
has no automatic solver/device/precision fallback or saved training state.

`FullReference(library).fit(x, y)` returns theta, diagnostic statistics and the
actual objective-evaluation count. The optimizer remains SciPy; this is not a
fully C++ optimizer or a language-only speed claim. `build.sh` compiles against
the existing Galvani LibTorch installation, with no new provider or dependency.

The source-span alternative is `ReferenceSpan`. It factors XXᵀ and solves in
an orthonormal source basis, reconstructing the full-width coefficients.
Only roundoff-scale null modes are removed and the original-space gradient
is checked at every accepted iterate. Gram construction, eigendecomposition,
projection and reconstruction all count toward fit cost. This can trade memory
for repeated-math savings; it is not the default full-width route.

The initial `Span`/`span.cpp` experiment retained the old Torch L-BFGS optimizer.
It reached a tighter optimum quickly but FAILED held-output parity, including
changed labels. Its files and the frozen head dependency preserve that failed
instrument; it is not an accepted replacement for the sklearn-fitted head.

`reference_smoke.py` independently checks full objective/gradient vectors on
real mini/full source matrices, NaN/±Inf rejection, exact source/query/scaler
provenance, and frozen label-free held predictions. `reference_benchmark.py`
compares complete contemporary fits on one node, cold and warm ABBA, records
actual sklearn nfev/njev/nit, probes memory in fresh subprocesses and checks a
second source fold. `memory.py` separates loaded RSS, sampled fit RSS and process
peak RSS including loading. `held.py` never reads held labels.

Evidence is in `evidence/next_wave_16nov/solver343*`. Existing producers, active
libraries, packages and the official submodule are unchanged. Engineering
qualification does not establish a development-accuracy/full-system gain,
sealed-worker compatibility or upload approval.

`reference_classmajor.cpp` is a separately registered layout probe. It copies
only the small parameter/gradient buffers and keeps the same optimizer
interface. Its fixed-parameter microbenchmark must win before new fit tests;
it is not an accepted replacement merely because it compiles.
