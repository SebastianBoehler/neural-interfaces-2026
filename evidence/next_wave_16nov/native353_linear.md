# §353 fully native linear training head — prepared, not yet runtime-qualified

`native/include/ni26/linear.h` exposes `LinearFit fit_multinomial(X,y,C)` and
`StandardScaler::fit/transform`. The optimizer executes vendored unmodified
SciPy1.18.1 C L-BFGS-B directly through reverse communication. It does not import,
call or embed Python, a SciPy extension, or an interpreter callback. The objective
is the previously qualified§343 class-major C++ float64 softmax kernel.

The existing optimized LP64 native OpenBLAS library supplies vector BLAS/LAPACK.
Its SciPy symbol prefix is a native library ABI, not a Python dependency.
`native/cmake/linear.cmake` requires the explicit library path. Root should include
that fragment after defining `ni26_base`; it adds the native optimizer/objective/
scaler and `ni26_linear_source` test. Vendor origin, exact source hashes, BSD
license and existing binary dependency notices are retained under
`native/vendor/scipy_lbfgsb`. No binaries, data or checkpoints are committed.

Preserved contract: three fixed classes, unweighted mean CE+||W||²/(2*C*N),
unpenalized bias, zero initialization, class-fast optimizer vectors, history10,
factr64 (=ftol64eps), pgtol1e−5, maxls50, maxiter2000 and the reference's default
maxfun15000 check after accepting an iterate. `LinearFit` reports native status,
message code, convergence, iterations/evaluations and final loss/gradient norm.
Callers must reject `!converged`; no alternate optimizer or weaker stopping is
selected. No sample-weight option is present because this route is unweighted.

Native scaler uses population variance (ddof0), corrected two-pass accumulation
without a full centered-matrix allocation and sklearn's float64 near-constant
criterion, including exact zero-variance columns. Invalid/nonfinite inputs fail.

## Prospective real-source qualification (before any new native fit)

Inside the coordinator's registered353 build/parity budget, export the immutable
§343 P01 source2465×44239 matrix/y, its original sklearn head, scaler source slice
and label-free original within0/full0 query62-prefix fixtures. No held labels or
accuracy are loaded. `native/tests/linear_fixture.py` only prepares the test
oracle/archive; it is not native training or runtime code.

Run `ni26_linear_source FIXTURE.pt OUTPUT.pt` as a standalone C++ executable.
Require native convergence with exact82iterations/88evaluations; independent
LibTorch-autograd original objective/gradient error≤1e−9; original-space maximum
gradient≤1e−5; objective≤reference+1e−5; source and both label-free query posterior
maxerror≤1e−3 with ZEROall changed labels. Require real96×257 source scaler plus
constant-column mean/scale/transformed output errors≤1e−12. Exercise NaN/±Inf,
invalid labels and C rejection. Record source/library/configuration/thread/row
hashes and component fit time, not a whole-pipeline or language speed claim.
Numeric prior: same82/88 and zero labels, probability error≤1e−6; native fit wall
not slower than the343 hybrid at matched node/thread/quality (unmeasured).
Any failed math/counter/label/scaler guard blocks adoption; no tolerance change.

Current validation: all owned C++ translation units/test pass local syntax-only
checks against installed LibTorch headers; vendor C passes C99 syntax-only;
fixture exporter passes py_compile. No local training or independent cluster
allocation was run. Real-source integrated build/run remains coordinator-owned.
