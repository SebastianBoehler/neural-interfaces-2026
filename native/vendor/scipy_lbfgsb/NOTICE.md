# Native L-BFGS-B dependency

Unmodified SciPy 1.18.1 C L-BFGS-B source, BSD-3-Clause. No Python module,
SciPy extension, Python C API or interpreter callback is compiled or invoked.

Upstream: https://github.com/scipy/scipy/tree/v1.18.1
Tag tree: e4e854eaa8f18d807cd3496028e257e36caa93cc
Files are copied from scipy/optimize/src/{lbfgsb.c,lbfgsb.h,
blaslapack_declarations.h}, scipy/_build_utils/src/scipy_blas_defines.h and
root LICENSE.txt. SHA-256 values are recorded in source.sha256.
The vendored upstream source is exempt from the project's 300-line source limit.

The first qualified build links the existing LP64 optimized native OpenBLAS
library used by the reference installation, with BLAS_SYMBOL_PREFIX=scipy_.
That prefix names native BLAS symbols; it does not call Python or a SciPy module.
The exact library is libscipy_openblas-5f890258.so, SHA-256
2a5b4c1a6311873c59132b2ceaaf3c26258a17350b7cd27ea8782a47e3322981.
CMake requires its explicit path; no different BLAS or integer ABI is selected
as a fallback. Changing this dependency requires new numerical qualification.

OPENBLAS_BINARY_NOTICES.txt is the unmodified notice file distributed with that
installed wheel. It includes OpenBLAS/LAPACK and dependent runtime notices.
If distributing those native libraries, include their applicable notices and
runtime dependencies. No native libraries are committed in this directory.
