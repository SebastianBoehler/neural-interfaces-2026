# A100 worker parity build

The tested package uses the regular Python `Solver` adapter and runs the decoder in C++.
The worker build matches frozen's float32 filter reflection, Hilbert transform and complex
centering, Euclidean alignment, CUDA Adam updates, and rest decoder session copies.
The default CMake build includes the portable parts of these fixes. The remaining
arithmetic is tied to the verified Linux worker image and kept in this overlay.

## Use the tested binary

On Galvani, use `package/` in
`/mnt/lustre/work/gehler/gfh527/ni26_native366_exact_release_v1_20261117`.
Check all files with `sha256sum -c package.sha256` from that directory. The manifest SHA-256
is `d33cfb35de9944bcb3d3b703091a12c6fe2a6b4eb6f900a6396d1d28079fe925`.
The release files match the measured package byte for byte. It also includes the DUCC license.
Use the same `runs/cloud/ni26_worker.sif` image as jobs 3122783 and 3122794.
Its SHA-256 is `7fc0389c46ac6bec7c4395a9815e9ffee04d207a2a3700aeb3aef318bf79e2d0`.

## Rebuild the worker overlay

Apply `worker.patch` with `git apply --unidiff-zero` to a scratch copy of the source at
`d0acaa8f0e5790573d55cc8a4a4c767994faf9ae`. Place `core_complex_mean.h` beside the patched
`core.cpp`. The patch reproduces the two tested translation units exactly.

The Galvani link recipe is `build_core.py`. It expects a fresh build directory containing
the patched `core.cpp` and `networks_input.cpp`, plus `filter.cpp` and `include/ni26/math.h`
from the pinned revision. It reuses the existing §360 CMake objects and the §366 CUDA Adam
and rest objects at the paths declared in the script. The build and the two replay scripts
are preserved with their source and input manifests in the job directories.

- `hilbert.cc` uses the SciPy v1.18.1 DUCC source tree
  `a3bde2769727b08af3f0abc12e503c31eeb39423`. The source tree, license, and exact compile
  command are in `ni26_native366_ducc_v3_20261117`.
- `ea.cc` reproduces NumPy's covariance order and calls the same native LAPACK and BLAS
  symbols exposed by the worker's NumPy 2.5.2 OpenBLAS library. The exact compile command
  is in `ni26_native366_ea_native_v1_20261117`. The worker reports failure if those symbols
  or the library are missing.

[A100 worker evidence](../../evidence/next_wave_16nov/native366_worker.md)
