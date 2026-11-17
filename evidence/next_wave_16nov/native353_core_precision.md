# §353 core precision diagnosis and qualified CAR repair

The new native port's CAR float32 reduction order caused its component guard
failure. NumPy sums the fixed channel axis sequentially; Torch mean used a
different reduction order. The first source window's mean differs by1.144409e−5.
No equation, dtype, weight, stopping rule or declared threshold changed.

## Complete original source comparisons

| Output | Before | After CAR repair | Unchanged bound | Changed labels |
|---|---:|---:|---:|---:|
| real core p1 |2.64226994319e−7|2.34737229654e−10|1e−7|0|
| Hilbert core p2 |7.51212217006e−8|4.76986029754e−9|1e−7|0|
| standardized real-core features |1.44679133658e−4|7.73144308797e−7|1e−4|n/a|
| tail |4.49995818386e−10|8.78741523991e−13|1e−7|0|
| reapply |3.87502019708e−9|1.70894409735e−11|1e−7|0|

ALL96 actual P01 source windows were compared. Original p1/features failures
remain recorded in `native353_core_precision_diagnosis.json`.
The repaired complete result is `native353_core_precision_repair.json`.

## Cause isolation before implementation

First-window C++ instrumentation reproduces the captured native p1/p2 EXACTLY.
The Python stage reconstruction reproduces the actual frozen Python core.
All CAR/filter/covariance/log/whitening/slow/feature/logit stages were captured.
Keeping every downstream native operation unchanged, replacing ONLYCAR with the
actual NumPy CAR lowers real-core error from2.64227e−7 to2.34737e−10 (>1000×).
Hilbert error falls7.51212e−8→4.76986e−9. Replacing ONLY temporal means does not
help (real-core2.65003e−7); changing both gives3.56160e−10. Thus no temporal-mean,
SOS, FFT or spectral patch is justified by this diagnosis.

The minimal shared `common_average_reference_float32` helper preserves float32
and sequential41-channel summation. Both TangentCore and source-core preprocessing
use it. BEFORE the repaired downstream run, ALL96 CAR arrays matched NumPy
exactly across1,889,280 float32 values. No NumPy code was vendored; the existing
project nc_preprocess already uses this scalar accumulation rule.
The component probe projects native core features into its TSEM inputs, so tail
roundoff also shrinks without any tail code change.

## Artifact and execution proof

Diagnosis3120503: CPU1/16GiB,86s, exit0:0, MaxRSS1,594,252KiB.
Repair3120535: CPU2/16GiB,144s, exit0:0, MaxRSS2,885,432KiB.
Total374allocated CPU-core-seconds=.10389CPUh/0GPU; both jobs complete.
Only math.cpp/core.cpp/core_training_window.cpp and a small CAR test TU compiled.
Sixteen unchanged object/static-library entries, including the ORIGINAL probe
object, were pinned and reused. Exact original CMake linker commands were used
with the new artifact directory. No full NN rebuild occurred.
`native353_core_precision_build.json` records sources, objects and commands.
Worker-local dependencies were fingerprinted AFTER placement and verified at end;
shared application/model/Torch NFS identities were pinned before execution.

Validated base:
`/mnt/lustre/work/gehler/gfh527/ni26_core_car_repair_v1_20261117/build/libni26_base.so`
SHA256 `70d149780e714f2763beb631804a14b5b77ed7fa2c8d974393b9f36bffe41cb5`.
Output SHA256 `a9cbba17bfb5a4493b53ed49671c2f0d9a120b8ceee158b8f1287f2f0e6dab61`.
Source manifest SHA256 `73166f4bd26ca827d31ea73fe6d8c624a257de5322c07b7175af4f7d3fcaa85f`.

Scope: host-Python source fixture versus this new native component port. No
training fit, held label, accuracy, full ensemble, clean SIF/A100 worker, runtime
or performance claim follows. The existing source/model/library artifacts and
failed outputs were not overwritten. Native source-training parity remains a
separate, unexecuted qualification.
