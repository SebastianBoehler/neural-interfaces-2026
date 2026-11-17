# §343 CPU solver repair — qualified v4

**Prepared and qualified:** `FullReference` in `studies/17nov/solver343/reference.py`,
backed by `reference.cpp`. The C++ kernel validates source X once, borrows it,
reuses buffers and fuses softmax/CE/L2/finite checks. Internal class-major matrix
products copy only small parameter/gradient buffers. SciPy receives its original
class-fast layout and uses EXACT reference L-BFGS-B/C=.3/float64/zero-init/gtol1e−5/
ftol64eps/maxiter2000/maxls50. Bias is unpenalized. This remains an existing SciPy
optimizer plus custom C++ kernel, not a fully C++ optimizer or language comparison.
No active producer, production binary, submission or official-submodule change.

## Matched complete fits

Job3118904 passed on galvani-cn102 with six bound logical CPUs. All validation,
callbacks, optimizer work and output reconstruction count. Independent verification
is outside both timed fits. Source/scaler/row/library/affinity hashes and actual
optimizer counts are in `solver343_v4_qualified_prefix.json`.

| Workload | sklearn | v4 full-D C++ | Change |
|---|---:|---:|---:|
| P01 cold fit |18.5253s|15.7469s|−15.00%|
| P01 warm ABBA median |19.7467s|16.4362s|−16.76% (1.201×)|
| P01 warm CPU core-seconds median |117.3624|93.3867|−20.43%|
| P10 independent cold fit |15.4332s|10.7565s|−30.30%|
| P10 CPU core-seconds |91.4340|60.4633|−33.87%|

P01 both use82iterations/88actual value-gradient evaluations; P10 both use65/71.
Independent real mini/full original objective/gradient-vector and NaN/±Inf guards
pass. Six fixed-parameter paired kernel calls per method improved v3 .1005s to
class-major .08164s (~18.8%); this is a kernel result, not an additional full-fit
speedup claim versus v3 measured in a separate allocation.

Fresh-process 10ms sampled memory probes, KiB:

| Measure | sklearn | v4 full-D C++ |
|---|---:|---:|
| Loaded resident baseline |1,414,648|1,293,624|
| Sampled fit peak RSS |1,492,444|1,365,372|
| Increment above loaded RSS |77,796|71,748|
| Process peak including data loading |2,272,392|2,270,384|

The fit increment is6,048KiB smaller (7.77%). Loaded baselines differ by121,024KiB;
do not attribute the complete absolute-peak difference to the solver. Process
peaks are dominated by loading. This is one fresh process per method and two
warm fits per method on one shared node. No energy or general hardware claim.

## Complete label-free held streams

Job3119053 passed CPU2/8GiB in165s allocated wall,141.70s measured script time.
ALL four old Python62-window feature canaries pass FIRST: max feature error
≤3.77e−6 and relative L2≤5.59e−9, against1e−4/1e−6 bounds. The frozen native
causal producer runs explicitly on CPU device0 with original prior/scaler,
pinned source/query rows and within0/full0 permutations. Each session resets.

| Complete stream | Windows | Sessions | Max probability error | Changed labels |
|---|---:|---:|---:|---:|
| P01 within0 |190|1|7.35e−9|0|
| P01 full0 |190|1|5.38e−9|0|
| P10 within0 |577|3|6.65e−8|0|
| P10 full0 |577|3|5.86e−8|0|

ALL1534windows meet≤1e−3 and zero non-tied-label bounds; all labels agree.
P01 uses existing frozen source heads. Because the earlier benchmark did not
save P10 coefficients, one source-only P10 pair was exported on CPU2 with the
same objective/scaler/optimizer/options/counts and source/62-prefix guards before
full inputs were generated. This validates these two people/two orders, not all
eight orders, all folds, an immutable submission artifact or worker compatibility.
No held labels or accuracy were read. Details: `solver343_full_held.json`.

## Failed shortcut retained and scope

The old native370-evaluation solve took~40.8s in a fresh repeat, not historical243s;
busy-node repeats took~46.7s. Its stricter stopping did different work. The initial
source-span Torch solver reached a tighter optimum in~10.73s and passed source
parity, but FAILED held parity: P01 .04063/two labels; P10 .04560/one label. It is
not adopted. Against the old tightly converged native head, that span instrument
has held errors≤.000776 and zero label changes; matching reference optimizer/stop
restores held parity. This supports stopping/trajectory as the main discrepancy.
V3 was separately qualified before the controlled v4 layout change; all failed
and intermediate evidence remains in the companion JSON files and git history.

V4 library SHA256:
`7a2715ad8b912a8c464d304ca1a671bed9dd889deaf5b8bf99907fe6efa939bd`.
Build3118894→kernel3118896→strictsmoke3118902→benchmark3118904→fullheld3119053.
All jobs are complete. Total allocated compute, including failures/builds:
8824CPU-core-seconds=2.4511CPUh;0GPUh (`solver343_compute.json`). Wrong-dependency
smoke3118873 failed before any fit; its log is preserved. No dev/full-system/
sealed/rank/worker/fully-C++-optimizer claim or upload follows from this repair.
