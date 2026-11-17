# §343 CPU solver repair

**Qualified engineering implementation: v3 FullReference.** It keeps the full
44,239-feature objective and the reference SciPy L-BFGS-B optimizer/stopping.
C++ borrows and validates source X once, reuses buffers, and computes analytical
value/gradient with fused softmax/CE/L2/finite checks. This is a hybrid existing
SciPy optimizer plus custom C++ kernel, not a fully C++ optimizer/language claim.
No active producer, package, submission or official-submodule change.

## Matched complete fits

Job3118880 passed on galvani-cn102 with six bound logical CPUs. All setup,
validation, callbacks and output reconstruction count. Independent verification
runs outside both timed fits. Input/scaler/source/row/library hashes are in
`solver343_v3_qualified.json`; kernel/finite checks are in `solver343_v3_smoke.json`.

| Workload | sklearn | v3 full-D C++ | Change |
|---|---:|---:|---:|
| P01 cold fit |18.4002s|16.0840s|−12.59%|
| P01 warm ABBA median |19.4943s|16.4958s|−15.38% (1.182×)|
| P01 warm CPU core-seconds median |115.959|93.147|−19.67%|
| P10 independent cold fit |13.2956s|11.4127s|−14.16%|
| P10 CPU core-seconds |79.042|64.447|−18.46%|

P01 both use82iterations/88actual value-gradient evaluations; P10 both use65/71.
The source objective and gradient guards pass. Label-free held within0/full0
maximum probability errors are ≤2.86e−9 on P01 and ≤4.96e−7 on P10, against a
preregistered1e−3 bound. All held labels agree, including all non-tied cases.
Every fixture scaler and source row matches the actual fit exactly. The query
person/count/order/source exclusion and immutable fixture hashes pass.
No held labels or accuracy were used.

Fresh-process 10ms sampled memory probes:

| KiB | sklearn | v3 full-D C++ |
|---|---:|---:|
| Loaded resident baseline |1,414,736|1,412,768|
| Sampled fit peak RSS |1,504,564|1,494,888|
| Increment above loaded RSS |89,828|82,120|
| Process peak including data loading |2,272,388|2,270,440|

The fit increment is7,708KiB smaller (8.58%); absolute fit peak is0.64% smaller.
This is one fresh process per method. Process peaks are dominated by matrix
loading and do not establish a general peak-memory improvement. Timing repeats
are two warm fits per method, on one node with concurrent unrelated work.
No energy/worker/runtime/full-system or generalization-gain claim follows.

## Failed source-span shortcut and its diagnosis

The original native370-evaluation solver took40.8s in a fresh quiet-node repeat,
not the historical243s. Contemporary busy-node repeats took46.7s. sklearn used
weaker stopping; those timings did not isolate language.

The first source-span implementation reduced2465×44239 to2465×2464 and fitted
in~10.73s with a tighter objective. Only a centering-null eigenvalue was removed;
Gram relative reconstruction error was3.35e−15. Source outputs looked similar,
but the untouched reference's held outputs did not: P01 maxprobability error
.04063 and two within0 changed labels; P10 .04560 and one full0 changed label.
Those strict failures remain failures; v1 is not adopted.

On the exact pinned held fixtures, v1 span versus old tightly converged native
has maxprobability errors .000776/.000607 and zero changed labels. Using the
reference optimizer/stopping restores span held parity to <6e−9. This supports
stopping/trajectory differences as the main cause; source parity alone was
insufficient. The reference-optimizer span route costs ~18.51s and extra memory;
the full-D v3 route currently has the better measured tradeoff.

## Provenance and scope

Immutable workspaces are `/mnt/lustre/work/gehler/gfh527/ni26_solver343_*_20261117`.
V3 library SHA256 is
`f8f91b9b1ab723bf13b49a7a87731e4a3792b0d58b2ccc34f26be0d09c006841`.
Build3118872 and strictsmoke3118875 precede benchmark3118880. An incorrectly
queued smoke3118873 ran before its library existed and failed12s; its log was
preserved and corrected3118875 ran only after the actual build. No fit was
produced by3118873. Earlier v1/v2 evidence remains in the companion JSON files.

Future-source metadata corrects the legacy `accepted_gradient_checks` label:
full-D observes accepted iterations; SciPy itself performs the real gradient
stop. The mathematical implementation and options are unchanged by that rename.
One separately registered class-major GEMM layout microbenchmark is pending;
its source is `reference_classmajor.cpp`, not an accepted replacement yet.
