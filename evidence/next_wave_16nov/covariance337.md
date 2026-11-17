# §337 covariance-target pretraining × source exposure coverage

Prepared 17Nov2026; owner scheduled_research_0652. Native-reference bridge owner fresh_solution_audit (§336).
This is preparation and small real-data engineering work, not a released whole-model matrix.
No P02/P03 predictions, query labels, package, upload, new corpus or gated assets are authorized here.

## Exact anchor and matching boundary

Pinned CPU base `/mnt/lustre/work/gehler/gfh527/ni26_raw312_cpu_20261117_v2`,
manifest528284dc7c07c835534ca94e6637c00df44a38755b62313d5cfbfac25a265ef4.
Build original session_bn.make('fbcnet_raw_sbn',d,0): alignFalse, nine4–40Hz bands,width32,
4log-variance summaries,1152features and original3×1152 classifier. The installed classifier uses
PyTorch MaxNormParametrize(.5); preserve underlying `.parametrizations.weight.original` and effective
weights separately. Changing a `max_norm` attribute would not remove that parametrization.
Source scale is sqrt(trace(meanSOURCEcenteredcov)/41), not uncentered RMS. Input cast float64,
divide centered scale, cast float32. Original group32×2source-session chunks/batch64 and `_order`,
SessionBN training, CE unweighted/no label smoothing, AdamWlr.001/wd.001, OneCycle40epochs,
seed0/no augmentation/rotation remain the final supervised recipe. No287skip or replacementclassifier.
P01 excluded from fit/statistics. Source6 quality needs a matched source6 native control; historical
source7 R61.776% is not a source6 parity target. No large fit before §336 zero-control proof.

## Controlled first coverage instrument

First held source person P09; final supervised source6=P04/P05/P06/P07/P08/P10.
Pretext window exposure compares fixed3=P04/P05/P06 with all6, equal steps/batch64 exposures.
The SAME final six-person supervised fit follows either pretext. This is one selected subset contrast,
not general causal evidence for diversity, new people overall, or corpus breadth.
Use a COMMON3-person static teacher reference AND target scaler for both exposure arms.
This prevents indirect six-person aggregate teacher statistics entering the three-person exposure arm.
Person/class exposure counts and repeated/padded rows must be reported before any scientific fit.
The preparation guard uses no labels; source class coverage remains unverified until the source-label guard.

Pretext alone uses original native session chunk permutations, flattening each original batch into
32-row chunks and regrouping pairs across epoch cycles. It never drops a final32-row chunk.
All pretext arms of the same coverage share the identical trace; final supervised native batches remain unchanged.
Sixteen source-only batches are fixture traces, not a committed pretraining duration or final pilot schedule.

## Teacher and objective

All51covariance blocks×861 +328slow coordinates retained:44239, without PCA truncation.
Source-fixed mean-log-covariance reference and slow mean use only COMMON3 rows.
Source standardization also uses only COMMON3 teacher vectors, shared across both exposure arms.
Equal group losses:2091diagonal,41820off-diagonal and328slow standardized coordinates.
Native covariance targets include bands below the native4Hz floor and slow means. They may be structurally
unrecoverable from its log-variance representation. Report separate losses; do not promise full retention.
A train-only DIRECTlinear1152→44239 head has51,007,567parameters, ~194.6MiBfloat32 weights.
Weights+gradients+two Adam moments ~778.3MiB, excluding body, activations and temporary allocations.
Its initialization preserves caller CPU RNG. It must not replace or alter the deployed native classifier.
Native feature tap and optimizer bridge execute only after §336 supplies its verified API/trace.

Future scientific contrasts must include287moment pretext, fullcov+slow pretext, shuffledcovtarget,
extra-supervised matched-step control and adequate native40 baseline. Neither a reconstruction gain nor
head-only optimization proves useful classification. No such large matrix is released at this checkpoint.
Pretext duration, masks, optimization gates, source-held metrics and full resource budget require registration
once the exact bridge and real optimization/cost guard pass. No exposed-query selection is permitted.

## Current execution and owner handoff

Tiny source-only target preparation registered before dispatch, CPU2/24GiB≤5min; no labels or NN fit.
3117831 failed56s at fixture row lookup after constructing teacher/scaler: concatenated person caches
were incorrectly assumed globally sorted. Explicit sorted-ID/inverse mapping fixes the lookup and asserts
all selected row identities. Immutable v1 failure preserved; no source/teacher/sampler/objective changed.
Corrected v2 retry3117832 uses `/mnt/lustre/work/gehler/gfh527/ni26_cov337_prepare_v2_20261117`.
It checks Native/Python full teacher parity, standardization, group coverage and equal source-only exposures.
3117832 COMPLETED54s; actualscript50.855s, full teacher38.606s. COMMON3 teacher/scaler used943rows.
Native/Python44239-coordinate maximum error7.53e−14. All standardized source coordinates pass
zero-mean/unit-variance and full group coverage. Fixture contains256actual source windows and both
16×64=1024exposure traces, with all respective3/6sourcepeople present. No labels or optimization ran.
Source SHA29d1c4b43fdcba0b409c240763d294e1b70d90d527ce5b6cc3257b0410282bb2.
Fixture SHA06644bae5a331022c755184ad71089b08c8e079271917e4a4223eafa8bd3d187.
`prepare.json` records exactcommonrows/reference/scaler/cache/fixture hashes; compactcopy committed.
Actual preparation cost (failed56s+successful54s)×2CPU=220coreseconds=.0611CPUh, no A100.
Shared334v4 source7/source6 banks remain immutable and were not modified.

§336 source7 native-versus-observer proof passed before our dependent guard. API/source/guard hashes are
recorded by the optimization report. Our adapter uses recipe.native(d) with its OWN source6 selected labels,
not recipe.source(), which reads seven-person labels. P09 remains−1; all other excluded labels remain−1.

## Completed tiny optimization/cost guard

3117842 COMPLETED23s/2CPU, actualscript19.871s, eight optimization steps8.912s, MaxRSS2352980KiB.
Native preprocessed input scale25.81793528 and batch/global-row hashes recorded. Native `_train` capture
preserves source preprocessing/order/build, prepares originalSessionBN, and taps actual1152classifierinput.
The native classifier retains its maxnorm parametrization, receives no gradient, and stays EXACTLY unchanged.
Auxiliary initialization preserves the native CPU random state. Body gradients are positive/finite at all8steps,
and body parameters change by maximum.00342449. This is a genuine body-learning path, not head-only fitting.
Repeated source-batch group MSE diagonal/offdiagonal/slow:1.871/2.187/1.537→1.018/1.425/.845;
overall decrease41.235%. Eight steps on one repeated batch establish optimization feasibility, not convergence,
held-person reconstruction, useful classification or adequate original-source6 quality.
Native source6 class counts634/633/631; fixed1024exposure class counts3people337/347/340,
6people324/355/345. Person exposures are intentionally not uniform:3peopleP04:608/P05:224/P06:192;
6peopleP04:256/P05:128/P06:128/P07:32/P08:96/P10:384. Coverage is confounded with this fixed subset
and finite window trace; no causal diversity claim. Complete traces remain in the immutable fixture.

Total own337 allocated CPU=(56+54+23)×2=266coreseconds=.0739CPUh; noA100. MaxRSS preparation
v1/v2=5521600/3477732KiB; optimization2352980KiB. No local training or generated checkpoints committed.
Completed engineering checkpoint actual17Nov07:29:23UTC /08:29:23CET.

Next decision: register a SMALL scientifically matched pretext/extra-supervised/native-control source study,
including a genuine native40 source6 quality anchor and source-held prediction criteria. Full teacher generation,
pretext duration, controls and cost need explicit measured release; no large matrix is authorized here.
CPU optimization does not prove A100 compatibility or cost. Any A100 route needs real native/parity/gradient/session
preflight first. No full pilot, coverage claim, P02/P03 inference, or promotion follows automatically.
