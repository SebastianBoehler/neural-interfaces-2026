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
No native feature tap or optimizer bridge is implemented until §336 supplies its verified API/trace.

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

Upcoming decision: collect target preparation; await §336 verified native-vs-observer/API and source recipe.
Then preregister only a small native-body-gradient/real-target loss/cost smoke, preserving head/RNG/sampler.
No full pilot, coverage claim, P02/P03 inference, or model promotion follows automatically.
