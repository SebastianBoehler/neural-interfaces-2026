# §349 frozen ONEBN phase-prior counterfactual

Complete17Nov2026,13:06:43UTC/14:06:43CET. Zero optimizer steps; no new classification method/training.
This follows347's late-phase pooling harm on exposedP02/P03/P09. Not independent confirmation/full-system gain.
“Native” here is the original stockPyTorch FBCNet/SessionBN reference, not the customC++ stack.

## Actual mechanism and controls

ActualFBCNet has ONE `spatial_conv.1` SessionBN,288channels,32same-session training chunks,
momentum.1,biasedwithin-chunkvariance and k20causal inference. It averages chunk variances;
it does not add between-chunk/session mean variance. No upstreamBN/dropout exists in this model.
Real64sourceactivation/2chunk oracle reproduced the native mean/variance update exactly;
independent biasedvariance error≤2.10e−8, actualeval/current-prefix consumer error0.

The initial proposed31×40official-only refresh was superseded UNRUN before any349computation.
It would confound phase mixture with final-weight refresh and update count. The actual intervention
used the EXACT347interleaved62×40=2,480source-statistic updates, matching ALL40base+40extra hashes,
seed0base/seed347extra, nativeEMA initialization0/1 and32same-session chunks/padding/multiplicity.

| Arm | Frozen weights | Source of new populationBNprior |
|---|---|---|
|A|OriginalA6raw40|Original prior|
|B|347extraofficialrepeat|Original prior|
|C|347late3.5–7.5pooling|Original prior|
|B_recal|SAMEB|B's own final activations, original+eligibleofficialrepeat|
|C_mixed_recal|SAMEC|C's own final activations, original+actualeligiblelate|
|C_official_recal|SAMEC|C's own final activations, original+eligibleofficialrepeat|

Only each model's running_mean/running_var changed. Allweights,affines,filters,G/scale,k20 were fixed.
No C←B buffer copy, all-net.train/dropout, query aggregate or optimizer. A separateONEBNclone was
put in training mode ONLY to validate native statistic updates on actualsource activations.
Source original1,898rows/late1,894eligibility preserve347's true800end bounds/fourunanchored exclusions.
The primary phase contrast is C_official_recal−C_mixed_recal; genericrefresh controls are B_recal−B
and C_mixed_recal−C. C_official_recal versus strongA determines whether the loss was recovered.

## Source observations, not query selection

| Source pooledNLL | Originalofficial | Qualifiedlate |
|---|---:|---:|
|B|.35525|1.00599|
|C|.54550|.61544|

B's subtraction recall falls .874→.218 across phase. Penultimate subtraction–word class-direction
cosine is .1006(B)/.2845(C), versus motor-imagery–word .8246/.8593, on matched1,894source trials.
These are standardizedsource centroid descriptions, not physiological causality or information absence.
The learnedCmodel decodes late source windows better thanB while worsening original-source task loss.
Genericfixed-weight refresh barely changes sourceNLL(~1e−6); official-only Cprior changes sourceNLL
.545497→.533812. These source diagnostics were fixed before new query forwards/labels.

## Completed query result

OriginalB/C ALL16probability banks reproduced BITWISE before any intervention. Allsixarms/allthree
people/alleightorders, sourcebuffer/unchangedparameter/activity/reset/prefix/net-onlydeepcopy guards
passed before349labels. The original officialscorer and fixed scheme-equal order summary were unchanged.

| Exposed person | OriginalC |C_mixed_recal|C_official_recal|Phase contrast|StrongA|
|---|---:|---:|---:|---:|---:|
|P02|50.5968%|50.5968%|49.7891%|−.80767pp|65.6460%|
|P03|54.2773%|54.2773%|54.1393%|−.13796pp|56.2275%|
|P09|54.2894%|54.2894%|55.3760%|+1.08657pp|64.0610%|

B_recal−B and C_mixed_recal−C accuracy are0 for allthreepeople, despite active buffer/probability changes.
The phase-prior replacement has mixed effects and does not recover347's large damage. C_official_recal
still trailsA by15.85685/2.08819/8.68505points. It does not establish that other normalization forms,
late data or whole families are useless. The source phase/class/task-loss conflict remains; this single
counterfactual does not prove which learned body/head mechanism caused the remaining loss.
No newaccuracy method, jointphasehead/training/grid/current-system consumer or promotion follows.

## Provenance, scope and cost

Immutable `/mnt/lustre/work/gehler/gfh527/ni26_recal349_source_v1_20261117` source
5289c02467542dec14016cc028c92a1ccdbe07723adb3f80c2ee12dd83901a7a.
Immutable `ni26_recal349_query_v1_20261117` source27c00649fb6154116ebe4fb2d78469eccd85a0781395e1e27eadd08401a226e8.
Source bufferfilea7acbc1b1fea8b3c5dde0761cfee3cf4275dc8bba9ac336867062739140ef65e remains private.
Allactualsource/query helpers, model/state/input/batch/permutation/statistic/metric hashes in compact JSON.
No EEG, statistic vectors, models or predictions committed. Full source-only means no query neural
forwards/aggregate/labels during statistic fitting; legacy unused PER-WINDOW query covariance caching
still occurs in AlignedNet construction, as explicitly qualified in346/347. It never enters rawalign=False
predictions or source-fitted G/scale/weights and is not a package/export compatibility proof.

3119534/3119559/3119566 allCOMPLETE0:0:65sCPU2 +53sCPU6 +15sCPU2 =478coreseconds=.13278CPUh,
0GPU/0optimizersteps, nojobsremain. Source52.330sscript/query39.527s; queryMaxRSS1,822,888KiB.
Net-only verification copies avoid the earlier dataset-bearing deepcopy overhead; no broad memory/speed claim.
Combined346/347/349=8,040allocatedcoreseconds=2.23333CPUh/0A100. Not a replacement for14:51CETresearch.
User's new priority is the full active C++ stack;349artifacts stay immutable, no furtheraccuracy branch here.
