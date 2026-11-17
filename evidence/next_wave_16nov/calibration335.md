# §335 source-held view calibration

Actual start: 17 November 2026,06:59:51 UTC /07:59:51 CET.
Question: does calibrated full-complement evidence improve unseen-source-person prediction,
compared with uniform predictions and a calibrated source semantic/translation model?
The §334 result motivates this follow-up; it is not independent hypothesis discovery.

The finite Gaussian model retains all44,239 dimensions. Source whitening uses rank16plusdiagonal,
with an invertible transform. The semantic rank-two span and complete complement form two views.
For η>0, semantic variance is2/η and residual variance is(D−2)/(ηρ). Gaussian normalization terms
are constant across the finite hypotheses and class labels, and therefore cancel in posterior ratios.
η=0 is the uniform predictive limit, not a finite-variance Gaussian parameter estimate.
No target quotas, pseudo-labels, current-chain probabilities or target-wide normalization enter inference.

Fixed prospective grid: η=.001/.01/.1 and residual multiplierρ=1/100/10000, nine mixture paths.
Each precision also has static semantic, semantic-only translation and original residual-aware
translation controls. Unchanged334(.25,1) and uniform predictions bring the total to20 paths.
Semantic-only translation removes residual evidence about offsets as well as residual class geometry;
it is deliberately different from the original334 translation control.

Each held person P04–P10 uses the corresponding immutable334 source6 fixed-reference bank.
The likelihood refits whitening, class means, semantic basis and finite nuisance priors on six people.
Held-person features enter inference only after fitting. All20×8order banks from all seven folds
must be saved before held-person scoring. P01 is excluded; P02/P03 are never transformed or scored.
All source paths and results are retained. The source-held metric can reject every candidate.

No-held-score smoke3117689 completed53.107s, fold bank generation52.506s, original334 probability
identity exactly0. Independent directGaussian, zero-uniform, neutral/cyclic, prefix/no-future,
reset/deepcopy/A-B-A and final-set-posterior order guards passed on actual source windows.
Projected conservative1328.84CPUcoreseconds including smoke is below the7200 limit.

Immutable source path `/mnt/lustre/work/gehler/gfh527/ni26_cal335_v2_20261117`;
source manifest SHA25402c912a0af2b175669f8f76fffde8fa572b2a517b1c5996144cf5784121f8.
Array3117719 runs seven independent2CPU/24GiB/5min folds, at most four together.
Collector3117726 requires successful completion of the entire array before scoring.
Worst allocated budget including smoke4910coreseconds; expected actual approximately850coreseconds.
No exposed-query release, package, upload or further selection follows automatically.

Primary prediction quality is official-cell/class-balanced negative log-likelihood (NLL), then
Brier score and original official cell-balanced accuracy. Delivery schemes within/full/runs and
people receive equal weights. Uniform NLL must equal log3=1.0986122886681098; its argmax ties
make accuracy uninformative. The inherited NLL clips probabilities below1e−12.
Mechanism activity and prediction quality are distinct. Report residual/permutation logit ranges,
separate geometry/translation/permutation marginal entropies and distances from uniform,
and residual-evidence ablation. More active residual inference alone does not establish benefit.


## Completed source-held results

Array3117719 completed all seven folds; collector3117726 completed4s after all folds.
No P02/P03 inference or score was produced. Uniform NLL passed exactly to2.7e−15.
All source6 exclusions, manifest and independent numeric/session guards passed.
The source grid is selected evidence, without an independent nested calibration test.

| Path | η | ρ | NLL | Brier | Accuracy % |
|---|---:|---:|---:|---:|---:|
|Uniform|0|—|1.098612|0.666667|33.333|
|Original334|.25|1|7.315442|0.939555|50.239|
|static|0.001|—|1.074198|0.650366|54.471|
|semantic_translation|0.001|—|1.074010|0.650209|57.131|
|translation|0.001|—|1.074011|0.650210|57.144|
|mixture|0.001|1.0|1.074011|0.650210|57.144|
|mixture|0.001|100.0|1.074058|0.650247|56.633|
|mixture|0.001|10000.0|1.074259|0.650399|55.444|
|static|0.01|—|0.961653|0.575108|54.471|
|semantic_translation|0.01|—|0.938013|0.559839|56.468|
|translation|0.01|—|0.938245|0.559974|56.418|
|mixture|0.01|1.0|0.938245|0.559974|56.418|
|mixture|0.01|100.0|0.956346|0.570358|55.864|
|mixture|0.01|10000.0|0.964811|0.575191|55.322|
|static|0.1|—|2.735798|0.780608|54.471|
|semantic_translation|0.1|—|3.231960|0.852269|50.988|
|translation|0.1|—|3.252140|0.854246|50.909|
|mixture|0.1|1.0|3.252142|0.854246|50.909|
|mixture|0.1|100.0|2.876608|0.786721|53.877|
|mixture|0.1|10000.0|2.755653|0.760311|55.288|

At η=.01 semantic-only translation improves NLL against uniform on all seven people:
P04 .06352; P05 .03411; P06 .19769; P07 .32118; P08 .30784; P09 .10170; P10 .09815.
Its mean NLL improvement is .160599. Translation improves over static by .023640 at this precision.
The best mixture (η=.01,ρ1) is worse than semantic-only translation on all seven people,
although its mean .000231 NLL deficit is small. Higher residual precision does not reverse this.
The registered ≥.01 NLL extra-residual benefit and ≥5/7-person improvement condition failed.
Calibration rescued useful semantic evidence from the .25 overconfidence; activation did not rescue
useful permutation or residual class inference in this finite source-template model.

## Activation versus useful geometry

At η=.01,ρ10000 residual class contrasts range3.18–4.65 across people, and residual ablation
changes probabilities by .0584–.1991. Thus all seven meet the registered numerical activity condition.
But geometry concentrates on one hypothesis and permutation remains uniform to numerical precision.
This is not evidence of successful geometry transport. At ρ100, geometry L1 distances from uniform
average .480–.591 and permutation only .0040–.0334; residual contrasts remain below .1.
The small intermediate-path permutation response is measurable, not demonstrated useful identification.

A preregistered source-only index audit3117727 completed12s/2CPU. It deterministically refit the exact
P04 six-person model, with no held labels, new score, parameter selection or exposed inference.
In all three P04 sessions at η=.01,ρ10000, posterior mass1.0 selects the global zero class-geometry
and zero-translation hypotheses; all six permutations remain1/6. Atρ1 the geometry probabilities
stay near1/7; atρ100 the null geometry receives .368–.372, with genuine but modest non-null weight.
The global template is zero because pooled class-mean contrasts define the semantic span.
Consequently stronger residual evidence can reject finite source geometries and offsets rather than
learn a transferable class permutation. The index audit establishes this collapse for P04 only;
all seven held people show concentration, but their selected indices were not separately audited.
This is a concrete model limitation, not a claim that EEG lacks relational class structure.

## Next decision

Do not release a P02/P03 geometry claim from this study. The calibrated semantic/translation control
contains useful source-held evidence, while the registered finite-geometry paths add no NLL benefit.
If the whole generative direction continues, preregister a source-only test that separates rejection
of source-template magnitude from class-geometry matching. Preserve the null geometry as an explicit
control and require non-null geometry plus permutation identification and better predictive NLL.
Simply deleting the winning null hypothesis would force activation without proving useful inference.
An exposed-query calibration check would require a separate frozen choice and authorization; none ran.
No family closure, chain integration, package, upload or large expansion follows from these results.

Actual Slurm allocation: smoke55s×2CPU + sourcefolds(61+32+30+29+25+26+52)s×2CPU +
collector4s×2CPU + indexaudit12s×2CPU =652CPUcoreseconds=.1811CPUh. No A100 allocation.
This includes real fit/bank/scoring preparations; it excludes authoring time and other agents' jobs.
Full canonical remote result SHA and all20×7 quality scores are retained in calibration335_result.json.
Repeated per-session/order latent summaries are reduced to min/mean/max in the compact local copy;
complete original marginals remain in the immutable remote result.json. Source/metric/feature hashes
are included. Shared334v4 source7/source6 bank content remains unchanged.

Actual completed outcome clock:17Nov2026,07:10:40UTC /08:10:40CET; elapsed10m49s since start.
