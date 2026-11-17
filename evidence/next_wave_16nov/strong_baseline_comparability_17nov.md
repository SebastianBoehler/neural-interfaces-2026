# Strong-baseline comparability — 17 November 2026

**Finding:** §313 BASE is a newly converged unweighted strong-core head, closely matching §305 Nconverged. It is not the historical head called `original` in §306. The 62.716% versus65.318% whole-model gap does not establish a feature/cache defect or a spectral-family limitation.

## Scope and owning artifacts
- Read-only source/metadata/model-vector/prediction audit. No command labels, new accuracy, partial score, training, scheduler change or running-snapshot mutation. Existing complete reports were read as authorized.
- §313 producer: `/mnt/lustre/work/gehler/gfh527/ni26_s313_cpu_fits_v1_20261117`, `source.sha256` SHA348af6f3574ae06e3b3150915fc14ac0cc986b4b590c00b63a58573febfae777. Actual source checks passed.
- §305 producer: `/mnt/lustre/work/gehler/gfh527/ni26_s305_converged_v2_20261116`, `converged.sha256` SHAf989c60e289599a603a653ebe92a8d35dba40500527650abb2001010ef31dbaa. Actual wrapper-source checks passed; base producer manifest21c1af2e…59bf is shared with §313/306.
- §306 complete CPU matrix: `/mnt/lustre/work/gehler/gfh527/ni26_s306_expert_cpu_v2_20261116`, `expert.sha256` SHA19a0c8f9cf9526951e3589001e083ebd2b4b3113d76124e2b210348d8640e107. Actual source checks passed; `source.sha256` is not its manifest name.
- Shared historical references: `ni26_s301_native_reference_all_20261116/runs/s301_reference_all/Pxx.npz`; shared caches: `ni26_s301_native_producer_v3_20261116/runs/s301_native_inputs/Pxx.npz`.

## Verified control definitions
| Item | §313 BASE | §305 Nconverged | §306 original |
|---|---|---|---|
| Representation |44239 strong features only; `design.PARTS['BASE']=('strong',)`|Same strong caches|Same causal strong features|
| Source people/rows |Nine outer-source people, canonical sorted rows|Same source rows/reference guards|Same nine-person frozen reference|
| Source scaler |Fresh StandardScaler on cached native features|Historical reference mean/scale|Historical reference mean/scale|
| Fitted head |New unweighted float64 C=.3 multinomial logistic|New unweighted float64 C=.3 multinomial logistic|Loads historical sklearn coefficients; no refit|
| Optimizer |§309 explicit CPU0 L-BFGS|§301 CPU L-BFGS; separate tight sklearn comparison|Historical `fit_head('lr0.3')`: max_iter2000, default tolerance|
| Target causal kernel |CPU0 CausalNative|GPU1 Native|CPU0 CpuNative|
| Session prior/reset |Same native combine API, source prior parts/counts excluding held person; new handle per session|Same|Same|

- Both converged native fits use zero initialization, weight-only L2/(2*C*N), unpenalized intercept, history10, max_iter2000/max_eval2500, gradient1e−8/change1e−12 and strong-Wolfe. Source: native_core/head.cpp `objective/fit_impl`; §313 solver.py calls the same §309 API with C=.3/device0.
- The historical head is different despite the same C and nominal model class. Its default stopping was already distinguished from a tight solve in §301/305; P01's recorded default16iterations/loss.001929436 differs from converged loss.001598503. Lower training loss is not evidence of better held-person decoding.
- Label convention is the same [0,1,2]: reference_all.py explicitly checks sklearn classes; both native objectives directly index those integer labels. No label permutation was found in the owning paths. I did not open labels.
- §306 `shared` is another control: source person×class×interface weighting. Its63.053% should not be substituted for the unweighted §313/305 control.

## Direct saved-artifact checks, all ten people and eight orders
- Every §313 `strong_cache_sha256` dictionary equals §305 `input_sha256`; all ten historical `reference_sha256` values also match. Different source cache/prior/reference inputs are not supported explanations.
- Fresh §313 versus historical scalers: maximum mean error6.204e−10, scale error1.951e−8, relative scale deviation6.775e−9. No feature differs by even1% in scale. Near-constant-feature amplification is not supported here.
- §313 versus §305 converged heads: source-objective differences≤1.579e−9; maximum coefficient difference after removing the softmax-common class component1.108e−5. Recorded gradients remain small (up to4.43e−7 for313BASE and4.41e−7 for305 across these folds).
- Reconstructed §305 Nconverged predictions = §306 archived-parity `original` + saved305Nconverged delta, aligned by exact row IDs. §306 original's maximum archived error and305Nfixed delta are both≤6.607e−8.
- Across21240window/order pairs,313BASE versus reconstructed305Nconverged: max probability difference.002071 and **4 argmax disagreements**. These are close separately solved controls, not bitwise identical.
- Against306original,313BASE changes **1726 argmax predictions**, max probability difference.7878. This is a substantial historical-versus-converged head change, already present without any spectral feature.

## Aggregation and limits
- Both complete whole-model reports use official participant×session×interface balanced accuracy and equal weighting of the within/full/runs schemes:313BASE62.715998%;306original65.318276%. The specific gap is not a person-equal versus cell-equal comparison.
- The305numbers75.966222%(Nconverged) versus76.217673%(control/Nfixed) are **full-current-system** scores, including mixing/stages/count/CP. Their−.251451pp effect need not equal a whole-member gap. The people/order effect is separately−.113pp with interval crossing zero.
- Verified explanation: the questioned comparison changes the fitted head/stopping context. The strong representation, source caches and historical references agree. CPU/GPU causal arithmetic and independently converged solves explain residual numerical differences only as a remaining hypothesis; their exact contribution was not isolated.
- “Continued optimization removed useful implicit regularization” is plausible, not established mechanistically by this audit. Do not turn it into a new hyperparameter grid or a claim that all converged heads are worse.
- No expensive common-source objective job was needed: complete saved source objectives, coefficients, scalers and prediction comparisons already discriminate the main alternatives. No evaluator was rebuilt or silently repaired.

Pointers: studies/17nov/spectral313/{produce,input_data,solver,design,score_whole}.py; studies/16nov/followups/native_core/{fit,converged,bridge,reference_all}.py and head.cpp; studies/16nov/followups/strong_heads/causal_native.py; studies/16nov/interface_experts/{run,cpu_bridge,head,score}.py; evidence/next_wave_16nov/{spectral_s313_whole_results,native_core_s305_results}.json; §306 complete report at `runs/expert306_results/results.json` in its CPU workspace.
