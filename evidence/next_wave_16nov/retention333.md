# §333 label-free embedding retention — complete

Question: do the exact nine frozen §329 embeddings linearly retain the source-fixed
51 covariance and 328 slow coordinates? This is a limited linear-accessibility
diagnostic, not an information-theoretic bound, classifier gain, or family closure.

All coordinates of the shared §334 static source7 teacher remain. The teacher's
reference and slow center use P04–P10 only. Query P02/P03 features are permitted;
no query labels, source labels, P01, new neural fit, or accuracy selection is needed.
The teacher is fixed before each window, rather than a target-session/prefix teacher.

One fixed ridge penalty .1 after source-only embedding/target standardization;
unpenalized intercept. Report each person's normalized reconstruction error for
2,091 diagonal, 41,820 off-diagonal and 328 slow coordinates separately. Normalize
against the source-mean predictor, not a query-fitted variance. All nine frozen
pretrained/random/pooled/retained/raw-concatenated banks must finish before reading
their query reconstruction results. Constant coordinates are counted and excluded
with source-only threshold1e-12. No ridge/width/query tuning.

Numerical prior: random pooled/retained may recover these source statistics almost
as well as pretrained; pretrained-minus-random relative error improvement expected
0.00 [−.10,+.10]. Raw concatenation expected to reduce slow relative error by
.10 [−.05,+.30]. A difference below .05 on these selected/exposed people remains
weak diagnostic evidence. Even a large gap cannot establish a causal accuracy limit.

Guard: immutable manifest, reference, source/row identity and all nine saved banks;
real-data independent augmented-least-squares ridge oracle and source-order check
≤1e-9. Actual source-only cost probe before held reconstruction. One CPU6/24GiB
five-minute guard, then at most one CPU6/24GiB25-minute pilot if projected full
runtime≤20min. All work Galvani, account≤32 running/244 active. No automatic pilot
release before actual guard/cost review. No package, evaluator or submission change.

§331 classifier embeddings and strong raw body are not part of this first stage.
Their restoration and compute need a separate guard/registration before use.

## Complete result,17Nov08:00CET

Guard3117678passed4s, independent augmented least-squares/source-order max
6.57e−16/1.46e−16. Source-only cost projected76.59s. Pilot3117681passed25s
(24.725sscript); all9fits/maps/numerical/order guards saved before query error
interpretation. No labels or new neural fits. No constant teacher coordinates.

Relative reconstruction MSE below1 beats the source-mean predictor. Values are
P02/P03 separately, not accuracy or independent population confirmation.

| Frozen embedding | Diagonal | Off-diagonal | Slow |
|---|---:|---:|---:|
|raw287|0.950/0.348|1.306/0.944|1.160/0.761|
|pretrained_pooled|0.864/0.670|1.157/1.053|1.131/1.046|
|random_pooled|0.898/0.887|1.100/1.088|1.063/1.090|
|pretrained_retained|0.944/0.668|1.304/1.248|1.265/1.041|
|random_retained|1.004/1.045|1.266/1.421|1.205/1.157|
|pretrained_pooled_raw|0.767/0.341|1.339/1.011|1.212/0.774|
|random_pooled_raw|0.891/0.369|1.350/1.023|1.129/0.810|
|pretrained_retained_raw|0.669/0.336|1.467/1.190|1.092/0.707|
|random_retained_raw|0.790/0.420|1.507/1.308|1.050/0.770|

Pretrained features recover diagonal structure better than matched random features
on both people in these views. Off-diagonal reconstruction is generally worse than
the source mean; extra retained patches and raw concatenation do not fix it. This
is evidence about cross-person transfer of a fixed linear probe. It does not prove
that covariance information is absent, nonlinear recovery impossible, or that this
explains classifier accuracy. Probe regularization was not selected on source-held
people. Scale/domain mismatch and probe capacity remain competing explanations.

Full44,239 targets, both query people and all9banks were retained. Full immutable
remote reportSHA b13dfb5c795fc2a4414de693b4e80cb0e5c1719f28bbf05f1e259a486c98e725.
Compact `retention333_results.json` retains all per-person standardized/relative
errors, map/scaler/row/source hashes and guards. No outer/system/package score or
promotion follows. No additional ridge/width or classification queries released.
