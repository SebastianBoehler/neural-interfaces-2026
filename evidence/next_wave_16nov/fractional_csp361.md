# §361 Fractional class covariance inside the complete cap250 chain

The v6.6 causal personalization stage fits filter-bank common spatial patterns
(CSP) on the most confident half of each predicted class. It uses hard class
assignments in the spatial covariance estimate, then trains a shrinkage LDA on
the resulting log variances. §253 compared argmax labels weighted by their
confidence, filtered labels and hard/agreement selection. Those forms keep
each chosen window in one class. §260 replicated the chosen hard-label stage.

Change only the spatial class covariance estimator. Keep the exact selected rows
and hard LDA labels. For the classes present in the selected rows, renormalize each
selected row's original causal posterior over those classes. Average that vector
with the one-hot hard assignment at fixed weights .5/.5. Use class membership
and its complement to weight the existing one-versus-rest covariance estimates.
Keep the original shrinkage, eigenproblem, predictor, count filter, recognition,
fine-tuning, refit cadence, output fusion and cap. The estimator sees only past
delivered windows of the same session. It requires no additional training data.

Use the untouched archived cap250 control with six dependent bases and eight
pinned orders. Independently replay the original count filter and original
personalization against archived finals. Exact labels and maximum score difference
`1e-8` pass. The real-window cache smoke must retain bitwise CP outputs versus
direct processing, produce active changed decisions, pass strict prefix/session
isolation, and verify unchanged behavior when recognition stops CP before its
first positive weight. Reuse original window-local covariance values in replay
without altering state. The original evaluator loads archival labels in its row
identity guard. The CP algorithm receives EEG, raw predictions and filtered
predictions only. Held accuracy is computed after the full 48-pair cohort.

Predict Fractional−Control **+0.10 pp** [−0.35,+0.55] in new-person official
equal-cell accuracy. Use scheme-equal weights and paired people/order intervals.
Report all bases and B2_pool separately. A negative upper interval refutes this
fixed form. An interval across zero or a small effect remains inconclusive.

Returning people are a separate conditional check. Verify the §321 proof and
control artifact for every order and both held sessions of P04 and P10. If every
actual recognition stops CP before its first possible fusion at index 90,
the modified refit cannot change its predictions: upstream state never reads its
output, and the stage always returns the unmodified filtered result. Confirm
that implication on an actual-window smoke with the largest observed stop index.
Report the result as a structural identity on that conditional proxy, not a fresh
whole returning replay or a guarantee about later-recognized sealed sessions.

One CPU2/8GiB/8min no-score smoke. Release one CPU2/8GiB/40min complete replay
only when all guards pass, projection is below 2,100 seconds, and peak RSS is
below 6 GiB. Maximum 5,760 allocated core-seconds/zero GPU. Frozen submission
source remains the reference. There is no promotion or upload.

Smoke 3122089 passed on Galvani in 109 seconds on two CPUs. On the actual
B2_pool/within0 stream, all 2,655 original scores matched the archive bitwise.
The direct window path matched the cached covariance path bitwise. Fractional
covariances changed 21 decisions before scoring. The prefix, session-reset
and recognition-at-63 guard passed. Peak process RSS was 1,577,844 KiB and the
conservative full replay projection was 1,491 seconds. Full run 3122094 uses
the same pinned source manifest `9e889431…9ccd`.

## Complete full-system result

Job 3122094 completed in 17m17s on two CPUs. The original scores matched all
48 archived controls bitwise. The fractional estimator ran 6,048 refits and
changed 934 predictions across the dependent six-base, eight-order comparisons.
The project replay and all guards passed before scoring with the unchanged
official metric.

| New-person full-system dev | Frozen control | Fractional | Change and 80% official interval |
| --- | ---: | ---: | ---: |
| Six-base mean | 76.217673% | 76.216162% | −0.0015 pp [−0.0469, +0.0490] |
| B2_pool | 76.279335% | 76.222243% | −0.0571 pp [−0.1761, +0.0747] |

The six-base people/order change was +0.0059 pp [−0.0382, +0.0534], with four of
ten held people improving. The fixed effect is inconclusive and gives no
improvement over the frozen system. P04 and P10 are new-person folds in this
comparison. In the separate conditional returning proxy, all 32 audited sessions
stop CP before any fusion, so this change cannot affect their predictions.
Later recognition can still allow CP influence. Returning accuracy was not rerun.

The two allocations used 2,292 CPU core-seconds and zero GPU. Full compute took
958.81 script seconds and peak process RSS was 1,608,700 KiB. Caching was used for
research replay only. No worker latency or sealed score follows.
[Proof summary](fractional_csp361_proof_summary.json) · [Full results](fractional_csp361_result.json)
