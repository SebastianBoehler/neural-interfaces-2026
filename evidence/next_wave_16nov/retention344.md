# §344 — covariance readout improved; task decoding worsened

All four fits and collector completed on17November2026 before11:46CET. On the already-exposed P09
**INDIVIDUAL-MODEL** instrument, concurrent covariance learning C loses to the exact native reference A.
The fixed population-BN covariance probe improves at the final checkpoint, so retaining that measured
readout does not improve this task decoder. No full-system score, package, upload or rank claim follows.

| Arm | Fixed treatment | Accuracy | Versus A | Cell/class-balanced NLL |
|---|---|---:|---:|---:|
| A | Native40 |64.061%|0|.820121|
| B | Covariance64→native40 |63.736%|−.325pp|.820397|
| C | Same64→task40+covariance |61.175%|−2.886pp|.856511|
| D | Same C; person×class-deranged auxiliary targets |61.891%|−2.170pp|.847770|

C−B is−2.561pp; C−D is−.715pp. C loses to A under all eight pinned delivery orders. Those orders share
one person and seed; they are not eight independent transfer tests. All three registered positive
contrasts fail in this instrument. No family closure or empirical people confidence interval follows.

## Actual checkpoint evidence

Each entry below is SOURCE6-fitted ridge(.1) prediction error on P09 teacher targets, divided by the
source-mean prediction error. Lower is better; values above1 fail to beat that source-mean predictor.
All44239coordinates are retained. These are population-BN linear-readout diagnostics, not causal
SessionBN classification, nonlinear information bounds or proof of physiological signal.

| Arm/checkpoint | Diagonal | Offdiagonal | Slow |
|---|---:|---:|---:|
| All arms initial |.304176|1.150652|1.228978|
| A fine0 |.304176|1.150652|1.228978|
| B/C/D fine0, after identical64pretext |.512896|1.154105|1.101731|
| A fine10 |.357032|1.149854|1.171329|
| B fine10 |.357652|1.157806|1.211054|
| C fine10 |.364392|1.146385|1.173631|
| D fine10 |.386316|1.148089|1.159021|
| A fine40 |.274176|1.150086|1.280035|
| B fine40 |.296958|1.163694|1.322454|
| C fine40 |.260576|1.133976|1.255320|
| D fine40 |.298008|1.144838|1.276287|

Pretext improves the slow-coordinate readout but worsens diagonal/offdiagonal relative to initialization.
The slow readout then worsens during B task training, while diagonal improves. This mixed trajectory
supports neither general forgetting nor a claim that pretext first learned every useful covariance group.
C's final probe is better than A/B/D in all three groups, while its accuracy, NLL and Brier score worsen.
Offdiagonal/slow remain worse than source mean even for C. Better fixed-probe error is insufficient here.

The identical B/C/D64-step pretext source losses decrease: first8→last8 group means are
3.653→2.847 diagonal,1.696→.961 offdiagonal,1.784→1.411 slow. Native task CE, first→last31-step epoch,
is1.1343→.4054 for B,1.1344→.4440 for C and1.1344→.4354 for D. Concurrent auxiliary MSE declines
1.4736→.7403 for C and1.5134→1.1530 for D. These training losses establish active objectives;
they do not establish held task benefit. The task/auxiliary tradeoff remains a competing explanation.

## Guards, scope and cost

The actual native loop/AdamW/OneCycle zeroaux twin reproduces eight native updates, batch trace and
Torch RNG exactly. Full A and B reproduce historical338A/337C final body hashes and every probability
in all eight causal banks bitwise. The snapshot instrument therefore preserves those strong controls.
All four arms save actual body+auxiliary and source/held features at initial/fine0/fine10/fine40.
The collector verifies every artifact/source/row/head/RNG/batch/target guard and prepares all16
independent ridge maps, including their numerical/source-order oracles, before opening P09 labels.

C/D share alpha.5150670026694046, fixed from the true first SOURCE batch body-gradient ratio.1.
D preserves exact epoch-exposure target multisets within person×class and has no row self-assignments.
D can mix sessions within those groups; C>D would not isolate physiological window geometry.
COMMON3teacher P04–P06, source6P04–P08/P10 and full44239targets are unchanged. Neural training is still
PyTorch; only teacher feature extraction reuses the C++ backend. P02/P03 inference never occurs.

Smokes3118674/3118676/3118695, fits3118755_0/1 and3118756_2/3, collector3118759 consume
**3.9483 allocated CPU core-hours**, versus8authorized; Slurm TotalCPU is3.6250hours, peakRSS4,447,004KiB.
There is zero A100 use and no remaining344job. Earlier engineering failure/cold-cost stop remain recorded.
Raw result SHA702237cb…fbf47 and immutable source manifestcecb10cc…7d455d are pinned in compact
`retention344_result.json`; actual checkpoints and full trace stay in its named Galvani directory.

No automatic weight sweep or extra person matrix is justified by this result. A later different learning
mechanism needs its own prospective claim; this negative jointly trained form does not close the family.
