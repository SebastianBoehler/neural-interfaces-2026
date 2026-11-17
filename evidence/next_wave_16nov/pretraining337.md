# §337 matched source6 pretraining pilot

Actual phase start17Nov2026,07:54:05UTC /08:54:05CET. Owner scheduled_research_0652.
This continues prepared337 teacher/optimization guards, not an exposed-person query expansion.
P09 is a previously used source-held instrument (including335), so this is selected evidence.
P02/P03 are excluded from inference/scoring; P01 contributes no fitted statistics or labels.

## Prospective fixed comparison

All four arms use64pretextsteps×64windows=4096exposures and the SAME coverage6input trace,
then exact native40epochs/source6 supervised training. Shared unchanged nativeA6 comes from338.
No duplicate native control training. Coverage3vs6 is deferred; no diversity/corpus breadth claim.
Source6=P04/P05/P06/P07/P08/P10; P09 excluded from training/reference/scaler fitting.
Common teacherreference/scaler use ONLYP04/P05/P06, unchanged from prior337 preparation.
Native inputscale uses allsource6 for EVERYarm, explicitly a common six-person aggregate prior.

| Arm | Pretext objective | Auxiliary head | Matching limit |
|---|---|---|---|
| M |287means/FFTlogpowers, mean/power groups equal|1152→287,330911params|different lossdimension/capacity/FLOPs fromC|
| C |full44239covariance+slow,3groups equal|1152→44239,51007567params|all51blocks+slow retained|
| S |Ctargets shuffled across actual4096exposures|EXACTsame asC|target multiset preserved, every input/target row different|
| E |unweighted native3classCE|original classifier|equal steps/windows, different FLOPs/auxcapacity|

M/C/S preserve the original maxnorm-parametrized native classifier during pretext; E updates it intentionally.
All arms train the native body. Auxiliary initialization preserves native CPU RNG. Final40 supervised epochs
restore the original initial Torch/Numpy/augmentation RNG states and original source batches/dropout sequence.
Learned body weights and BN running statistics survive pretext. Native classifier remains original architecture.
No input mask/augmentation,287skip, replacement taskhead, alignment change or loss weighting is introduced.
64pretextsteps are a small fixed budget, not convergence. A train-batch loss reduction does not prove transfer.
Native9bands4–40Hz may struggle with low/slow teacher targets; report groups separately.

Teacher fullsource6+P09per-window targets use COMMON3reference/scaler exactly. Moment targets use COMMON3
mean/std and original287mean/sixFFTlogpower statistics. Held windows contribute no aggregate fit statistics.
Fixed ridge(.1) probes use populationBN frozen1152features, source6feature normalization/fitting and all44239
teacher outputs. Independent augmented-least-squares/source-order numerical guards precede held error.
This probe measures linear accessibility after supervised fine-tuning, not total information or causality.
It does not separate information learned during pretext from information erased during final40epochs.

## Completed engineering guards

Teacher3117929PASS86sCPU2/78.611script; sourcea2158a60…ca4c9.
Teacher SHA b679c40c5234c2f1de14052c73a36c3e754c1f9f98b42a7d1bdca393ae1b8dcc.
COMMON3reference/scaler hashes exactly match the prepared fixture. Full source6/P09 row hashes retained.

Smoke3118009PASS46sCPU6: all4×8pretext body-gradient/head/RNG/embedding checks; projected4.059CPUh.
Review then strengthened S from a globaldataset derangement to an EXACT finite-exposure multiset derangement.
The deterministic combinatorial assignment uses no score/parameter selection and rejects row identity.
Legacy global map remains in the immutable teacher artifact but is explicitly superseded for S.
Corrected smoke3118015PASS35sCPU6, all4paths; M/C/S/E8steps1.687/3.571/3.711/1.677s.
Conservative own-phase projection14522.18+oldsmoke276+allocation/scriptcorrection≤60=14858.18coresec,
4.127CPUh within root-authorized5CPUh. Shared338A6 compute is accounted by338 separately.
No A100 compatibility or speed is inferred from CPU guards.

## Running jobs and exact control dependency

Canonical root `/mnt/lustre/work/gehler/gfh527/ni26_pretrain337_pilot_v3_20261117`,
source SHA b9970a7e3ad25489f5a2b20ef570a7a53bf1eac2fa6050e9e2278e36765137a1.
Array3118019tasks0–3, at most4running; CPU6threads6/24GiB/10min each.
Queued17Nov08:14UTC /09:14CET, expected4–6min once started; hard10min, no automatic extension/retry.
Collector3118023: afterokALL4 + whole338collector3117917; CPU6/16GiB/4min.
Current queued/running status is not a passing result. No pilot held metrics have been read.
Per-allocation writableTorch/Triton caches prevent inherited inaccessible parent-job cache paths.

A6 control root `/mnt/lustre/work/gehler/gfh527/ni26_combinations338_v2_20261117/output/source6/A`;
source37ee61961a09c79344dd5c92d0395a28e971b5c0f6db7aa2fba94f5aa1a9ec6a.
Checkpoint eadd5c23c9f7cac239bf78a99993d8d3e47392d24d166e0ede6fce5b0e6ac9f8;
8orderbank a3ae189611f5a4d46048f19ec397d758bb07fe47ab7f8984817eb66bdd2596c0.
All4initialnative states, final40batch/RNG/scale/1240steps/classifier/criterion and source-row hashes
must match A6 before any P09label reader. All prediction/embedding/probe-map banks and guards first.
Source quality reuses exact native classifier on populationBNfeatures only with actual16window parity≤1e−5.

## Preregistered readout and limits

Primary P09 official cell-balanced accuracy, equalwithin/full/runs delivery-scheme weights.
Also cell/class-balanced NLL/Brier/class recalls and three full-target reconstruction group errors.
Adequate A6 requires complete40epochs, sourceCE<1, every sourceclassrecall>.10, heldBA>.40;
otherwise report contrasts as weak/inconclusive rather than a prior family ceiling.
Selected priors: C−M+1pp[−3,+5]; C−S+1pp[−3,+5]; C−E0pp[−3,+3]; reconstruction reduction.05[−.10,+.20].
One source-held person/seed cannot settle generalization. No model choice, new precision/epoch tuning,
P02/P03query, chain integration, package, upload or wider matrix follows automatically.
