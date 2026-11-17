# §346 direct information access and raw-source inventory

Actual start17Nov2026,11:23:35UTC/12:23:35CET. Initial source/coverage decision completed by11:31UTC.
Scientific-discovery/coverage-audit and engineering-workflow skills used. This branch inherited STATE,
targetedLEDGER, OFFICIAL_CONTRACT and parent result summaries: fresh worker, not independent historical isolation.
Codebase HEAD at implementation:4b0005fcdb17aba6427b27a87ca8611b4151ac0b.

## Available training information

`inventory346.py` reads only six SOURCE people's BIDS sidecars/events, without EEG or P01/P02/P03/P09events.
Exact per-file hashes and counts: [direct346_inventory.json](direct346_inventory.json).
There are80source recordings/10sessions: ALL500Hz/41EEG/0EOG/0EMG. Event columns are only
duration/onset/sample/trial_type/value. No finer subtraction operand/word letter/response target was found.
Among1,898source command cues, declared duration totals15,184s against7,592s in official1–5s crops.
Another7,592nominal cued-task seconds exist. These are not independent extra trials, validated exported crops
or a guaranteed benefit: cue/feedback conditions and overlap must be retained in a new training intervention.
Twenty source BaselineCE/OE recordings contain3,740s. Native500Hz adds temporal bandwidth, not sensors;
anything unavailable in the official120Hz tensor could serve only training-time supervision.

Actual `studies/17nov/raw312/data.py:export_person` selects only `*task-Baseline*_eeg.edf`.
Its data path did not implement the broader task-outside-official-window form in the original312registration.
Historical core crop pooling was negative; that does not test every native neural training intervention.
No download, external terms, new corpus or additional labels were assumed.

## What this pilot changes

This is direct access to existing current-window information by the strong frozen raw decoder's readout.
It is not new raw information or an architecture-family claim. Mean/power skips already exist in312/320;
core1 has328signedslow samples; SessionEM has32frontal signed-trajectory features; Hilbert/313 cover phase.
Earlier S2 also combined slow/PCA, band-power and gaze inputs. No claim that slow information was globally absent.

Four whole INDIVIDUAL-MODEL arms use exposedP09/source6/seed0:

| Arm | Actual representation/readout |
|---|---|
|A|Exact original338source6/A checkpoint and original constrained native head|
|B|Frozen1152native causalSessionBN features, new sourceStandardScaler+unweightedLR C=.01|
|C|B's1152features plus328signed1Hzlowpass samples and41log40–58Hzpowers|
|D|Same1521width/readout asC; direct369source coordinates deranged within person/session/run|

All direct features use the original M1-referenced41channels/current480samples only. NoCAR, target metadata,
other windows or500Hz signal enters them. Source nativefeatures use one pinnedwithin0 causal session traversal;
queries use all8orders with identicalk20normalization and source-frozen weights/buffers/scalers.
This source readout fitting differs fromA's original training-time32-windowchunk statistics. B is mandatory.
C>B alone cannot establish C>A. The Dshuffle is label-blind but may retain chance/structured class alignment;
it is not a proof that all task information was destroyed. No physiological origin is inferred.

Pre-registration§346 appears inLEDGER before implementation/scoring. Numeric priors C−B/C−D+.5pp[−2,+3],
C−A0pp[−3,+3]; selected one-person instrument, no blind confirmation/peopleCI/system/rank claim.

## Execution and guards

Immutable Galvani source `/mnt/lustre/work/gehler/gfh527/ni26_direct346_v1_20261117`, manifest
cc433f9b3f36dc381debb5b7a803784a67c5b397958e852cfecdbd36dc429dde.
Smoke3119200PASS,CPU2/24GiB,76sallocated/9.112sscript. No held transform/labels.
All144real source rows: native head identity, causalprefix, frozenstate, deepcopy/interleaving/reset,
directbatch independence, active extra coefficients and source-run derangement passed.
Prefix probability error4.46e−7. Features3.263s/threeheads.669s; conservative fullforecast380.28s.
Actualsmoke152coreseconds plus forecast6×(380.28+15)=2523.68coreseconds=.701CPUh <1CPUh.
Single fullpilot3119204submitted11:33UTC,CPU6/24GiB/10min. No automatic grid or outer matrix.
The fullpilot must save all3sourceheads before held transform, all4×8prediction banks and zero/identity proof
before querylabels. A must match every historical probability≤1e−6/zeroalllabels.
The original official metric owns scoring; stocksklearn is explicit, not a §343nativeC.3route claim.

## Completed result and interpretation

Fullpilot3119204COMPLETE0:0,65sallocated/57.479sscript,CPU6/peakRSS4,184,948KiB.
Every originalA probability bank passed historicalidentity beforelabels, maximum8ordererror0.
All three new heads converged in78/89/107iterations; all4×8banks/source-onlyheads/reset/provenance passed.
The producer saved heads_before_query.json BEFORE held transforms, and before_labels.json BEFORE labels.

| Arm | ExposedP09 officialcellBA | Pooled per-windowNLL |
|---|---:|---:|
|A originalstrongraw|64.0610%|.82719|
|B matchedlatentrefit|60.6861%|1.03933|
|C direct369+latent|65.4469%|.95294|
|D deranged369+latent|60.6397%|1.03199|

C−A+1.38590pp, C−B+4.76084pp, C−D+4.80723pp. C−A is positive7/8orders.
The direct summaries improve decisions in this single selected instrument, including against strongA;
the extra-summary signal is not explained merely by greater width in the Dcontrol.
B loses3.37494pp versusA, and C's pooledNLL remains worse thanA. Neither nativehead re-fitting nor
probability quality improved globally. A's NLL here is pooled over windows, whereas337reported cell-equalNLL;
the difference is the diagnostic's weighting, not changedAposteriors (which reproduced EXACTLY).
Allorders share ONEalreadyexposedperson. No people-nestedCI/independent confirmation/family closure follows.
No fullsystem, returning-person, sealed-worker, ranking or promotion claim. The current system already
contains related slow/core/gaze information; another correlated member need not stack.

Compact exact results/proofs: direct346_result.json, direct346_smoke.json,
direct346_heads_before_query.json, direct346_before_labels.json. Source/scaler/head/checkpoint/metric hashes
are retained; EEG, heads.npz and probability arrays remain private onGalvani and are not committed.
Total542allocatedCPUcoreseconds=.15056CPUh,0A100; bothjobscomplete, noneleft. ComputeJSON ownsjobcosts.

The initial next-step proposal used a source7 checkpoint. Root instead authorized the stronger
NO-REFIT transfer instrument below, preserving the exactA6body/readouts rather than changing sources.

Actual final collection17Nov11:35:09UTC/12:35:09CET,11m34afterinitialclock; report/commit followed.
This audit is additional work, not replacement for scheduled14:51CET15minresearch.

## Completed no-refit P02/P03 transfer check

Prospectively appended§346at11:37UTC afterP09result but BEFORE either extensionquerylabel.
Root requested SAMEsource6body, A/B/C/Dreadouts/scalers/C=.01/direct369features, ZEROnewfits.
No source7/A7substitution. ActualA6nativeAcontrol was generated independently for eachperson.
BothP02/P03 were excluded from actualsource6body/readout/scale fits and remain previouslyexposedpeople;
this is a selected transfer check, not independent confirmation. Querycounts are in the complete proof.

Immutable `/mnt/lustre/work/gehler/gfh527/ni26_direct346_extension_v1_20261117`, source
e84ae47c1a8b569ad955a414be4ca6b295c6e26843dd2aaef4a06972bfa506f3.
Job3119224COMPLETE0:0/32sallocatedCPU6/25.358sscript;3.087ssource-onlyrealpathsmoke beforequeries.
Originalhead/scaler array hashes match exactly; reconstructed frozen NumPy versus actualsklearn
source probabilities are BITWISEall3heads. Completebothpeople×4arms×8orders/source/frozenstate/
prefix/deepcopy/cross-personinterleaving/reset guards were saved before either person'slabels.
The separate zero-addition guard is array-arithmetic identity only, not a packaged disabled-member test;
the meaningful originalmodel reset checks execute the unchanged native model repeatedly.

| Person |A originalstrongraw|B latentrefit|C direct369|D deranged369|C−A|
|---|---:|---:|---:|---:|---:|
|P02|65.6460%|57.7460%|64.1639%|56.6377%|−1.48213pp|
|P03|56.2275%|51.7703%|51.3909%|51.8832%|−4.83656pp|

C−B P02+6.41782pp/P03−.37933pp; C−D P02+7.52614/P03−.49227pp.
CpooledNLLP021.03187/P031.37776 remains worse thanA.76537/.97809.
Thus P09's strong-reference decision gain did NOTtransfer with the frozen readouts: both otherpeopleloseA.
The direct summary remains useful over weakB onP02 but notP03; that cannot support robust improvement.
No automaticcurrent-systemconsumer/outermatrix follows; root's bothpeoplegain condition did not pass.
This does not close signedslow/highband information, alternative readout training or rawdata families.

Exactbothperson/allorder/provenance reports: direct346_extension_result.json,
direct346_extension_before_labels.json, direct346_extension_source_smoke.json.
Additional192CPUcoreseconds=.05333CPUh; total734=.20389CPUh,0A100, noownedjobsremain.
Original542coresecond study stays separate above. Newsourcefiles/snapshots preserve every originalartifact.
Actual finalcollection11:40:34UTC/12:40:34CET,16m59after11:23:35start; documentation/commits followed.
