# §345 — source-person-crossfitted task distillation

Owner: fresh_accuracy_345. Fresh chat retained canonical history; not blind.
Status: source/zero smoke passed; honest teacher and two native40 arms queued, no held labels opened.
Primary comparison is INDIVIDUAL-MODEL exposed P09/source6/seed0 against the exact raw native reference.
Current FULL-SYSTEM DEV reference remains76.217673% across new people; no system gain claimed.

## Actual coverage and choice

| Mechanism | Owning implementation and actual scope | What remains distinct |
|---|---|---|
| REVE task distillation283 | `studies/15nov/grid283/teacher.py` fits rv2 on9people and emits the SAME source training windows' noncausal posteriors; outer person excluded | Honest source-person OOF task targets from the strong classical core were not that test |
| Covariance337/344 | `pretraining337/pretext.py` predicts44239unlabelled tangent coordinates;344 retains this auxiliary during classification | Those are structure-reconstruction objectives, not task-aligned teacher probabilities |
| Output edits207 | `coregain.py`52block gains / `wsp_final.py`3×3output transform; previous research audited actual code | Not an EEG-conditioned residual against a full-system crossfit anchor |
| Episodic220 | `episodic.py`63-row episodes/21rows per class/8balancing steps | Not arbitrary causal-prefix class mixtures |
| Generative334/335 | Covariance Gaussian geometry, label permutations, semantic translation; calibration can choose null geometry | Conditional inference still needs a genuinely predictive semantic anchor; another geometry weight is unsupported |
| External317/278 | ShinWG auxiliary on weak312prototype; direct externalWORD pooling already tested | A newly named task-private loss is not new information |

Chosen question: transfer the classical core's task uncertainty into the exact native raw decoder.
This changes supervision within existing EEG information. It is not an architecture-width sweep,
a new source corpus, or proof of missing information. Original source6 raw reference64.061% is the
strong matched pilot comparator. The teacher's often-quoted69% tenfold result is NOT its pilot quality.
Both teacher and raw reference will be measured in this same P09/source6 context.

The separate full-system conditional residual test remains STOPPED: ordinary LOSO source predictions
include P09 in other folds.339's source8 banks are raw-model logits, not nested full-system anchors.
Minimum instrument: one outerP09 context with each source-person's COMPLETE anchor trained without
P09 or that source person; matching final/pre-mix placement and8pinned orders. No such bank was reused.

## Independent data readiness

Current research1051 pinned CMx7-MM as MI/MA, notWORD, with contradictory MI sidecar description and
cue-count/pacing limitations. On Galvani the actual assets exist: `data/cmx7mm/windows_1-30.npz`
(655,837,464bytes) and `data/cmx7mm_raw/sub-001`…`sub-030`, with `ses-01`…`ses-03`.
This inventory is not provenance validation or an accuracy result. It does not establish24versus30
exported participant identities, task labels, native reference/units, cue positions or acoustic controls.
Minimum instrument: export row→raw participant/session/task/trial/cue provenance, compare EDF marker
annotations with TSVs and old exporter, and group overlapping crops by trial before any accuracy.
No new corpus/weight download, terms acceptance or provider execution occurred. No corpus accuracy ran.

## Prospective protocol and resources

Ledger345 owns numeric prior/refutation and exclusions. Source6=P04/P05/P06/P07/P08/P10.
For each source teacher, prior/scaler/original sklearnLR C=.3 fit only the other5; P09 enters no fit.
Per-person native cache provenance and original Python whole-session feature canaries precede fitting.
Original Python causal transform uses only delivered prefixes; independent session reset and prefix
canaries precede labels. Source targets use fixedwithin0; teacher query receives all8pinned orders.
Arms: exact native40A; B=CE+.5*T²*KL, T2; C=teacher posterior derangement within person×trueclass.
B/C use original `SessionNet.dist`, identical1152features/constrained head/SessionBN/batches/optimizer.
Derangement preserves class/person soft means and all posterior vectors, and breaks within-class
uncertainty. B>C would isolate that uncertainty, not every form of teacher information.

Immutable source manifest:787dc5927f7c1067322aa337cc31aceed37e2e7627305cb5335b9412276bb378.
Smoke3118879 COMPLETED67s; source P04 teacher original LR fit+17prefix canary10.16s,16iterations.
Zero-KD native state, RNG and batches exact; active KD changes weights and has31positive-KLsteps.
The functional smoke broadcast one source posterior; scientific training uses genuine per-row OOF
probabilities. Separate3118895 verifies direct KL-only body/head gradients using those actual targets.
Conservative total projection8831CPUcoreseconds=2.453coreh, before added≤.1coreh gradient guard.
Hard requested allocation ceiling3.734CPUcoreh,0A100h; original budget4CPUcoreh+.5A100h.
Teacher3118885 → paired native40array3118886; collector3118887 also requires activity3118895.
Workspaces `/mnt/lustre/work/gehler/gfh527/ni26_distill345_{smoke,teacher,pilot,activity}_v1_20261117`.
Only small JSON evidence is fetched. No full outer matrix, new evaluation harness, package or upload.

## Additional raw/source metadata check (no EEG accuracy)

`cmx345_header_audit.json` reads actual sub001/session01 MI/MA EDF headers and annotation tracks;
EDF annotations are empty. Both files expose32recordedEEG channels plus Status, sample rate250.001Hz.
`cmx345_status_audit.json` reads ONLY the Status signal: its1-valued trigger pulses match all19MI
and73MA TSV events within5ms. This validates those two files' event timing against a measured trigger,
not the contradictory task description or cognitive condition. No EEG sensor samples were scored.
The24person export has8308rows/fourclasses `arithmetic,motor_imagery,nback,pretask`, sourceclassids0–3.
It maps to41channels with25listed interpolated channels. Classids are NOT PROTEUS classids.
For sub001/session01, MI task export contains18onsets30,39,46,…200seconds: the first TSV25s target is
absent. MA export uses26,30,34,…198s (4sgrid), while target cues are26,28.5,31,… (2.5sgrid).
Those differences need an exporter/crop provenance explanation before this becomes a trial instrument.
They do not prove wrong labels; command-source semantics and acoustic timing still differ from PROTEUS.
The first metadata serialization failed on numpy int64 after reading headers; corrected conversion
saved the exact metadata. No scientific model score or source selection was made from that failure.

## Cost-route correction and current jobs

Original Python fullfold teacher cost159.78s invalidated the17window projection; teacher3118885 was
cancelled3m49 and pending students/collector/activity cancelled before labels. CPU-native smoke3118903
failed35s because historical `Native.causal` hardcodedCUDA1; frozen tangent.cpp supportsCPU0 explicitly.
Replacement calls the same frozen library withCPU0, retaining original sklearnLR, and fails on parity errors.
Fullsource smoke3118923 passed85.75s: sourceP04 all3sessions41.384s, original-Python first17perstream
2e−5guard and fresh-prefix exact checks pass. Both zero-KD and active-update checks pass again.
Current main sourceSHA5f3247730ac5ab9f483943062adbc1881280ad795511a1d32de40d272c86d34c.
Corrected conservative total3.898CPUcoreh; root prospectively authorized≤8CPUcoreh/≤.5A100h after cost.
Hard all requested allocations, including cancelled/failed jobs,≤4.23CPUcoreh/0A100h.
Current teacher3119013 → native40array3119014 + KL-only gradient3119015 → guarded collector3119016.
Workspaces `ni26_distill345_{smoke,teacher,pilot}_v3_20261117`. Original snapshots remain immutable.
