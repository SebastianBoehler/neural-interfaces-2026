# §348 CMx7-MM independent instrument qualification

Status: **metadata audit complete; no EEG accuracy, training or full-system result**.
17 November 2026: started 12:22 UTC; initial decision sent 12:29 UTC; completed 12:42 UTC.
Fresh worker retained history; not blind. Applied scientific-discovery, coverage-audit and engineering-workflow.
Canonical STATE/STATUS untouched. No new dataset/weights downloaded. No GPU or Slurm allocation.

## Decision

The existing export is traceable. It **does not qualify a universal cognitive-rule validation claim**.
It can support a separately registered, **condition-associated MI-versus-MA stress test**, using
PROTEUS-only models. Acoustic exposure and task acquisition remain inseparable in these recordings.
Holding out people or sessions cannot remove that confound. There is no WORD or independent
three-class confirmation. No fitting or accuracy scoring was released.

## Primary sources and remaining contradictions

[Author paper, 24 March 2026 v1](https://arxiv.org/html/2603.22933v1), Methods 2.2:
MI means right-arm movement imagery after beeps. MA means adding/subtracting spoken digits, restricted
to 0–9. Tasks use separate randomized recordings; all sessions occur within one visit. The paper reports
18 MI and 72 MA events. It flags participant 008 as incomplete, 010–014 with device failures, and
017/019 excluded after quality inspection. Some exclusions conflict with existing raw/export presence.
The [published version, 20 August 2026](https://doi.org/10.1038/s41597-026-08125-y) was located;
its full text was inaccessible through the publisher identity redirect. Detailed statements above
are attributed to the pinned author version, not silently to the published revision.

[OpenNeuro source](https://github.com/OpenNeuroDatasets/ds007554/tree/c76eec74254a1ba570224e210e744a2a1fcf3111),
revision `c76eec74254a1ba570224e210e744a2a1fcf3111`, 23 March 2026;
[dataset JSON](https://github.com/OpenNeuroDatasets/ds007554/blob/c76eec74254a1ba570224e210e744a2a1fcf3111/dataset_description.json)
identifies CC0 and DOI `10.18112/openneuro.ds007554.v1.0.0`.
Sampled MI sidecars still describe auditory 2-back. Task instructions support MI identity, but do not
prove a harmless copy error. All 67 retained MI recordings contain 19 target rows. The exporter drops
the first, at 25 s, as presumed task-start. TSV labels it target; sampled Status uses the same pulse as
later cues. The remaining 18 match the paper count. First-marker semantics remain an explicit inference.

## Actual exporter and historical consumer

Galvani `scripts/export_mental.py` matches git `e5512a1` byte for byte, SHA-256
`d1dc977a2b961d5fc9c8b1bc0448167dbf720ee8a33ba5be7dbad6063a86fe61`.
It filters/notches/resamples recordings to 120 Hz, scales volts to µV, rejects flat/outlier channels,
then interpolates onto 41 PROTEUS channels. It retains four dataset-local classes.
MA uses nonoverlapping 4 s blocks at 26, 30, …, 198 s, not 72 separate trials.
MI uses beep+0 to +4 s, differing from official +1 to +5 s. Pretask windows at 1, 5, …, 21 s have
no established rest-task semantics. Existing `cmx345_header_audit.json`/`cmx345_status_audit.json`
record raw-header/Status checks. §348 did not repeat those checks or relabel the earlier ≤4.814 ms
sampled discrepancy as a new result.

The corpus was already used for training. Galvani `runs/diag/wave9.sh` calls `experiments/pretrain.py`
with seven older corpora plus CMx7-MM as m8c: EEGNeX/IFNet, seeds 0–5, EA and rotation.
All 12 m8c checkpoints exist. The loader gives CMx7-MM its own four-class head and merges task
recordings by person|session for EA. `docs/archive/OVERNIGHT_2NOV.md`, Wave 9, reports selected old
teacher-pair increment +1.67 against base +2.65, difference −0.98.
**+1.67 is a historical comparison increment, not teacher-alone accuracy.** Its old chain and selection
remain historical. A PROTEUS-only learner evaluated on CMx7-MM people is an untested direction.
A trunk pretrained on CMx7-MM cannot use these people as independent validation without new exclusions.

## New metadata measurement

One-recording smoke passed first: 50 MA rows. Full audit passes exact exporter-derived onsets/classes,
recording-log counts and unique row identities for **all 8,308 rows, 201 recordings and 24 people**.
It never requests archive member X or decodes EEG samples. It hashes existing bytes and owning sources.

| Task | Recordings | Active rows | Pretask rows |
| --- | ---: | ---: | ---: |
| MA | 65 | 2,860 | 390 |
| MI | 67 | 1,206 | 402 |
| N-back | 69 | 3,036 | 414 |

The export includes 008 (one session), 017 and 019. Participant 009 is absent after saturation rejection;
010–014 are absent. There are 63 paired MI/MA sessions. A proposed exclusion of 008/017/019 leaves
21 people and 57 paired sessions. This is a prospective design, not a corrected official census.
N-back has 121 events in every retained recording, another nominal-paper/event-count mismatch.

Exporter logs recover rejected target channels: CP1 in five recordings, F3 in one, P4 in one.
The nominal 19 shared recorded channels therefore reduce to **16 guaranteed native channels**:
`C1 C2 C3 C4 CP2 Cz F4 FC3 FC4 Fz P3 PO7 PO8 Pz TP7 TP8`.
The 25 interpolated channels are a union across recordings, not a per-row mask.

**Every MI crop overlaps 0.5 s of annotated sound; every MA crop overlaps 2.0–3.0 s.**
This is metadata-derived exposure, not measured audio or EEG accuracy. Task is identifiable from that
metadata by construction. Participant splitting cannot fix it. Delaying a 4 s crop cannot remove the
ongoing MA prompt schedule of 2.5 s.

Full manifest: Galvani `runs/diag/external_instrument348_v2/rows.tsv.gz`. Each row carries person,
session, recording, class, start/end, preceding event, overlapping event indices, prompt count/duration.
[Compact evidence](external348_metadata.json) retains hashes, missingness and exposure counts.
Full per-recording evidence: `runs/diag/external_instrument348_v2/audit.json`.
Existing export SHA-256: `f01e9cba08e2cafb1682d7e35185c9018a522a5bd5bf77694fb88b6d933ebe7d`.
Smoke plus two full passes used <7 CPU seconds, 0 GPU. A full pass took approximately 2–3 s.
The compact report pins the final executed audit source.

## Minimum next artifact and controls before accuracy

1. Freeze a person/paired-session manifest before outcomes. CMx7-MM people must not enter fitting,
   calibration, selection or the pretrained teacher. Person independence does not erase campaign selection.
2. Use the 16 guaranteed native channels in identical order, with common-reference CAR. Declare units,
   filters, 120 Hz and the crop mismatch. Any new crop requires a separately pinned raw export.
3. Register one PROTEUS-only MI-versus-SUB comparison with source-only calibration, class order, split,
   people-nested uncertainty, budget and inference guards. CMx7-MM MA differs from PROTEUS subtraction;
   dominant right-arm imagery also differs from non-dominant-hand imagery.
4. Report per person/session. Group correlated recordings and MA blocks; nearby crops are not extra
   independent trials. MI cue index and MA block index must retain different meanings.
5. Include pretask future-condition negative controls and pacing diagnostics. Models cannot consume
   those metadata. Ongoing acoustic/recording differences remain; any positive is condition-associated.

There is no automatic next wave or new accuracy prior. The row-mapping prediction passed.
Fully resolved trial semantics did not: first-cue, metadata and quality-control limits persist.
The ≤10 ms Status prediction received no new test; the earlier sample remains historical evidence.
