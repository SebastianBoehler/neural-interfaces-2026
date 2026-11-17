# §342 spatial readout: completed pilot, 17 November

Scope: **INDIVIDUAL-MODEL**, SOURCE6 training and previously exposed P09 query.
No FULL-SYSTEM DEV, SEALED LEADERBOARD or ranking improvement was measured.

Job3118608 completed11:15:09–11:16:06CET, exit0:0, on A100cn219.
All32 source-selected fits and numerical/source guards finished before P09 labels opened.
The eight-arm training/inference program took51.19s; summed head-fit calls11.46s.
Peak allocated GPU memory1.114GB. These are component costs, not worker or CPU speed comparisons.

| Frozen encoder | Pooled400 | Temporal1600 | Spatial16400 | Full65600 |
| --- | ---: | ---: | ---: | ---: |
| Pretrained Atom-L |44.630%|40.086%|51.347%|44.199%|
| Matched random |36.369%|36.782%|41.002%|42.511%|

The official metric averages the two P09 interface-cell balanced accuracies.
Spatial16400 keeps all41 electrode positions while averaging four temporal tokens.
Temporal1600 does the opposite. Full65600 retains both dimensions.
Every arm selected C=.001 using held SOURCE-person P10 NLL before query labels.

Pretrained spatial versus pretrained pooled: **+6.717 individual-model percentage points**.
Pretrained spatial versus matched random spatial: **+10.345 individual-model percentage points**.
Both P09 interface cells improve in those contrasts. One person and one seed do not establish transfer.
Pretrained full tokens do not improve over pooled output, despite higher source training accuracy.
This supports readout geometry as a useful knob in this tested model; more dimensions alone did not help.

The stronger original raw PyTorch reference from §338 scores64.061% on P09.
Spatial51.347% remains12.714 points lower. That historical reference differs in device/seed and is contextual,
not a newly matched §342 training arm. No automatic outer expansion or candidate promotion follows.

## Correctness and preserved failure

Exact frozen default Atom-L weights, current seed329 random initialization, source rows, target-free
scalers, source-only C selection and full32-fit manifest were pinned before query metrics.
Same-weight CPU/batched fitted-readout probability error was at most4.858e−6, with zero non-tied changes.
Against old cached pretrained P09 features, pooled/temporal fitted probabilities differed by
1.037e−5/9.072e−6, with zero non-tied changes. The registered probability bound remained1e−3.

Failed pilot3118549 is preserved. Same-weight B3→B8 numerical batch geometry explained the eight
temporal coordinates outside the original feature bound; source anchors passed the unchanged bound.
The prospective repair added fitted-probability checks before labels; it did not assert historical
query-feature identity. Old random encoder identity remains unproven; all new random arms share one state.

Snapshot `/mnt/lustre/work/gehler/gfh527/ni26_spatial342_pilot_v3_20261117`.
Source SHA256 `2ace9a0075c34e8e50f26b04caae1cd4e36b75201d7c4c2c012f614ccd8c2628`.
Pre-metric artifact SHA256 `eea079268526fbd692803f69c00f75e9d8115d0a1834ba31e76a9345cc099d32`.
Exact collected artifacts: [guard](spatial342_guard.json), [results](spatial342_results.json).
The result pins the actual official metric module and pre-metric artifact hashes.

Next decision: retain this scoped positive while prioritizing stronger-reference learning and solver cost.
Further spatial architecture work needs a registered matched learning/control question, not a width sweep.
