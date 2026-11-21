# Fable Round 8 · delivery scheme audit

[Preregistration](../../docs/LEDGER.md#fable-r8--delivery-scheme-audit-21-november-registered-before-scoring) · [Dispatch](DISPATCH_21NOV.md) · [Quarter ZIP qualification](fable_r6_diet.md#production-qualification)

The sealed evaluator fixes the order within each session for every submission. The organisers have kept the shuffle scope private. This audit compares the same five stacks under within-run, per-interface and full-session delivery. All scores use the official cell-balanced metric and real `production-20261120-32a94ab7` fold assets. Returning P04/P10 use visit 1 for enrolment and visits 2/3 for evaluation. The new and returning results carry equal weight in the joint score. The order blocks reuse people.

## Delivery review and dispatch

- Harness `19ae67ef` is on `origin/main`. Compared the full PROTEUS permutations against its parent at 71000, 71808–71811 and nearby boundary seeds. Seed 71000 matched byte for byte, SHA-256 `605be1e5c9fff593f8475714847a03e8e5554e47e108b4c9b6f4673fd9fcf6f3`.
- Direct calls to `delivery()` without acquisition metadata raised `ValueError` at 72000, 73000 and 74000. With real metadata, every stream retained its rows and the three schemes preserved the specified run or interface blocks. Local `python replay/check.py` printed `OK`.
- Synced the reviewed harness to Last Push and Fable R6 Diet on Leonardo and Galvani. All four copies give SHA-256 `9c057995959fdc9a556c8c79bdf8bb6d9f0baf53074209eed870675b089f8d7d`. The concurrently running quarter confirmation keeps its separate legacy full-shuffle release.
- Audit release `431eff98`. Leonardo `cap250/runs/fable-r8-scheme-21nov`: Slurm smoke 65049416 → 84-task grid 65049470. Galvani `ni26_fable_r8_scheme_20261121`: smoke 3149306 → 84-task grid 3149309. Each grid depends on its own smoke. Leonardo is the A100 reference. Galvani uses the production FP16 MANAS path on the RTX 2080 Ti.

## CountFilter latch preflight

Galvani CPU smoke 3149318 exercised one real fold. Job 3149319 independently ran the signed hss CountFilter view, lag and thresholds on every registered delivery, using the exact fold assets. It opened no accuracy metric. Receipt SHA-256 `01f51f446ad673c2b288ea54dd2bce095c440f06c6f2eb9aad94f05c3173f423`. The latch window below is the first prediction on which the latch can affect output.

| Order block | Streams | Latch by window 60 | Latch at any window |
| --- | ---: | ---: | ---: |
| Within-run fit | 72 | 69 | 72 |
| Within-run held | 72 | 70 | 72 |
| Per-interface | 36 | 6 | 27 |
| Full shuffle | 72 | 7 | 9 |

The 139/144 within-run latches by window 60 release the terminal urn rider. The full-order activation rate is 9/72 (12.5%), above the strict <5% package requirement. Team B and SessionEM weights affect posteriors downstream of the EEG-only detector and cannot fix this rate. The accuracy audit continues.

## RTX smoke retry

Galvani `3149306` passed both returning folds. P01 within-run order 72004 changed one of 190 labels when em_half ran separately from native ba_six. The mismatch blocked grid `3149309`. The amended release `adb2e097` runs em_half and quarter live on RTX 2080 Ti and checks each against its own captured upstream inputs. Smoke `3149334` gates grid `3149335`. Terminal urn smoke `3149336` waits for the first within-fit shard, then rider `3149337` waits for the smoke and the complete grid. The original Leonardo release uses the strict same-tape replay.

## Leonardo held result and urn rider

By 19:16 CET, all 48 A100 within-run shards were complete. The collector matched the independent EEG-only latch at every delivery and reproduced the official metric independently for all within-run arms and cohorts. Held hss was 77.566% new, 82.426% returning and 79.996% at 50/50. Quarter gained **+1.646 pp [paired 80% +1.245, +2.030]** at 50/50, with +0.946 pp new and +2.347 pp returning. P04 gained +3.012 pp and P10 +1.681 pp. Quarter versus em_half gained +0.444 pp returning [paired 80% +0.219, +0.700].

Galvani CPU batch 3149596 verified the rider on one stored A100 P01 stream. The first collector 3149789 completed the replay but hit a NumPy integer serialization error. The fixed collector 3149794 completed by 19:26 CET. Its receipt SHA-256 is `7b5dcdf50bfb6dd16d69cd1afb0318e409b5aa5bcba7730056ba31aa10b19818`. The terminal urn changed 252 fit and 239 held decisions. Fit gain at 50/50 was +0.110 pp [−0.027, +0.253]. Held gain was +0.036 pp [−0.111, +0.175], with +0.031 pp new and +0.041 pp returning. The A100 rider misses the +0.2 pp threshold.
