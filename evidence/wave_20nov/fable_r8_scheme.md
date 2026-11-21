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

## Complete Leonardo accuracy grid

All 84 A100 shards completed by 19:28 CET. Shard 65049470_81 finished all streams near its time limit, and its conditional retry was cancelled by Slurm. The A100 runs the signed archive's BF16 MANAS path. Receipt SHA-256 `f84987a00c7cd4707b83a7005f09ca0030270761d17b21d8ad238435ee79c437`. Intervals are paired 80% intervals from 20,000 people-then-orders draws. Deltas are percentage points against hss.

### Within-run fit 72000–72003

hss balanced accuracy was 76.913% new, 82.483% returning and 79.698% at 50/50.

| Arm | New Δ [80%] | Returning Δ [80%] | 50/50 Δ [80%] | P04 Δ | P10 Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| nine25_fc | +0.802 [+0.177, +1.344] | +1.294 [+0.881, +1.831] | +1.048 [+0.661, +1.420] | +1.562 | +1.027 |
| ba_six FP16 archive | +0.700 [+0.177, +1.178] | +2.524 [+1.421, +3.654] | +1.612 [+1.015, +2.197] | +3.553 | +1.496 |
| em_half | +0.700 [+0.177, +1.178] | +2.357 [+1.065, +3.681] | +1.529 [+0.840, +2.213] | +3.609 | +1.105 |
| Quarter | +0.700 [+0.177, +1.178] | +2.256 [+1.240, +3.305] | +1.478 [+0.919, +2.024] | +3.214 | +1.297 |

### Within-run held 72004–72007

hss balanced accuracy was 77.566% new, 82.426% returning and 79.996% at 50/50.

| Arm | New Δ [80%] | Returning Δ [80%] | 50/50 Δ [80%] | P04 Δ | P10 Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| nine25_fc | +0.614 [+0.036, +1.151] | +1.366 [+0.740, +2.116] | +0.990 [+0.550, +1.432] | +1.958 | +0.774 |
| ba_six FP16 archive | +0.946 [+0.553, +1.342] | +1.543 [+0.907, +2.224] | +1.244 [+0.874, +1.616] | +2.166 | +0.919 |
| em_half | +0.946 [+0.553, +1.342] | +1.902 [+1.088, +2.713] | +1.424 [+0.990, +1.854] | +2.679 | +1.126 |
| Quarter | +0.946 [+0.553, +1.342] | +2.347 [+1.606, +3.031] | +1.646 [+1.245, +2.030] | +3.012 | +1.681 |

### Per-interface 74000–74001

hss balanced accuracy was 74.276% new, 78.451% returning and 76.364% at 50/50.

| Arm | New Δ [80%] | Returning Δ [80%] | 50/50 Δ [80%] | P04 Δ | P10 Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| nine25_fc | +0.612 [+0.043, +1.065] | +0.535 [-0.736, +2.015] | +0.573 [-0.076, +1.288] | +0.946 | +0.124 |
| ba_six FP16 archive | +0.728 [+0.022, +1.257] | +1.792 [+0.304, +3.469] | +1.260 [+0.427, +2.138] | +2.942 | +0.641 |
| em_half | +0.728 [+0.022, +1.257] | +2.572 [+1.099, +4.121] | +1.650 [+0.891, +2.492] | +3.206 | +1.937 |
| Quarter | +0.728 [+0.022, +1.257] | +3.617 [+2.810, +4.643] | +2.172 [+1.623, +2.688] | +4.143 | +3.091 |

### Pinned full 71808–71811

hss balanced accuracy was 73.565% new, 79.375% returning and 76.470% at 50/50.

| Arm | New Δ [80%] | Returning Δ [80%] | 50/50 Δ [80%] | P04 Δ | P10 Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| nine25_fc | +0.271 [-0.184, +0.745] | +0.599 [-0.081, +1.310] | +0.435 [+0.032, +0.858] | -0.041 | +1.240 |
| ba_six FP16 archive | +0.692 [+0.257, +1.071] | +1.624 [+1.235, +1.962] | +1.158 [+0.866, +1.421] | +1.540 | +1.708 |
| em_half | +0.692 [+0.257, +1.071] | +2.004 [+1.648, +2.371] | +1.348 [+1.063, +1.616] | +2.025 | +1.983 |
| Quarter | +0.692 [+0.257, +1.071] | +2.363 [+1.993, +2.701] | +1.527 [+1.241, +1.787] | +2.216 | +2.510 |

Quarter cleared +0.8 pp at 50/50 on held within-run orders. Its +1.646 pp held gain exceeds +1.527 pp on the pinned full-shuffle orders, satisfying Fable's preregistered refutation rule on A100. The direct Turing grid supplies the production-precision cross-check.

## RTX 2080 Ti production-precision validation

The 60 direct held, per-interface and pinned full-shuffle shards completed by 20:38 CET. The live source patches use FP16 MANAS. Independently applying the official metric reproduced every held within-run arm. Zero labels differed between ba_six, em_half and quarter on the unrecognised windows in these three blocks. The signed EEG-only detector matched all live latch ages. Validation receipt SHA-256 `91fc747967f09d1f151b4f2a1f793e0daec10e808daa61b3b404d6ff10eb24f2`.

### Within-run held 72004–72007

hss balanced accuracy was 77.370% new, 82.624% returning and 79.997% at 50/50.

| Arm | New Δ [80%] | Returning Δ [80%] | 50/50 Δ [80%] | P04 Δ | P10 Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| nine25_fc | +0.939 [+0.263, +1.614] | +1.232 [+0.705, +1.902] | +1.085 [+0.641, +1.546] | +1.692 | +0.772 |
| ba_six FP16 | +1.224 [+0.799, +1.660] | +1.411 [+0.804, +1.978] | +1.318 [+0.951, +1.684] | +1.971 | +0.852 |
| em_half | +1.224 [+0.799, +1.660] | +1.803 [+1.070, +2.576] | +1.514 [+1.100, +1.933] | +2.481 | +1.124 |
| Quarter | +1.224 [+0.799, +1.660] | +2.217 [+1.512, +2.908] | +1.721 [+1.324, +2.115] | +2.819 | +1.614 |

### Per-interface 74000–74001

hss balanced accuracy was 74.185% new, 78.252% returning and 76.219% at 50/50.

| Arm | New Δ [80%] | Returning Δ [80%] | 50/50 Δ [80%] | P04 Δ | P10 Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| nine25_fc | +0.797 [+0.258, +1.243] | +0.737 [-0.467, +2.149] | +0.767 [+0.138, +1.479] | +1.349 | +0.124 |
| ba_six FP16 | +0.745 [+0.045, +1.293] | +2.128 [+0.652, +3.604] | +1.436 [+0.591, +2.318] | +3.475 | +0.780 |
| em_half | +0.745 [+0.045, +1.293] | +2.839 [+1.498, +4.255] | +1.792 [+1.046, +2.611] | +3.740 | +1.937 |
| Quarter | +0.745 [+0.045, +1.293] | +3.884 [+2.991, +4.777] | +2.314 [+1.736, +2.859] | +4.677 | +3.091 |

### Pinned full 71808–71811

hss balanced accuracy was 73.660% new, 79.375% returning and 76.518% at 50/50.

| Arm | New Δ [80%] | Returning Δ [80%] | 50/50 Δ [80%] | P04 Δ | P10 Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| nine25_fc | +0.362 [-0.053, +0.782] | +0.599 [-0.081, +1.310] | +0.481 [+0.086, +0.891] | -0.041 | +1.240 |
| ba_six FP16 | +0.503 [+0.141, +0.841] | +1.657 [+1.259, +2.020] | +1.080 [+0.816, +1.331] | +1.605 | +1.708 |
| em_half | +0.503 [+0.141, +0.841] | +2.039 [+1.668, +2.429] | +1.271 [+1.011, +1.525] | +2.025 | +2.052 |
| Quarter | +0.503 [+0.141, +0.841] | +2.363 [+1.993, +2.701] | +1.433 [+1.174, +1.676] | +2.216 | +2.510 |

**R8 decision. Retain the signed quarter ZIP `42fb63cf…3b07`.** The held RTX gain of +1.721 pp exceeds the +0.8 pp robustness bar and the +1.433 pp full-shuffle gain. Fable's Step 2 trigger stays closed. The independent latch test also exceeded the conditional package's <5% full-shuffle activation limit. The sealed shuffle scope and sealed accuracy remain private. Codabench was left untouched.

### Supplemental fit and terminal urn

At 20:45 CET, Leonardo had all 84 shards and Galvani had all 60 validation shards. The single remaining RTX fit shard finished by 20:56 CET. All 84 RTX shard hashes are in the complete receipt, SHA-256 `04806641eecd8ff0105c1e826d74972c6b4cbab017c74f21ddc48633c7ccbdad`. Eleven timeouts and three queued multi-visit tasks were recovered as separate orders. Recovery jobs were 3149890, 3149891, 3149926, 3149929, 3149933, 3149945, 3149950, 3149967, 3149977 and 3150127. The stitch test reproduced a completed real shard exactly and the collector checked all repaired rows against the registered deliveries.

For within-run fit 72000–72003, hss balanced accuracy was 77.280% new, 82.486% returning and 79.883% at 50/50.

| Arm | New Δ [80%] | Returning Δ [80%] | 50/50 Δ [80%] | P04 Δ | P10 Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| nine25_fc | +0.625 [+0.103, +1.080] | +1.326 [+0.895, +1.895] | +0.976 [+0.630, +1.320] | +1.560 | +1.092 |
| ba_six FP16 | +0.475 [+0.048, +0.853] | +2.556 [+1.529, +3.614] | +1.515 [+0.952, +2.066] | +3.551 | +1.561 |
| em_half | +0.475 [+0.048, +0.853] | +2.456 [+1.195, +3.741] | +1.465 [+0.807, +2.117] | +3.676 | +1.236 |
| Quarter | +0.475 [+0.048, +0.853] | +2.220 [+1.303, +3.149] | +1.347 [+0.855, +1.831] | +3.077 | +1.363 |

RTX terminal urn CPU job 3150594 completed by 21:01 CET. Receipt SHA-256 `81baa5996ef96c7845bbdec0253c8e28e7e0a919a18b6297003c9baa9b5bdd12`. All 48 tape hashes matched the direct grid. The metric calculation reproduced the direct grid within 5.68e−14 pp.

| Block | New Δ [80%] | Returning Δ [80%] | 50/50 Δ [80%] | Changed labels |
| --- | ---: | ---: | ---: | ---: |
| Within-run fit | +0.042 [-0.149, +0.242] | +0.295 [+0.094, +0.516] | +0.169 [+0.029, +0.314] | 250 |
| Within-run held | -0.025 [-0.210, +0.159] | -0.059 [-0.489, +0.266] | -0.042 [-0.272, +0.154] | 232 |

The RTX held rider misses +0.2 pp. The repeated local `python replay/check.py` printed `OK`.
