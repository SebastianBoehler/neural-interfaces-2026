# F15e whole S1C on Scherer

F15e completed 18 November 2026 at 20:59:43 CET. It fit the original q-on, deterministic q×0.75 and whole-q dropout arms on all 2,655 PROTEUS rows at seeds 0 and 1. Scherer contributed no rows to any fit. The assessment reused the nine §411b Scherer people, two visits and pinned full deliveries 411 and 1411. Each endpoint uses the full q vector. [Preregistration](../../docs/LEDGER.md#f15e--external-holdout-for-whole-fixed-feature-dropout)

| WHOLE EXTERNAL DEV balanced accuracy | Seed 0 ordinary | Seed 0 bounded | Seed 1 ordinary | Seed 1 bounded |
|---|---:|---:|---:|---:|
| Original q-on | 42.0271% | 42.0503% | 40.4729% | 40.7937% |
| Deterministic q×0.75 | 40.7804% | 41.2467% | 39.8909% | 39.7983% |
| Whole-q dropout | 41.8221% | 42.1925% | 40.7341% | 40.5754% |

| Paired comparison in percentage points, 80% interval | Seed 0 | Seed 1 |
|---|---:|---:|
| Bounded dropout minus deterministic scale | +0.9458 [+0.2646, +1.6138] | +0.7771 [−0.0298, +1.5873] |
| Bounded dropout minus original | +0.1422 [−0.6184, +0.9028] | −0.2183 [−0.8333, +0.4167] |
| Ordinary dropout minus deterministic scale | +1.0417 [+0.2447, +1.8122] | +0.8433 [+0.0360, +1.6336] |
| Ordinary dropout minus original | −0.2050 [−0.7077, +0.2976] | +0.2612 [−0.6055, +1.1574] |

The registered bounded comparison with deterministic scale remains undecided. Its point gain held on both orders at both seeds, and the ordinary comparison had positive lower bounds on both seeds. The original q-on comparison remains undecided. The separate PROTEUS zero-q mechanism retains its undecided verdict. [F15 held-new result](../fbc323_skip/result-18nov.md)

Ferranti 511128 passed the paired four-epoch source smoke. Jobs 511129_0 and 511129_1 completed all six 40-epoch fits, with 27,200 of 108,800 source row visits masked in each dropout arm. Galvani 3131235 passed the G smoke. All nine banks from 3131236 completed, and 3131237 independently sealed them and scored the 2,130 labels. Every exported label matched the independently audited raw code and cue. Both normalizers passed 432 interleaved reset checks across the combined runs, with 216 exact first-window agreements and maximum prefix error 1.535e−6. The 80% intervals resample people, visits and orders in paired draws.

Remote receipt root is `/mnt/lustre/work/gehler/gfh527/ni26_friday_s1c_skip_external_20261118/`.

| Artifact | SHA-256 |
|---|---|
| `actor.sha256` | `9d552ab6df96d7aaa171cb523b2a90ba9988f2f698e1544c7802bb960d9ea0e0` |
| `score.sha256` | `096dfc7f35a0f8dfa1e6cd142b3fe91812904edce1163ef53b6ab7a635be4600` |
| `before_labels.json` | `ff25cdfee588ad9085ca85d9ac97bf26218621aca5a0b454a9813e2caaff2884` |
| `result.json` | `604eb04ca1cd5536c96ef73d84c138d799d4fd5defd1a4cacf75559d3606312e` |
