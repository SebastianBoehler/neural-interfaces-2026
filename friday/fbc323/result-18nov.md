# F13 bounded past-window influence in S1C

Bounding past normalization weights improved held-new WHOLE DEV accuracy by **+0.668 and +0.708 percentage points** across the two original seeds. Both paired 80% intervals cross zero, so the registered outcome is **undecided**. The exact control reproduced the historical S1C mean of **59.045274549%**. F13 reused the original weights and changed only this evaluation-layer calculation.

| Ten held-new people, eight deliveries | Seed 0 | Seed 1 |
| --- | ---: | ---: |
| Exact original S1C | 58.4879% | 59.6027% |
| Bounded past influence | 59.1557% | 60.3105% |
| Paired difference, pp [80%] | +0.6678 [−0.0581, +1.6432] | +0.7079 [−0.1514, +1.9341] |
| P04 difference, pp | −0.0582 | −0.3445 |
| P10 difference, pp | +0.3635 | −0.1129 |
| Decisions changed | 624 of 21,240 | 606 of 21,240 |

Both full-session shuffled orders gained +1.108 to +1.348 pp across the seeds. Within-interface gains averaged +0.035 and +0.182 pp, and within-run gains averaged +0.750 and +0.710 pp. The official results weight the three schemes equally and use the same 28 participant/session/interface cells as §323. The bootstrap pairs people and delivery orders. These are reused public development people. The waveform remained a standalone decoder.

All 20 Galvani A100 folds authenticated the original roster, raw data, orders and epoch-40 held-excluded checkpoint. The original posteriors matched the archived arrays bit for bit on all 42,480 decisions. The bounded arm preserved all 224 first-window probabilities exactly. Every fold exercised clipping. The activation oracle maximum absolute error across final folds was 0.000001760 and the maximum prefix posterior difference was 0.000001611. Interleaved sessions reproduced the bounded prefixes exactly. The CPU collector authenticated all decision banks before opening held labels.

Galvani smoke 3128529 exposed cancellation in the candidate's FP32 variance. P10 smoke 3128541 passed using float64 moments for clipped channels and the original FP32 values elsewhere. Array 3128543 stopped after several diagnostic prefixes lacked a clipped window. The updated oracle picked the earliest active transition using activations alone. P01 smoke 3128589 passed with exact original posterior and first-window parity. Final array 3128591 and collector 3128611 completed on 18 November by 17:42:25 CET. The cap and paired outcome rule were fixed before any held-label scoring. [Registration and amendments](../../docs/LEDGER.md#f13--limit-persistent-artifact-influence-in-the-original-causal-fbcnet)

| Artifact | SHA-256 |
| --- | --- |
| Original §323 source manifest | `88437e69b5eaf7ff84a3635df88fe68b38394abbc527629b0f41c7fcc24eb6d5` |
| Original §323 whole result | `89dc9aa77860b8ba4f06925294565d50eb7cf5874f310d598bdad24b4d04a92d` |
| F13 replay manifest | `5b83f8c5304619dfbf479b5bad3b7fcec38dc749e935cb1e1b3ebbdd65bac8ec` |
| F13 scorer manifest | `92960fd949df622773d03845f1e3420a847d46d234d5cddca4b03bbf0594de94` |
| F13 decision bank manifest | `e0af1086b94f0d2466cced5339981a9c13a74cbfed586854bcc6601f685a5e24` |
| F13 `whole_results.json` | `bd26268827c2ce7612a61b348c70b1c10d7c9cd54cef030c0034df21df7cdccd` |

The Galvani release `ni26_friday_f13_20261118` contains the authenticated result, fold receipts and preserved smoke attempts. Replay and scoring code are pinned at `b5b2a2b`.
