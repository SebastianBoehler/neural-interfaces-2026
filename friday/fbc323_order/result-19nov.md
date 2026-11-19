# F31 · Learn label-free order on interior waveform patches

19 November 2026, 07:09:01 CET. True-order training improved over exact hard CE on seed 1. Both prespecified bounded effects remain **undecided** across seeds. The ordered source head learned direction on training diagnostics, reaching 61.95% and 63.20% versus 51.12% and 51.12% for the shuffled assignment. Held inference uses the original raw decoder. This is standalone DEV on ten fit-excluded PROTEUS people across eight deliveries. [Protocol](prereg-19nov.md)

| Inference | Seed | Exact hard CE | True order | Shuffled labels | Order − hard, pp [80%] | Order − shuffled, pp [80%] |
|---|---:|---:|---:|---:|---|---|
| Bounded | 0 | 59.1613% | 58.7739% | 58.4898% | −0.387 [−0.968, +0.050] | +0.284 [−0.051, +0.623] |
| Bounded | 1 | 58.8654% | 59.5095% | 59.4688% | +0.644 [+0.237, +1.059] | +0.041 [−0.503, +0.601] |
| Ordinary | 0 | 58.5458% | 58.2542% | 58.1562% | −0.292 [−0.773, +0.064] | +0.098 [−0.234, +0.509] |
| Ordinary | 1 | 58.3882% | 58.8093% | 59.0452% | +0.421 [+0.129, +0.726] | −0.236 [−0.692, +0.252] |

Bounded order-minus-hard for P04/P10 was −0.039/+0.590 pp at seed 0 and −0.166/+1.300 pp at seed 1. Against shuffled it was +0.158/+0.320 pp and −1.294/+0.902 pp. Both people are held-new. Each learned arm used 1,958,400 opposing orientation examples per seed across the ten folds, with 474,120 matching sham flips. The original 99-sample FIR radius fit inside the 120-sample auxiliary margin. The source fit compared actual padding, affine descriptor invariance and antisymmetric logits before scoring. Held checked 672 exact first windows, 1,344 exact resets and 1.470e−6 maximum prefix drift. F16 hard checkpoints and decisions were exact.

Ferranti `511623/511624` and Galvani `3134427/3134428` passed. Collector `3134429` sealed all 20 blind prediction files and then opened held labels. The wrong order scheme name produced NaN in absolute scores and per-person effects. Strict JSON stopped before publishing a complete result. Scorer-only retry `3134521` restored the original 4/2/2 scheme aggregation and verified the identical blind bank before reopening labels. It did not refit models or rerun held predictions. The official per-order metric and paired bootstrap were unchanged. [Recovery record](prereg-19nov.md)

| Artifact | SHA-256 |
|---|---|
| Original source actor | `832bf16a1af007cadef7742a2ee3bb08cd27c51a9dccf4ebd9a691f6be38e24d` |
| Original held decision actor | `12d8348244e7769d2e80e8e07158952185ad242e3037d63addf4d78cec368d8a` |
| Original blind decision manifest | `a40d5e40ec256384aad788d6926e56e4183a048c93995918e81c37e09aa05b22` |
| Failed partial JSON | `628b6fc0a57200a8c0e7d7061e876a008a6d7d2869494b51bca84184868afe3e` |
| Scorer-only recovery manifest | `e8b2752ac8c6a137b6dd2ffdfed891331fca86b43b756fee0d0436d4edf78624` |
| Complete results | `d4f491be1e192e5c5a01fcf1d0a20b74aa90d5f0707e1bc21e6d21a525adf3e0` |

The source and blind banks remain in Galvani `ni26_friday_s1c_order_20261119`. Complete results are in `ni26_friday_s1c_order_score_v2_20261119`.
