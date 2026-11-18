# F-C13 raw-only S1C in one cap250 slot

The primary replacement scored **48.7566% FULL-SYSTEM EXTERNAL** versus **48.7169%** for exact cap250. The **+0.0397 percentage point** effect stays undecided. Both seed estimates changed sign between orders. Keep stock cap250 as the submission reference. This test used the challenge's balanced accuracy cell mean on the reused Scherer cohort of nine people, two visits and two orders. It produced 4,260 matched decisions per arm. [Preregistration](../../docs/LEDGER.md#f-c13--replace-one-live-cap250-raw-waveform-member-with-raw-only-s1c)

| Replacement | Score | Change, pp [paired 80% interval] | Changed | Errors rescued | Correct stock decisions lost |
|---|---:|---:|---:|---:|---:|
| Seed 0, primary | 48.7566% | +0.0397 [−0.3208, +0.3704] | 62 | 24 | 22 |
| Seed 1, replication | 48.7401% | +0.0231 [−0.1852, +0.2315] | 51 | 21 | 20 |

| Replacement | Order 411, pp | Order 1411, pp | Motor imagery, pp | Subtraction, pp | Word, pp |
|---|---:|---:|---:|---:|---:|
| Seed 0 | −0.1521 | +0.2315 | −0.3571 | +0.2679 | +0.2083 |
| Seed 1 | −0.0463 | +0.0926 | −0.1984 | +0.2083 | +0.0595 |

Both seeds helped H and D and hurt E and G. The primary effect ranged from −1.3690 pp on E to +0.8929 pp on H. The primary changed more than ten decisions and had a positive net rescue count, as forecast. The +0.20 pp point forecast and the registered advancement condition were missed. Both arms used member 48's unchanged 1/90 share of the base mix. Their new probabilities reached both tails and the actual adaptation path. All 36 stock sessions matched the archived predictions and work counters. Each candidate reached the mixer on all 4,260 deliveries. Downstream work counters matched stock for both candidates.

| Replacement | A100 paired median time | Maximum time ratio | Maximum after each worker's first full session |
|---|---:|---:|---:|
| Seed 0 | 1.0064× | 1.6947× | 1.0705× |
| Seed 1 | 1.0085× | 1.7846× | 1.0723× |

The 31-window warm-up left a first-full-session cost. The ratio exceeded 1.10× in the registered comparison, and the largest ratios occurred when the replacement ran first on that allocation's first full session. The final column is a descriptive check. RTX timing and held-PROTEUS full-system accuracy remain unmeasured.

The initial bitwise comparison across independent stock and identity runs failed. The repeat verified bit-for-bit forwarding from the hook into the mixer on every control window. Independent stock-to-stock and stock-to-identity differences were both 1.694e−7, with exact member argmax, terminal labels and activity. Recording-order control was bitwise exact. Both replacement seeds passed prefix and interleaved reset checks in the full-stack smoke. The standalone actors matched 8,520 reference decisions outside the stack. Four participants exceeded their 2e−5 cross-device probability tolerance, reaching 2.711e−5 versus the CPU oracle. [Actor qualification](model/README.md)

Galvani 3131684 passed the full-stack smoke at 22:19:43 CET. All nine 3131685 tasks completed, with the last at 22:31:48. Scorer 3131686 locked the bank before reading labels and completed at 22:32:01 CET on 18 November 2026. No Codabench attempt was used.

| Receipt in `ni26_friday_c13_worker_v3_20261118` | SHA-256 |
|---|---|
| `source.sha256` | `5adc8b82340f5328ea86feaff90677b974bbe755ebe75fef6d8a7c087265e4be` |
| `smoke.json` | `b1b393d27d6223013699ce06db0c312dae47ccbe17f61ed6e9792c728c0fa151` |
| `score/before_labels.json` | `db68ca82bfe6397f63628594dabd387c20aeff6bf84a283df8332197fb88dfc2` |
| `score/result.json` | `a928969a0e776a77367af4634b03b394797eba8c637a07eabaf4ed1f9dac5b97` |
