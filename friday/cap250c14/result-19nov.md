# F-C14 · Full-system raw cognitive odds × directional alignment

19 November 2026 · Exact cap250 worker · FULL-SYSTEM EXTERNAL

The cognitive-only arm gained 0.2910 percentage points over exact stock cap250. The combined arm gained 0.1455 pp and lost 0.1455 pp relative to cognitive-only. Both fresh-order effects and the paired bounds are required for advancement. All effects remain undecided and the combined arm missed the accuracy gate. The comparison uses 4,260 matched decisions per arm on nine reused Scherer people and new orders 2411 and 3411. It used the official equal-cell balanced accuracy and paired people, visit and order resampling. [Preregistration](../../docs/LEDGER.md#f-c14--cross-raw-cognitive-odds-with-past-only-directional-alignment) · [Real worker smoke](README.md)

| Arm | Balanced accuracy | Change vs stock, pp [paired 80%] | 2411, pp | 3411, pp | Rescues / losses |
|---|---:|---:|---:|---:|---:|
| Stock cap250 | 47.8836% | — | — | — | — |
| Directional | 47.8340% | −0.0496 [−0.4001, +0.2778] | +0.0926 | −0.1918 | 28 / 30 |
| Cognitive | 48.1746% | +0.2910 [−0.2149, +0.8201] | +0.6349 | −0.0529 | 51 / 39 |
| Combined | 48.0291% | +0.1455 [−0.3737, +0.6978] | +0.3505 | −0.0595 | 54 / 48 |

The factorial interaction was −0.0959 pp [−0.4630, +0.2778], with −0.3770 and +0.1852 pp by order. Combined minus directional was +0.1951 pp [−0.3241, +0.7278]. Combined minus cognitive was −0.1455 pp [−0.3770, +0.0926]. The factor effects averaged over the other factor were −0.0976 pp [−0.3290, +0.1190] for directional and +0.2431 pp [−0.2381, +0.7376] for cognitive. The cognitive average was +0.4464 and +0.0397 pp across the two orders. These averages used the same locked prediction bank and bootstrap design.

| Arm | Motor imagery, pp | Subtraction, pp | Word generation, pp | Warm A100 median / worst |
|---|---:|---:|---:|---:|
| Directional | +0.4067 | −0.4861 | −0.0694 | 0.9946× / 1.0524× |
| Cognitive | +0.2183 | +0.0099 | +0.6448 | 1.0072× / 1.0703× |
| Combined | +0.4861 | −0.4762 | +0.4266 | 1.0097× / 1.0739× |

The worst directional ratio was on Scherer A visit 2 order 2411. Both odds arms had their worst ratio on A visit 1 order 2411. All three arms passed the A100 runtime thresholds. The accuracy gate withheld the RTX follow-up.

Each odds arm consumed 4,260 live S1C predictions and projected 3,540 warm raw-family contributions. On 4,122 windows its raw posterior change exceeded 0.01. Directional and combined retained the same 428 capped covariance updates. All 36 sessions matched stock's work counters and passed the archive and coverage checks. Galvani `3132604` completed its final producer at 00:19:11 CET. Collector `3132605` sealed all nine banks before reading gold and completed at 00:19:21 CET.

| Artifact in `ni26_friday_c14_factorial_v1_20261118` | SHA-256 |
|---|---|
| `before_labels.json` | `d17fbf1e2a94790948f795ab9bb45f4f03ec39e9661634c8892f381111fa8421` |
| `result.json` | `f6aa0b53a02a249ac9cb85c79cb8ad613622039f1d2e6d7828d1baf9a509b641` |
