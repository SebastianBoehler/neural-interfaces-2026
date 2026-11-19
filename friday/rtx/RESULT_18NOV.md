# D-RTX · Both frozen releases ran on RTX 2080 Ti

**Cap250’s predictions changed after adaptation began.** The frozen v5.1b package matched all 960 A100 labels. The frozen cap250 package matched 906 of 960 (94.375%). Both packages reproduced every archived stage work counter. The cap250-minus-v5 difference fell by **0.729 percentage points** on the same delivered EEG, outside the preregistered ±0.30 pp range.

| Same 960 paired deliveries, official cell mean | A100 | RTX 2080 Ti |
| --- | ---: | ---: |
| Frozen v5.1b | 48.854% | 48.854% |
| Staged v6.6cap250 | 48.750% | 48.021% |
| Cap250 minus v5.1b | −0.104 pp | −0.833 pp |

All 54 cap250 differences followed the onset of neural optimizer refits at zero-based age 20. Scherer A changed 16 labels and G changed 38. G’s first 42 cap250 labels repeated exactly in two separate RTX allocations. This rejects label parity on the measured proxy. The mechanism of divergence remains unisolated. Both visits and deliveries reuse the same two previously examined external people. Codabench’s actual RTX card, drivers, sealed labels, route mix and duration remain unknown.

| Paired RTX allocation | Old first stream / other three | Cap250 first stream / other three | Warm cap250 / v5.1b |
| --- | ---: | ---: | ---: |
| Scherer A, old then cap250 | 39.612 s / 57.061 s | 85.268 s / 112.377 s | 1.969× |
| Scherer G, cap250 then old | 29.246 s / 38.652 s | 65.719 s / 79.476 s | 2.056× |

Each container loaded the immutable Solver with the same Linux worker SIF and pinned official adapter. The measured streams used one EEG window per prediction and the official fresh-model reset on every session. Peak cap250 reserved CUDA memory was 2.672 GB on both cards. The staged weights set the session fine-tune timer to 1e9 seconds. The 250-window count limit bounds the refits. Its session timer cannot materially reduce work under an ordinary evaluation deadline.

Galvani smoke 3128675 completed in 3m52s. Paired jobs 3128676_0/_1 completed in 5m23s/4m06s. Collector 3128677 completed in 12s after sealing predictions. Replay printed OK. Owning snapshot `ni26_friday_frozen_rtx_20261118`, source commit `dddb9db`.

| Artifact | SHA-256 |
| --- | --- |
| `rtx_source.sha256` | `aa753873dea152a1d860ba862d1994809c627a59f52dc972410be54d1116663c` |
| `rtx_before_labels.json` | `d900a44fc45c2124468b5f41af602ce587d5269dbefe479d244560b72c2533b5` |
| `rtx_result.json` | `5dba731255008c5070999fe68cae9c136d632b98321d160ed607ef8ef8ebd368` |

This measures hardware sensitivity of the staged release. Package selection still depends on the broader external evidence, the known sealed v5.1b score and the absence of a demonstrated replacement. [D-RTX preregistration](../../docs/LEDGER.md#d-rtx--frozen-package-conformance-on-an-rtx-proxy)
