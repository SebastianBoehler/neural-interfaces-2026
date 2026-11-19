# F-C25 · FBLight and raw waveform factorial

F-C25 crosses exact adapted FBLight doubling with the original waveform, bounded hard-label waveform and bounded motor-supervised waveform. The fixed primary treatment combines FBLight and hard-label waveform. Each of the six arms has its own complete RouteB, count filter and causal spatial personalizer. F-C25d adds an independently evolved stock identity. [Registration](../../docs/LEDGER.md#f-c25--cross-actual-fblight-doubling-with-matched-raw-waveform-replacement-on-external-people)

| Instrument | Immutable Galvani release | Jobs |
| --- | --- | --- |
| External · nine Scherer people and fresh order seeds 16411/17411 | `ni26_friday_c25_external_v1_20261119` | Smoke 3134864, array 3134865, scorer 3134866 |
| Held-new · ten PROTEUS people in within0 and full0 | `ni26_friday_c25d_held_v1_20261119` | Smoke 3134878, within0 3134879, full0 3134880, scorer 3134881 |

External smoke finished 08:42:01 CET. It matched both archival anchors and exact private stock on all vectors. Real G was unrecognized with 100 warm FBLight frames. P10 recognized at age 10 and consumed 21 routed cores through each verified branch.

| Receipt | SHA-256 |
| --- | --- |
| External worker manifest | `b2fb378eea66b383dacce27da1085176ff60044323a47f3cc500ad9a9a00b318` |
| External scorer manifest | `0ede0b1212835780d5251e8c7cecc2561d183874d03059559fe95e621767ce5c` |
| External smoke | `731d15247a04b5dec575804c164f56761cb8eb2a319db9d03a21d541cccb809a` |
| Held worker manifest | `ad64fbaeaec446ffa0d3ec44a1facea23a24e5baf2b98daa7386bd940ce6eb10` |
| Held scorer manifest | `bcdb1a1d608a78de8e1e2fec134d9fe1229938aa16d8d333cd238cad3c82584e` |

Reconstruct the external source from `git archive ef3a6244 friday/cap250_f32_external` into a new directory. Verify the parent's `code.sha256` and `scoring.sha256` before applying [contrast.patch](contrast.patch) with `patch -p1` from the extracted subdirectory. The held worker uses `git archive ef3a6244 friday/cap250c22`, its parent manifests and [held.patch](held.patch). Both variants leave measured parent releases intact. The external CPU actor lock is the existing verified F32x lock `0292022abd39067cefebd0e3a0bd8f1834a04bfc67d7d41081cfe6abaae8ebd2`.
