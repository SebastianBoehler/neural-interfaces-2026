# F-C22 · FBLight and multiresolution waveform substitution

19 November 2026, 06:21:19 CET · FULL-SYSTEM DEV · Ten reused held-new PROTEUS people in two fixed deliveries

The combination gained **+0.538 pp** within0 and **+0.778 pp** full0 against exact cap250. It gained **+0.154 pp** and **+0.289 pp** against FBLight doubling (FBL2) alone. Both paired intervals cross zero. With FBL2 also enabled, the conventional matched pupil reached virtually the same full0 score as the multiresolution pupil. The combination missed its registered improvement criteria and remains an alternate for testing. [Registration](../../docs/LEDGER.md#f-c22--cross-adapted-fblight-doubling-with-multiresolution-raw-substitution)

| System or paired contrast | Within0 | Full0 |
|---|---:|---:|
| Exact cap250 | 78.3241% | 73.7390% |
| FBL2 alone | 78.7073% | 74.2272% |
| Multiresolution alone | 78.5605% | 74.2591% |
| FBL2 + conventional pupil | 78.7874% | 74.5169% |
| FBL2 + multiresolution pupil | 78.8617% | 74.5166% |
| Joint vs cap250, pp [paired 80%] | +0.5376 [+0.2165, +0.9495] | +0.7777 [+0.5778, +1.0025] |
| Joint vs FBL2 | +0.1544 [−0.0712, +0.3856] | +0.2895 [−0.0053, +0.5156] |
| Joint vs multiresolution alone | +0.3013 [−0.0436, +0.7006] | +0.2575 [+0.0581, +0.5164] |
| Joint vs FBL2 + conventional | +0.0743 [−0.0897, +0.2660] | −0.0003 [−0.2228, +0.1647] |
| Interaction | −0.0819 [−0.4123, +0.2146] | −0.2307 [−0.4910, −0.0260] |

The interaction subtracts both individual gains from the joint gain. The full0 interaction indicates that summing the individual gains overstates the measured joint gain. The joint improved five participants in each delivery compared with FBL2 and regressed two within0 and three full0. P09 improved in both. The two deliveries reuse the same people. Intervals resample people 4,000 times over 28 official cells per delivery. Deployment and external performance remain unmeasured.

## Replay identity

Full smoke ran 379 delivered windows across P01 and a second naturally count-latched stream. All three stages of exact cap250, identity, FBL2 and multiresolution parents matched the signed runs bitwise. Both joint arms applied the original adapted FBLight doubling on 2,375 windows per replay and used the same independently authenticated two-pupil banks as F25c. All six arms executed actual count filtering, 126 strict-past CSP refits and 1,395 fusions. The independent scorer recorded and checked both sealed prediction banks before reading labels.

Remote release `ni26_friday_c22_cross_v1_20261119`. Implementation `276fdfd2`.

| Artifact | Job | SHA-256 |
|---|---:|---|
| Source manifest | — | `90e2a71bb75538a8598f61785df18722ec02861f6fd9f562fac4a43c931e9a3c` |
| Scorer manifest | — | `18c37643313077fd0d34d162bca15d575e76a164401d136475fd5eeb943d7d91` |
| Smoke | 3134261 | `abfaa04224ac8bf70679d31eef9d76c7c4de282df02105586f2a223a22cefac1` |
| Within0 receipt | 3134263 | `38bdc8457beec8c185cc82fd9a125c5666d8822bef87ad8bc421c0c99d445f37` |
| Within0 bank | 3134263 | `59be22faf6c0fe6e9d17c22edb23e12955636ebe48e5fd917ad91c240feea716` |
| Full0 receipt | 3134262 | `fc0b66ecae64458d15a585a98d65eaa58b6fd6d2b6aa0488a52687db19302453` |
| Full0 bank | 3134262 | `af867cbf32298f327df08a1bda3b58e90fe14c847ad28a76d2a6b68777f0c3f7` |
| Blind seal | 3134264 | `664f8a6234c216153a08b6a02259325538c87e4547df98303fa6e64559809a39` |
| Official-metric result | 3134264 | `309c3f6d368b4b1034e963233d54df61cb372b62b1b07f289cd95e1df26e63c0` |
