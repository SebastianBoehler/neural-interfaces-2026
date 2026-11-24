# Sealed delivery scheme from the six uploads

21 November 2026 · Track 2 · balanced accuracy (BA) in percent and differences in percentage points (pp). [Preregistration](../../docs/LEDGER.md#sealed-scheme-inference--21-november-registered-at-1933-cet-before-new-runs)

**Estimate among the four schemes: full session 45%, within run 35%, shuffled run order 10%, per-interface blocks 10%.** These are judgmental probabilities. Full has a narrow lead from the order-only PROTEUS rescore fit and the cap250 step. Within leads on the Scherer upload trajectory and the early PROTEUS upgrade proxies. A modest common backend effect makes the rescoring nearly indifferent among all four. Keep the qualified quarter diet on the closest clean PROTEUS evidence and runtime. The separate Scherer transfer flag remains material.

## What the receipts establish

The 10 Nov FAQ edit removed the no-shuffle and chronological-order guarantees. The edit describes a technical backend fix and keeps the exact private order undisclosed. Its public commit changes documentation and competition configuration. The public evaluator still checks that recordings stay contiguous and that each recording arrives in onset order. Replacing only the onset key is one plausible private implementation. The public diff gives no evidence that this is the patch the sealed worker received. A change in scoring, preprocessing, or stream construction could also contribute to the observed rescoring differences. [10 Nov FAQ diff](https://github.com/neural-interfaces26/2026-competition/commit/8734b2f6decd) · [Current upstream FAQ](https://github.com/neural-interfaces26/2026-competition/blob/8447e18119d5116de5d3f0232d467e326774f88a/codabench/pages/scientific_faq_bci.md) · [Current public streamer](https://github.com/neural-interfaces26/2026-competition/blob/8447e18119d5116de5d3f0232d467e326774f88a/benchmark_utils/streaming.py)

| Upload | Submission ID | Exact ZIP SHA-256 | Initial BA % | Comparable sealed BA % | Rescore Δ pp | Step Δ pp |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| NXIF | 1034698 | `0a562e62ce8e75cd2a03919e18a139844d1512d4a54af28d14802cb3b2c71627` | 77.94596 | 77.22920 | −0.7168 | — |
| REC + two tangent cores + TSEM | 1038262 | `9c31a16761729b9f69306de0fe88ee781f2e40c621e6b7cb47277a96a4bf5d36` | 81.96889 | 80.80264 | −1.1662 | +3.5734 |
| route3geb | 1040847 | `74433d9bf7ce983b325c0c2b13a82da764279594a11af27547f729f8d6395ff9` | 82.40266 | 81.34828 | −1.0544 | +0.5456 |
| v4.1 bio | 1048575 | `53fc16797f541a81c5bf96f2c1bdbc2356d270ee8f38f978b9ba672b19da14be` | — | 81.61546 | — | +0.2672 |
| v5.1b own | 1050185 | `4a0bc427f2c9c862ee779db5766badeae93f783446561f4dc8abb43589933bef` | — | 81.97669 | — | +0.3612 |
| v6.6cap250 | 1058969 | `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed` | — | 82.98730 | — | +1.0106 |

I checked all six local ZIP digests against the receipts. The original exact REC score comes from its upload entry. The rescore table rounds that original score to 0.8197. The organisers confirmed equal weight for the new and returning cohort scores on 21 Nov. The pre/post comparison alone cannot establish that the backend used identical cohort treatment before the fix. [Uploads 1–5 and owner rescore](../../docs/SEALED_UPLOADS.md#1011-nov-organiser-rescoring-of-all-track-02-sealed-submissions) · [Upload 6](../../friday/DECISION_18NOV.md#official-penultimate-receipt) · [Organiser clarification](../../docs/STATE.md#rules-upstream-main--8447e181-19-nov)

## What gets compared

- **Exact Scherer replays.** One seed, `71000`, four deliveries per visit, both visits of nine external people. Scherer supplies one interface. A shuffle inside its single interface is equivalent in distribution to full-session shuffle. The exact uploaded ZIPs have no Scherer enrolment overlay, so this instrument tests external new people. It cannot measure the sealed returning branch. Scherer interpolates 22 of 41 channels and has known transfer limits. [NI-V instrument](../wave_19nov/NIV.md)
- **Historical PROTEUS proxies.** The §62 and §169 tapes replay adaptation and routing under changed order. The model banks stay in their original order. The §62 primitive stage levels approximate NXIF and REC+TSEM even though the uploads also changed the base and feature paths. J gives the ten held-new people and two returning people equal cohort weight. Each person's cells are averaged before combining people. These choices differ from a general sealed cell mix. Per-interface orders `74000–74002` shuffle windows inside each interface's block and keep session state across both blocks. §62's archival Graz-first/BH-first findings used chronological windows inside each interface, so they test a different hypothesis. [§62 results in history](https://github.com/SebastianBoehler/neural-interfaces-2026/commit/2a3b985c59f8ddffc644967f036ab0ea786ce43d) · [Historical scorer](../../studies/12nov/segfp3.py) · [Own-space tape scorer](../../studies/13nov/shufpkg.py) · [Order construction](../../cleanslate/harness/data.py)
- **Causal held-new PROTEUS pair.** The v5.1b-like P0r and cap250 replays stream the corrected consuming stages with matched source treatments. They compare reused ten-person folds under one legacy order per scheme. A matched shuffled-run-order reconstruction stopped at a frozen waveform guard before held scoring. There is no paired causal per-interface result for these two upload families. [Within0](../../friday/causal_replay/within0-overnight-queue.md#complete-d-eawcf-result--19-november-0224-cet) · [Full0 and rejected runs0](../../friday/causal_replay/runs0_replay.md)

## Fit to the 10 Nov rescoring

The archival replay reproduced the original §62 score controls and repeated the recording-order labels exactly. Each row averages the two related B2 bases. Full, within and runs use the original §62 one-order fingerprints. Per-interface is the average of three new orders. Each vector is in upload order. Positive losses mean accuracy fell. `c` is a common scoring or backend change and `k ≥ 0` scales the dev loss to the sealed metric. Both fits minimise squared error to the three observed losses `[0.717, 1.166, 1.054]` pp.

| Scheme | PROTEUS J %, NXIF / REC+TSEM / route3geb / v4.1 proxies | Proxy losses, first three | Order-only fit `k` / RMS error | Free-offset fit `c` / `k` / RMS error |
| --- | --- | --- | --- | --- |
| Recording reference | 76.61 / 78.20 / 79.93 / 80.52 | 0.00 / 0.00 / 0.00 | — | 0.98 / 0.00 / 0.19 |
| Full session | 74.79 / 75.56 / 76.04 / 77.98 | 1.82 / 2.64 / 3.89 | 0.334 / **0.226** | 0.58 / 0.142 / 0.148 |
| Within run | 76.82 / 78.08 / 78.84 / 79.43 | −0.21 / 0.12 / 1.09 | 0.912 / 0.805 | 0.92 / 0.174 / 0.165 |
| Shuffled run order | 75.51 / 75.73 / 75.88 / 77.44 | 1.10 / 2.47 / 4.05 | 0.335 / 0.330 | 0.70 / 0.110 / **0.138** |
| Per-interface blocks | 75.21 / 76.12 / 76.41 / 77.17 | 1.40 / 2.08 / 3.52 | 0.383 / 0.291 | 0.69 / 0.123 / 0.157 |

- **The measured change supports full shuffle under a narrow assumption.** Full has the smallest error if a scaled order change explains all three drops. A free common offset leaves a 0.027 pp range across schemes. With only three outcomes, all four residuals are also close to a common-loss-only fit (0.191 pp). The per-interface mean loss, 2.33 pp, falls between within and full as predicted. [§62 protocol and cautions](https://github.com/SebastianBoehler/neural-interfaces-2026/commit/2a3b985c59f8ddffc644967f036ab0ea786ce43d)
- **The earlier correction matters.** The first two sealed drops have a ratio of 1.63. §164's held-new stage proxies yield 1.73 full, 2.45 runs and 2.92 within. The §205 correction explicitly withdrew the original inference from absolute losses. The ±0.25 pp sensitivity on each sealed drop already admits all three ratios. In the three-upload 50/50 fit above, a larger common offset is required to explain the two small or positive changes under within-run delivery. Neither proxy is an exact reproduction of the first uploads. [§164 replay](https://github.com/SebastianBoehler/neural-interfaces-2026/commit/484319ca5888bf393315ff6e98c7f9400b733fc0) · [§205 correction](https://github.com/SebastianBoehler/neural-interfaces-2026/commit/411a3597e440a8949552f2ecc8b94dca0c7cf8f6)
- **Post-fix PROTEUS steps tell a different story.** Across the same first four upload families, a unit slope and fitted common level give score residual RMS 1.19 full, **0.91 within**, 1.46 runs and 1.17 per-interface. Within also leads with slope free (0.61 pp). The measured NXIF → REC change is +3.57 pp while the closest replay of those stages gives +1.26 at best. Dropping NXIF leaves very similar fits for per-interface (0.200) and within (0.223), ahead of runs (0.541) and full (0.771). The historical proxy also changes the magnitude of the route3geb → v4.1 step across orders. That limits the inference from its apparent preference. [Upload receipts](../../docs/SEALED_UPLOADS.md)

## Exact uploaded ZIPs on Scherer

All six archives ran through every delivery. The table gives balanced accuracy for all 18 cells per column. Scherer has one interface, so the full row also estimates per-interface shuffling in this instrument.

| Scheme | NXIF | REC+TSEM | route3geb | v4.1 bio | v5.1b own | cap250 | All 15 pair differences RMS pp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recording reference | 45.28 | 48.12 | 48.54 | 48.54 | 48.54 | 48.84 | 1.050 |
| Full session / one-interface shuffle | 46.10 | 48.75 | 48.64 | 48.64 | 48.64 | 48.47 | 1.632 |
| Within run | 45.48 | 48.85 | 49.68 | 49.68 | 49.68 | 49.45 | **0.983** |
| Shuffled run order | 45.16 | 47.20 | 47.43 | 47.43 | 47.43 | 47.35 | 1.659 |

- **Post-fix steps.** Subtract each dev upload pair from its sealed counterpart and take the RMS. The 15 pairs share six uploads. Fitting a common level at unit slope gives the same ranking and RMS 1.053 full, **0.635 within**, 1.071 runs. It takes a +32 to +34 pp offset to align the absolute levels. A free slope gives 0.807, **0.605**, 0.622 pp. The within-run point fit beats the registered weak-fit thresholds, but the full-minus-within error gap has an 80% people interval of [−0.055, +0.580] pp. Allowing a free slope also leaves runs close. All three routed/own-space ZIPs emitted the same 8,520 labels and recorded zero routes. Their sealed returning gains are invisible here.
- **Rescoring steps.** Recording-minus-shuffled losses on NXIF, REC+TSEM and route3geb were **−0.827/−0.635/−0.099** full, **−0.205/−0.734/−1.144** within, and **+0.119/+0.919/+1.104** runs. Only shuffled-run-order has losses with the same sign as the sealed rescoring. Its order-only error is 0.362 pp vs 0.998 pp for each other scheme. A free shared offset gives 0.164 full, 0.191 within and 0.078 runs. This is a domain-specific signal. Full and within both improve Scherer relative to recording, unlike the main PROTEUS effect. Within beats full by only +0.174 pp [80% −0.437, +0.798] over the first three ZIPs, below the predicted +0.5 pp.
- **Weight of evidence.** Start from Fable's roughly 60% block-preserving prior. The PROTEUS order-only loss model, the full own-space point fit and the cap250 edge move some weight to full. The within-run upload-delta fit keeps within competitive. Shuffled run order fits the Scherer rescoring losses but has the weakest PROTEUS post-fix trajectory, while the per-interface evidence is limited to the historical stage proxies. The probabilities allow for model and cohort transfer error. A weighting that trusts Scherer upload deltas more than the later causal PROTEUS edge can put within-run first.

## Later upload contrasts

| Scheme | PROTEUS v4.1 J % | PROTEUS v5.1b J % | Own-space step pp [paired 80%] | Causal held-new P0r → cap250 % | Cap250 step pp [paired 80%] |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full session | 77.00 | 77.67 | +0.677 [−0.109, +1.463] | 73.07 → 73.74 | +0.667 [+0.033, +1.549] |
| Within run | 79.61 | 80.64 | +1.027 [+0.111, +1.943] | 75.78 → 78.32 | +2.546 [+1.641, +3.406] |
| Shuffled run order | 77.38 | 78.57 | +1.186 [+0.492, +1.880] | — | — |
| Per-interface blocks | 77.25 | 78.07 | +0.825 [+0.297, +1.345] | — | — |
| Sealed | 81.62 | 81.98 | **+0.361** | — | **+1.011** |

The own-space replay uses its own fixed recording-order recognition calibration. It reproduces the saved §169 v5.1b person scores. Both columns in the own-space comparison share each order and use the same ten held-new and two returning people at 50/50. The step is zero on new people. Full has the smallest point error against the sealed gain. The sealed returning mix and source-bank differences limit that evidence. [Own-space production receipt](../../docs/SEALED_UPLOADS.md#12-nov-2112-utc-upload-5-cand_v51b_own-v41--own-space-personal-stage)

At 50/50, a unit transfer of the held-new cap250 step implies a sealed returning step of **+1.35 pp full** or **−0.52 pp within**. If only half the held-new step transfers, those implications shift to **+1.69** and **+0.75 pp**. The independent shared-Sinc full0 check gives +0.479 pp [−0.222, +1.452]. Eight Scherer orders from NI-V supply an auxiliary returning contrast of +1.69 full and +1.79 within, with only tenable sign evidence from that enrolment proxy. Combining those external returning contrasts with the causal PROTEUS new contrasts at 50/50 gives +1.18 full and +2.17 within. For the six sealed receipts, this is the most informative late edge and weakly favors full. The conclusion depends on cross-cohort transfer. [Shared-Sinc check](../../friday/causal_replay/complete-sinc-result-18nov.md) · [NI-V enrolment limitations](../wave_19nov/NIV.md#zero-routing--false-recognition-arm-enrol_runsbatch--fa-usage-and-19-nov-fix)

Absolute held-new levels alone imply cap250 returning BA of 92.24% full or 87.65% within at 50/50. Both remain plausible because the real-fold 79.71% returning mean only covers P04/P10. The early 89.98% returning anchor came from a contaminated instrument. The NI-V combined JSON also drops NaN-padded people from its printed Cohort A levels and uses the obsolete 57/43 mixture. The calculations here use the separately verified PROTEUS levels and NI-V's Scherer-only contrasts. [Real-fold levels](laneTC_selector.md) · [Returning leakage audit](redteam_fable.md) · [NI-V collector](../../experiments/lanes/evaluation/independent-visit-validation/collect.py#L98)

The reconstructed stock fold extension stopped at the zero check. After correcting the worker import, P04's two returning visits matched the production tape. P01's rows matched and 7 of 190 labels differed. The dependency blocked the other deliveries. I retained the earlier causal pair and did not score reconstructed stock under the new orders. Both direct and Galvani-jump Leonardo SSH reached public-key authentication failure in this session. [Stock fold instrument](cap250_pipeline.md#production-launch)

## Consequence for the final package

**Quarter diet `42fb63cf…` remains my choice on clean PROTEUS and runtime evidence.** The independent R8 A100 grid puts quarter ahead of `833a2872…` em_half at 50/50 by +0.179 pp full, +0.222 held within and +0.522 per-interface. The same grid puts quarter +1.53/+1.65/+2.17 pp above hss. The fresh full-session confirmation gave +0.186 pp [80% +0.070, +0.327] against em_half, with −0.016 pp on new participants and +0.389 on the two returning participants. The exact signed quarter ZIP saved 9.74 ms/window in the paired RTX 2080 Ti timing. [R8 audit](fable_r8_scheme.md#complete-leonardo-accuracy-grid) · [Fresh confirmation](DECISION_21NOV.md#quarter-final-checks) · [Signed ZIP qualification](fable_r6_diet.md#production-qualification)

**Em_half is the conservative alternative if the coordinator treats the external returning result as a veto.** On Scherer, quarter − em_half at 50/50 was −0.34 pp full [−0.76, +0.07] and −0.78 within [−1.22, −0.35]. These are same-person visit-1-enrolled comparisons with substantial montage shift. Quarter changed only the recognised path in that instrument. This transfer conflict matters under either delivery scheme and becomes more adverse within-run. The receipt analysis cannot resolve it for the eight unobserved PROTEUS returners. [Scherer final checks](DECISION_21NOV.md#quarter-final-checks)

**A scheme-gated stack fails its own readiness rules.** The latch activated in 9/72 full-session streams (12.5% against the <5% limit). Quarter cleared the preregistered +0.8 pp held within-run robustness bar, and the urn rider gained only +0.036 pp versus its +0.2 pp bar. The R8 grid has no separate shuffled-run-order candidate contrast. [Latch and urn receipts](fable_r8_scheme.md#countfilter-latch-preflight)

## Verification receipts

All cluster inference ran through Galvani Slurm. Release `/mnt/lustre/work/gehler/gfh527/ni26_sealed_scheme_20261121`. Local `python replay/check.py` printed `OK`.

| Evidence | Slurm | Receipt |
| --- | --- | --- |
| Exact archive smoke and replay | 3149893 → 3149894 | All 6 smoke and 18 replay tasks completed. Smoke matched the NI-V full order `50399c0a…f60e2` and repeated it with zero label changes. |
| Scherer score and pair fit | Local score from 51,120 predictions | `external.json` SHA-256 `0655dbafa5f7a06fcd7ffd44c484f7ed7aea841d0cf1e4252248fb55c3ad9425`. I checked all 18 shard digests, each delivery row and the official BA against an independent NumPy implementation. |
| §62 proxy | 3149885 | `proteus_proxy.npz` SHA-256 `26c5b17816a167c7ef8e74c7b773dd1955fdb86f72e8198fbef0bfa6439d5b53`. Original stage values, permuted stream identity and recording-order repeat passed. |
| §169 own-space proxy | 3149903 | `v51_proxy.npz` SHA-256 `96c34ae4124bd48a855e1bb5481af8cd61f749d77702b23edcc502e75d58d23c`. Original person scores passed on both bases and all available pins used. |
| Archived NI-V cap250/v5.1b | NI-V v1 report | `ni26_niv_v1_20261119/out/report_cap250_vs_v51b.json` SHA-256 `c67d1cf6f4cb8d1774b578612ff36f4ab176273ea71e99ab542678f6aebf68fb`. Used only external contrast fields. |
| Stock fold reconstruction | 3149938 → 3149941 | Import fixed on retry. P04 matched. P01 failed the pinned-label prerequisite. Cancelled dependent 3149942 before any new-scheme score. |

Intervals use 20,000 paired bootstrap draws. Scherer draws people with both visits under the single shared seed. The archived own-space analysis resamples people within cohorts and order draws within each scheme, averaging the two related B2 banks inside each order.
