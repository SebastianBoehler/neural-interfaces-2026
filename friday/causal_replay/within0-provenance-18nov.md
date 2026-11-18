# Held-new within0 source provenance

The §430 B2_pool uses archived `FR17` from `ttt18within0LPC_{fbraw,fb,fbl,sinc}_base-all_loso-s{0..5}.npz` and pairs seed 0 with each family's `r15c250` output. §257 emitted canonical global rows, while the `within|0|all_loso` permutation defines delivery. All eight seed-0 base/refit files match byte-for-byte between the canonical Galvani repository and the immutable `ni26_s318_coral_fbl_20261117/runs/diag/ttt/probs18` archive.

| Actor and historical mixer slot | §257 within0 fit | Frozen state proposed for P01 |
| --- | --- | --- |
| FB raw, 5 | `3103241_0`, RTX 2080 Ti, cn114 | `3131435_0`, RTX 2080 Ti, cn111, `fbcnet_raw_sbn`, k=20 |
| FB EA, 6 | `3103241_3`, RTX 2080 Ti, cn114 | `3131438_1`, RTX 2080 Ti, cn112, `fb_sbn_rot05`, k=2 |
| Sinc, 7 | `3103241_2`, RTX 2080 Ti, cn112 | Fresh `3131531_0`, A100, cn222, `sincshallow_ea_ptm7r_rot05_sbn`, k=20. The within0 bank shares that exact state with full0. |
| FBLight, 8 | `3103241_1`, RTX 2080 Ti, cn114 | `3131437_3`, A100, cn229, `fblight_ea_rot05_sbn`, k=5 |

All four historical tasks used eight CPU threads and 64 GB, task order `fbraw,fbl,sinc,fb` and the original held-new folds. Slurm recorded `SCHEME=within PSEED=0 KIND=v51b ARMS=s15,r15 CAPS=100,150,250` from the canonical checkout. The four logs report pseudo-target `chain_v51b_within0.npz:384b097cdd4e`. The frozen path fits the other nine people, then builds each held session's causal waveform and forwards the unadapted model. That path precedes use of the pseudo-targets. The original job receipts identify the hardware and invocation. Parity of restored states still requires direct posterior comparison.

| §430 and prior checks | SHA-256 |
| --- | --- |
| `runs/chainout/s281_v66p_control_within0.npz` | `1783b05b9d7d3c0c23f31936ba8657f3d858a97d94661adce59a65c280ec7d48` |
| `runs/s281_v66_cp/zero318_within0_B2_pool.npz` | `da072fae3f18a3ef104396813fc463a72e9945ed52675e5ef2f9af5847dda931` |
| `runs/s281_v66_cp/control_within0_B2_pool.npz` | `65f0c1c60205bbea67df37701754fa6e3691e0997509dc063a79d19847163556` |
| `runs/s281_v66_route/within0.json` | `3a16eb9ce67d8451164f7038946d9753f317f49db8e96a686d5f16d36603925b` |
| `ni26_match456_stages_20261118/stages_within0.json` | `0d65e20fc8d2df8957425bd6b7373b3638b984c37f3ca83a70c998b0eca286fd` |
| `ni26_match456_stages_20261118/stages_within0.npz` | `a8cc81b662c2e0f2a8bd594a909cb18ec8588a71274057753834564bc723f153` |
| Prior EA bridge `orders_v2/within0/status.json`, 3131072 | `63353ba2bb3e123e624000370e967bee193f7ebfb94dcffe6688d21e93efc881` |

The contrast gallery uses the §430 recording-order B2_pool shadow. `recording_calibration()` rebuilds it from the original recording anchors with the identity delivery. The prior within0 EA bridge reproduced the archived incumbent pre-EM, P0r and recognition controls with zero routed sessions in its historical and corrected arms. Its status precedes the new Sinc input. Any subsequent paired bridge must recompute real recognition before accepting targets.

The [probe input manifest](within0_frozen_inputs.sha256) pins the tested bases, previous producer receipts, exact states, source and delivery. If a state cannot reproduce the original base, a future separately registered matched instrument must replace only that family's seed-0 `FR17` source stream in both arms. Apply the common mixture change `(new seed0 − old seed0) / 6` exactly once. Keep the five remaining frozen seeds and the original recording calibration fixed. Regenerate each arm's mixer, recognition, causal shadows, strict-past targets and all four dependent refits using its own verified base. The archived zero is retained separately.

Probe v1 (3131630) stopped on waveform metadata before loading any state or computing posteriors. P01 has one recorded session, so the interleaved reset check uses 64 P02 visit-1 waveforms from the pinned within0 delivery. The probe discards those predictions. Only P01 is compared to archived posteriors. The v2 release preserves the preregistered qualification thresholds.

## P01 frozen probe

Galvani 3131632 completed on RTX 2080 Ti cn111 in 14 seconds, at 21:58:27 CET. Job 3131631 stopped before inference because its assigned cn122 GPU was unavailable. The successful retry reused the exact v2 sources and excluded cn122.

| Input compared with original within0 LPC base | Maximum posterior change | Changed decisions |
| --- | ---: | ---: |
| Restored FB raw | 0 | 0/190 |
| Restored FB EA | 0 | 0/190 |
| Restored FBLight | 5.13805e−7 | 0/190 |
| Fresh Sinc | 1.22853e−5 | 0/190 |

All three restored states passed the fit-row, fixed-state, first-window, prefix and interleaved reset checks. The FBLight posterior stayed inside the 2e−5 threshold despite its A100 fit, refuting the forecast for P01. Fresh Sinc matched its within0 fold output bit for bit and changed the old stream above 1e−8. Prior bridge metadata confirms the recording calibration reference SHA is `3a16eb9ce67d8451164f7038946d9753f317f49db8e96a686d5f16d36603925b`.

The read-only receipt at `within0_frozen_probe_v2/p01.json` has SHA-256 `2424e9fced17bfe20830cb8f0a71a368dc1f6f4a1ec4acc2b470118d40ed1b33`. Source manifest SHA is `264cb49adf604532daa93d1a2e25d4a9c15f78df4e9327c14780752636021ebb`. The comparisons cover P01 frozen emissions. All other folds and legacy r15c250 warm parity remain unverified. Held gold stayed closed and the probe produced no accuracy.
