# Within0 historical warm and common Sinc inputs

## Historical warm control

The original §257 jobs were `3103241_0` (FB raw), `3103241_3` (FB EA), and `3103241_1` (FBLight). All ran on RTX 2080 Ti cn114 with eight CPU and BLAS threads. The executable was `studies/14nov/ftrelabel/ftrel.py` with `SCHEME=within PSEED=0 KIND=v51b ARMS=s15,r15 CAPS=100,150,250`. The archived logs independently report `chain_v51b_within0.npz:384b097cdd4e`. The untouched strict-target table predates those jobs, and the owner checks its emitted decisions against the chain before consuming strict-prefix targets. The installed table agrees with 97.6271% of the emitted chain decisions. The [input manifest](within0_warm_inputs.sha256) identifies both complete digests and the restored fold states.

The diagnostic compares the loaded owner's target vector bit for bit to an independent strict-prefix reconstruction for every visit. It validates the first 20 unadapted outputs and makes sure a real AdamW update runs five times at each refit age in each of the s15 and r15 arms. The authenticated archive and live warm outputs remain distinct from the fresh targets below.

The original P01 smoke `3132542` exactly reproduced FB raw and FB EA warm vectors, all 170 full-stream optimizer steps and all 60 steps per prefix. FBLight stopped at a repeated-prefix posterior difference of 1.87746e−7 before its archive comparison. The follow-up preserves the original runtime for historical comparison and uses a separate deterministic process to test for session leakage. Both FBLight receipts gate all its remaining folds. The two exact-family smokes retain their original gates.

## Common fresh Sinc input and paired targets

CPU job `3132551` finished at 23:33:29 CET on 18 November. The historical P0r, pre-EM and gate reproduced the pinned within0 controls exactly. Replacing just Sinc seed 0 moved the real six-seed slot by up to 2.09765e−6 and the live P0r shadow by 1.60581e−7. Both arms used the identical source and original recording calibration. Independent one-sixth mixture arithmetic agreed within 1.11e−16.

Both arms remained unrouted after 2,515 finite Bio scores and all contrast and Bio prefix checks. Each alias contains 240 refits and 24,180 strict-past entries. An independent read verified the complete alias hashes, within0 delivery, age indices, emitted posteriors and exact finite masks. The EA replacement changed strict-past target scores by up to 0.308393.

| Read-only artifact under `within0_sinc_bridge_v1` | SHA-256 |
| --- | --- |
| `source.sha256` | `6071836abdd193a05a15cb0be054d17e5d2d20985db73b29a23161859256fef3` |
| `status.json` | `73786ae9962346680109c1b8d8c6aabe9ed194e7ab1a5a8971b88b9917a8b055` |
| `bridge.npz` | `0e5f4c93b1270c5b329e81253ca9efa8331f184d33e0b6a811ae300742dea0e2` |
| `historical/chain_v51b_within0.npz` | `03aa72fdc9aa0c0a623afab7a01900120c253ae5201d3d6a28afccb7f5ec8768` |
| `corrected/chain_v51b_within0.npz` | `3dc4eacdf928c81ef84ebf24d32344564fc99782614fff71e75cc501674f1066` |

The target seal leaves matched refits and terminal scoring disabled. No held gold was opened.
