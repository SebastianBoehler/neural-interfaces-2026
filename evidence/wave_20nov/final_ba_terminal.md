# Final `ba_terminal` package · 21 November 2026

**Result.** The archive passed fold provenance, zero-dose parity, and eager/graph label parity. Both RTX 2080 Ti projections exceeded 3,600 s, and the Scherer B-ret result triggered the registered flag. Qualification finished by 05:26:49 CET.

`ba_terminal` keeps the exact hard-only Team B nine25 mixer and blends 10% of the B6 MANAS A1 posterior at the final decision, after count filtering and CSP. It softmaxes CSP log scores only when CSP fused the current window. The external and public fit weights stay in their original precision.

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `final_stack_hard_safe_st_team_b_nine25_ba_terminal.zip` | 419,084,625 | `c50a62913360a7baeab978db19b778749308b171ce1d92c66343d3a8803371c5` |
| `ba_terminal_niv.zip` | 419,083,483 | `4143ac070dbedb4b4962ab734407caed4629b5607af49b5e547c8041ed65c682` |

The production archive is staged in local `runs/sealed/final_zips/`, Leonardo `cap250/runs/ba-terminal-21nov/zips/` and Galvani `ni26_ba_terminal_20261121/zips/`. The embedded production MANAS weights have SHA-256 `b7728c22de2827f7a8747145fb5f53e5c93fbaab3f903ddfbc59124661e3467b`. The B6 MANAS A1 donor ZIP is `4bd61c969408980e00eff3ac501ed53bef593eb0e3be04d0e20a4216ebd0fb87`. The hard+nine25 parent is `6e09953709b2338507deaac1ec5f6afa4a21eef52de7814f7c2ae77f42e86ee9`. The builder source is `6d8d0890`. It checked parent inventory and data payloads byte for byte. It added `team_a.pt` and changed generated `submission.py`. The production archive is 399.67 MiB. The qualification run uses a 450 MiB local allowance because the historical project allowance is 90 MiB. [Official contract](../../OFFICIAL_CONTRACT.md)

The growth folds reported +0.11 pp held-new [−0.14, +0.35] and +0.36 pp returning [+0.12, +0.60] versus hard+nine25 across 16 held-out delivery orders. These 80% intervals resample people, then orders. The returning contrast covers P04 and P10. J was +0.23 pp. The unverified 57/43 mix with P04/P10 at 20/80 was +0.18 pp. [Growth report](laneTB_grow.md)

## Qualification

Leonardo v3 replay 65006813 printed `OK`. The CPU zero check 65006724 reproduced the hard+nine25 mixer, route outputs, terminal scores and labels **bitwise** on P01/1 (190 windows) and P04/2 (189) at shuffle 71120. Team A loads no weights at zero dose.

| Exact archive graph vs eager | Recording order | Shuffle 71000 |
|---|---:|---:|
| Leonardo A100, 65006725 tasks 0–3 | 0 / 2,655 changed | 0 / 2,655 changed |
| Galvani RTX 2080 Ti, 3146130 tasks 0–3 | 0 / 2,655 changed | 0 / 2,655 changed |

Each order includes all 14 sessions through the official Solver and `fresh_model`. Team A ran on every window. Both smokes exercised exact interleaved prefix reset, three injected load failures, and one live failure followed by valid labels in the same dead session. The load test checks MANAS against an independent eager CPU reference and then against eager on the actual GPU. The graph, input and output buffers, and adaptation state are session-local. [Dispatch](DISPATCH_21NOV.md)

The all-data sessions did not activate CSP fusion. On the pinned Scherer G visit 2 stream, 30 of 120 windows used CSP fusion. Graph and eager matched all pinned labels. The extra full-chain score check 3146188 exceeded its `2e-5` tolerance. Jobs 3146189 and 3146192 then alternated which path ran first in a fresh process. Both detected up to `1.07e-4` in the upstream mixer between the first and later streams. Later graph/eager pairs differed by at most `2.06e-8` across the traced scores. MANAS posteriors were identical in every repeat, and all terminal labels matched. These diagnostics exercise the actual normalization and blend on the fused branch.

### Official cap250 warm-up pair

Both `friday/qualify_package.py pair` jobs completed all four replays. Every replay preserved extracted files, printed four zero-error v6 streams and matched its repeat on every predictive metric. Cell-balanced accuracy was **100.0000%** for `ba_terminal` and **99.4579%** for cap250 on both GPUs. Both jobs returned `replay_pass: false` because the +0.5421 pp gap exceeded the symmetric ±0.5000 pp gate. The warm-up replay has 755 public windows from four sessions. Its candidate uses all-public fit weights.

| GPU | Job | cap250 wall, s | Candidate wall, s |
|---|---:|---:|---:|
| A100-SXM-64GB, native CUDA 12.6 Torch | 65006727 | 295.4 / 125.2 | 154.7 / 145.3 |
| RTX 2080 Ti, worker SIF | 3146132 | 124.4 / 127.2 | 183.4 / 171.5 |

### Additive runtime

Standalone paired MANAS timing on P01/1 and P04/2 (379 shuffled windows) includes graph capture and gives maximum posterior difference **0.0** on both GPUs. The CUDA graph reduced MANAS from 11.97 to 4.68 ms/window on A100 (61%) and from 19.39 to 14.82 ms/window on RTX 2080 Ti (24%). The archive uses the graph on both GPUs.

End-to-end timing counterbalanced cap250, hard+nine25, and both MANAS paths on six shuffled sessions (1,131 measured windows, plus one discarded warm-up session). It includes graph capture, model restore between sessions, and all adaptation. Add the per-window delta over cap250 for 20,083 windows to the cap250 sealed receipt of 2,554.9166 s, then add any increase in model load time. The official 3,600 s budget is on RTX PRO 4500. These estimates are hardware proxies.

| GPU and job | cap250, ms/window | Hard+nine25 add, ms/window | Graph MANAS add vs parent, ms/window | Eager projection | Graph projection |
|---|---:|---:|---:|---:|---:|
| A100 65006726 | 113.08 | +20.56 | +6.09 | 3,235.4 s | **3,090.6 s** |
| RTX 2080 Ti 3146131, cn111 | 133.04 | +55.51 | +17.18 | 4,121.5 s | **4,016.6 s** |
| RTX 2080 Ti 3146187, cn114 | 131.45 | +43.03 | +15.96 | 3,900.1 s | **3,741.3 s** |

Both RTX 2080 Ti placements miss the 3,600 s proxy by 141–417 s. Most between-node variation came from the hard+nine25 increment. The MANAS graph adds 15.96–17.18 ms/window over that parent on RTX 2080 Ti. The repeat ran on a second shared node because Slurm forecast the exclusive allocation after the cutoff.

## Team A provenance

The 12 fold manifests and all 96 partner receipts from the two Lane TB-grow confirmation blocks match their source and checkpoint hashes. Recomputing the seed-0 sampling draws from the signed public export reproduced the source roster below. Every selected public window appeared in a training batch. All people other than P04 and P10 contributed visit 1 only.

| Fold | People in the public fit | P04 visits | P10 visits | Windows | Weight SHA prefix |
|---|---|---|---|---:|---|
| `new_P01` | P02, P03, P04, P05, P06, P07, P08, P09, P10 | 1, 2, 3 | 1, 2, 3 | 2465 | `e8437dcb602d` |
| `new_P02` | P01, P03, P04, P05, P06, P07, P08, P09, P10 | 1, 2, 3 | 1, 2, 3 | 2466 | `031724531ac0` |
| `new_P03` | P01, P02, P04, P05, P06, P07, P08, P09, P10 | 1, 2, 3 | 1, 2, 3 | 2466 | `e7a3a819168f` |
| `new_P04` | P01, P02, P03, P05, P06, P07, P08, P09, P10 | — | 1, 2, 3 | 2089 | `935c50bc7d80` |
| `new_P05` | P01, P02, P03, P04, P06, P07, P08, P09, P10 | 1, 2, 3 | 1, 2, 3 | 2465 | `795a57bdd51b` |
| `new_P06` | P01, P02, P03, P04, P05, P07, P08, P09, P10 | 1, 2, 3 | 1, 2, 3 | 2468 | `ad4a261d564a` |
| `new_P07` | P01, P02, P03, P04, P05, P06, P08, P09, P10 | 1, 2, 3 | 1, 2, 3 | 2466 | `be812c69fc5f` |
| `new_P08` | P01, P02, P03, P04, P05, P06, P07, P09, P10 | 1, 2, 3 | 1, 2, 3 | 2466 | `718934f5b814` |
| `new_P09` | P01, P02, P03, P04, P05, P06, P07, P08, P10 | 1, 2, 3 | 1, 2, 3 | 2466 | `225b946f4ace` |
| `new_P10` | P01, P02, P03, P04, P05, P06, P07, P08, P09 | 1, 2, 3 | — | 2078 | `d5edda3dee7e` |
| `ret_P04` | P01, P02, P03, P04, P05, P06, P07, P08, P09, P10 | 1 | 1, 2, 3 | 2278 | `09ee6fed368c` |
| `ret_P10` | P01, P02, P03, P04, P05, P06, P07, P08, P09, P10 | 1, 2, 3 | 1 | 2270 | `c8e961510795` |

Each returning-person fold admits only the held person's visit 1. The other returning person's three visits remain in the public fit. Both production banks use all 2,655 windows. All 14 audited banks reproduced their stored statistics with maximum absolute difference **0.0**. The audit rebuilt the equal-session covariance, visit-1 identity galleries, identity scale and radius, enrollment flags, general classifier, personal heads, feature mean, and inverse standard deviation. It used the original frozen encoder and reconstructed the training sampler's seed-0 state before recomputing the fitted feature bank. During inference, the covariance accumulator, recognition, alignment, and prefix adaptation all start empty.

The 12 scored folds share a single 12-corpus external transfer (`4d0851442bc6035795f8d4967cf2c0b59f8497d881740554a9c4757daed9dee0`). The NI-V transfer (`ec6c1890ea3be4b1777642674a1a60261ed23d52b9d90da39e5a37ea7bfdb4ff`) uses 11 corpora. The audit verified all 36 and 35 shard hashes respectively and reconstructed the person-by-session counts actually sampled for external updates. These counts are in `out/audit_external.json` (SHA-256 `2d137c06517b071bb015624b6490f17bc397e9d59e7e478792c7f51d6882ce8f`) under the Leonardo run directory. The production transfer includes Scherer. The NI-V transfer omits it. Neither manifest listed PROTEUS IDs among its external source exports. The upstream MANAS corpus disclosure is the limit of the pretrained checkpoint audit.

## NI-V

The Scherer-free B6 A1 candidate uses an existing separately fitted transfer and production bank. B-new scores both Scherer visits. B-ret applies the byte-identical hard+nine25 enrollment from visit 1 on both arms and scores visit 2, with MANAS frozen. The pretrained MANAS release discloses clinical sources, TUH EEG and I-CARE. [Pinned MANAS model card](https://huggingface.co/MannasAI/pratibha-2/blob/863edc940d82a02744776a7820641bf1bc32310b/README.md)

Galvani 3146137 collected all 36 B-new chunks and all nine B-ret chunks. Both delivery schemes use eight pinned orders, 71000–71007. Differences are percentage points with paired people-and-orders 80% intervals.

| Scherer cohort | Delivery | Hard+nine25 BA | With MANAS BA | Δ pp [80%] |
|---|---|---:|---:|---:|
| B-new | Full session | 48.73% | 48.70% | −0.03 [−0.48, +0.42] |
| B-new | Within run | 49.11% | 49.29% | +0.18 [−0.10, +0.45] |
| B-ret | Full session | 56.67% | 56.15% | **−0.52 [−1.01, −0.02]** |
| B-ret | Within run | 58.00% | 58.16% | +0.16 [−0.43, +0.73] |

B-new stayed inside the registered limits. B-ret full-session triggered the registered flag because the upper 80% bound is below zero. The unverified 57/43 mix gives −0.24 pp [−0.57, +0.10]. The Scherer export interpolates 22 of 41 channels. Report `niv/out/report_ba_vs_hss_n25.json` SHA-256 `906a9e181d34b2f60058947aadc4df36c1c0e8fb65a41ce491770f422e570e36`.

No archive was uploaded.
