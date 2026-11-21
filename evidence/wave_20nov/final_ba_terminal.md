# Final `ba_terminal` package · 21 November 2026

`ba_terminal` keeps the exact hard-only Team B nine25 mixer and blends 10% of the B6 MANAS A1 posterior at the final decision, after count filtering and CSP. It softmaxes CSP log scores only when CSP fused the current window. The external and public fit weights stay in their original precision.

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `final_stack_hard_safe_st_team_b_nine25_ba_terminal.zip` | 419,084,625 | `c50a62913360a7baeab978db19b778749308b171ce1d92c66343d3a8803371c5` |
| `ba_terminal_niv.zip` | 419,083,483 | `4143ac070dbedb4b4962ab734407caed4629b5607af49b5e547c8041ed65c682` |

The production archive is staged in local `runs/sealed/final_zips/`, Leonardo `cap250/runs/ba-terminal-21nov/zips/` and Galvani `ni26_ba_terminal_20261121/zips/`. The B6 MANAS A1 donor ZIP is `4bd61c969408980e00eff3ac501ed53bef593eb0e3be04d0e20a4216ebd0fb87`. The hard+nine25 parent is `6e09953709b2338507deaac1ec5f6afa4a21eef52de7814f7c2ae77f42e86ee9`. The builder checked parent inventory and data payloads byte for byte. It added `team_a.pt` and changed generated `submission.py`. The production archive is 399.67 MiB. The qualification run uses a 450 MiB local allowance because the historical project allowance is 90 MiB. [Official contract](../../OFFICIAL_CONTRACT.md)

The growth folds reported +0.11 pp held-new [−0.14, +0.35] and +0.36 pp returning [+0.12, +0.60] versus hard+nine25 across 16 held-out delivery orders. These 80% intervals resample people, then orders. The returning contrast covers P04 and P10. [Growth report](laneTB_grow.md)

## Qualification

The A100 and RTX 2080 Ti real-window smokes passed. Each tested zero-weight prefixes, graph and eager decisions in both deliveries, exact interleaved prefix reset, three injected load failures, and one live failure followed by valid labels in the same dead session. The load test verifies the signed MANAS model against an independent eager CPU reference, then checks the graph against eager on the actual GPU. The graph, its input and output buffers, and adaptation state are session-local. Full qualification receipts are pending. [Dispatch](DISPATCH_21NOV.md)

## Team A provenance

The provenance audit reads the training manifests and weights, recomputes public rows from the signed export, checks the shared external transfer and the scored replay receipts, and rebuilds the stored source-only bank from the fitted model. The P01 smoke passed. The other twelve package audits and the external-lineage audit are pending.

## NI-V

The Scherer-free B6 A1 candidate uses an existing separately fitted transfer and production bank. B-new scores both Scherer visits. B-ret applies the byte-identical hard+nine25 enrollment from visit 1 on both arms and scores visit 2, with MANAS frozen. The pretrained MANAS release discloses clinical sources, TUH EEG and I-CARE. [Pinned MANAS model card](https://huggingface.co/MannasAI/pratibha-2/blob/863edc940d82a02744776a7820641bf1bc32310b/README.md)

No archive was uploaded.
