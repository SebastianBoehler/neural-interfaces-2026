# F-C17 · Amplitude-conditioned fractional alignment

The native 32-member group aligns each current window with a single accumulated covariance and common source prior. F-C17 keeps each covariance intact. It divides previous windows fractionally between two histories and uses the current window's relative broadband amplitude to select their mixture. The effective history keeps the original age. The actual InputStream applies the original source prior and adds the current covariance at unit weight. [Preregistration](../../docs/LEDGER.md#f-c17--current-window-amplitude-selects-two-fractional-ea-histories)

The permuted control uses the same current weight and source prior. In each completed group of four, it shifts the stored weights by two entries. It uses the same partial group as the informative arm. Each completed group's membership multiset and per-age history masses agree exactly.

## Real smoke

A100 job `3133104` completed at 02:04:57 CET on 19 November. The standalone check first called the actual package InputStream on Scherer G visit 1 order 411. It reproduced the disabled outputs and verified exact prefixes after interleaving another session. Both active arms matched stock on the first window and each other through the first four deliveries. Per-age exposure error was zero.

| Standalone measurement | Value |
|---|---:|
| Membership span | 0.63026 |
| Largest normalized branch covariance separation | 0.42117 |
| Aligned windows differing by more than 1e-3 between arms | 104 / 120 |
| Largest aligned tensor difference | 4.67751 |

The full cap250 worker then reproduced all 120 archived labels and counters in stock and disabled. The same aligned tensor reached all 32 intended members. The active arms differed by up to 0.15753 in final posterior and each changed one final decision relative to stock. Each arm performed the original ten refits and 200 optimizer steps. Aligned 31-row prefixes matched exactly after interleaving. The maximum terminal prefix drift was 5.42e-9 among active arms. The smoke read no held gold.

## Full paired campaign

Galvani `3133114` has nine matched person tasks with at most four running at once. It compares stock, informative and permuted on fresh full orders 6411 and 7411. Scorer `3133115` checks every task, archived stock control and ordered delivery before sealing the complete bank. The people are reused external participants.

| Artifact in `ni26_friday_c17_fractional_ea_v1_20261119` | SHA-256 |
|---|---|
| `source.sha256` | `b198a6152e0828d1273eb047b435f337e55e14fc857b9e21be45c377ad1a06de` |
| `scoring.sha256` | `c3c80577a56b42483901fc6d3f02a7b3b397920bfce7925ec1e9829ad17a936b` |
| `smoke.json` | `6657eb90366df2efa053c4be5ad7ce131e649ee7bc973bddae07de3f6c5c286f` |
