# F-C20 · Replace the five frozen outer raw FBC members

19 November 2026, 04:41:04 CET · FULL-SYSTEM DEV · Corrected D-EAwcf within0 and D-EAfcf full0 · Ten held-new PROTEUS people in both orders

Both waveform substitutions gained against stock in both deliveries. The hard pupil gained **+0.3048 pp** within0 and **+0.3314 pp** full0. The jitter pupil gained **+0.1932 pp** and **+0.4012 pp**. Jitter minus the matched hard pupil reversed sign, with both intervals spanning zero. The jitter-specific advancement condition failed. [Preregistration](../../docs/LEDGER.md#f-c20--replace-the-five-frozen-outer-raw-fbc-members-with-source-jitter-pupils)

| Whole-system comparison | Within0, paired 80% | Full0, paired 80% |
|---|---:|---:|
| Stock balanced accuracy | 78.3241% | 73.7390% |
| Hard two-pupil substitution | 78.6289% | 74.0703% |
| Jitter two-pupil substitution | 78.5173% | 74.1401% |
| Hard minus stock, pp | +0.3048 [−0.0087, +0.6407] | +0.3314 [+0.2144, +0.4508] |
| Jitter minus stock, pp | +0.1932 [−0.1345, +0.5576] | +0.4012 [+0.1259, +0.6515] |
| Jitter minus hard, pp | −0.1116 [−0.1976, +0.0029] | +0.0698 [−0.1597, +0.2571] |

Each delivery uses the same ten people and 28 official cells. The paired interval resamples people 4,000 times within each delivery. P04's hard/jitter changes were +0.8908/+0.7112 pp within0 and +0.1620/−0.0112 pp full0. P10's changed −0.3370/−0.6842 pp and +0.5106/+1.1949 pp. The two pupils' reuse of these public fits and the inherited auxiliaries limit the claim.

| Person | Within0 hard fixes / losses | Within0 jitter fixes / losses | Full0 hard fixes / losses | Full0 jitter fixes / losses |
|---|---:|---:|---:|---:|
| P01 | 4 / 2 | 3 / 2 | 0 / 0 | 1 / 0 |
| P02 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| P03 | 0 / 0 | 0 / 0 | 1 / 0 | 0 / 0 |
| P04 | 7 / 2 | 7 / 3 | 7 / 6 | 5 / 5 |
| P05 | 3 / 2 | 3 / 2 | 4 / 4 | 4 / 4 |
| P06 | 0 / 1 | 1 / 1 | 1 / 1 | 2 / 1 |
| P07 | 0 / 1 | 0 / 1 | 3 / 1 | 2 / 0 |
| P08 | 4 / 0 | 4 / 0 | 2 / 1 | 1 / 1 |
| P09 | 0 / 0 | 0 / 0 | 1 / 0 | 1 / 1 |
| P10 | 0 / 2 | 1 / 5 | 6 / 3 | 7 / 0 |
| **All** | **18 / 10** | **19 / 14** | **25 / 16** | **23 / 12** |

Hard motor/subtraction/word net counts were 0/+7/+1 within0 and +5/−1/+5 full0. Jitter was 0/+5/0 and +6/+2/+3. Both replacement effects include complete native CSP horizon, tail and threshold behavior.

## Verification

On both deliveries, the actual stock and native-mean identity agreed on all raw, count-filter and terminal scores. The protected P0r gate, LPC inputs and seed-0 warm deltas stayed exact. Each changed arm reran the original rebalancing, EM/TSEM, count and CSP. All arms completed 126 CSP refits and 1,395 fusions per delivery. CSP used each branch's own strictly past raw pseudo-targets. The original count filter changed 2,005 rows within0 and 308 full0. The changed raw winner counts were hard/jitter 28/26 within0 and 26/24 full0.

Both pupil seeds reproduced their corresponding signed F17 decisions on both orders, with independent prefix and interleaved resets. The maximum normalized raw-family movement was hard/jitter 0.4085/0.4180 within0 and 0.4693/0.4676 full0. Original normalization clipped one or more class values on 247 stock-identity rows within0 and 274 full0. The interventions clipped 723/730 and 756/759. The historical branch preserves its recording-bag reconciliation and native warm addition. The native package uses its own forward-mix arithmetic and needs a separately qualified comparison.

Galvani ran both complete instrument replays in about 4m08 on CPU2. This excludes per-pupil probability-bank production and does not measure whole-package deployment time. The first source smoke `3133827` failed before source probabilities were emitted because the evaluator helper expected an explicit no-grad zone. The corrected pupil smokes `3133840_0/1` passed in 14/13 seconds and the full `3133841` array completed all twenty authenticated folds. The integrated smoke replayed 190 treated P01 windows and a 189-window untouched count-latched peer, with interleaved 120-window prefixes. The scorer opened held labels only after both complete banks were sealed.

Galvani releases `ni26_friday_causal_rebuild_20261118/fc20_pupils_v2` and `ni26_friday_causal_rebuild_20261118/fc20_outer_v1`. Source `471d78ad`.

| Artifact | Job | SHA-256 |
|---|---|---|
| `fc20_pupils_v2/source.sha256` | — | `5925163a6331fc062a51f80f8262c05f7c12f069cfdba86d9e2cddf1f57d349c` |
| `consumer.sha256` | — | `1319f8c515cba34eb9c7347c7b94a584ae2f33b3225c01ecfb623c9f60506f5b` |
| `scoring.sha256` | — | `1efba993b0c9ea920f44b0f56c75e4cdc7436d31b2a828717484417bc876e568` |
| `smoke.json` | 3133843 | `0f68f8a13327e5c233586e5e19f54e90ea55942530f26fdabef9abccb8dd4401` |
| `within0.json` | 3133845 | `a2249de5725ed36472ef6126ce1195b502fe1f3b9a47bef3124edf02bb81ebae` |
| `within0.npz` | 3133845 | `73f14c4082174008aa43ab9b7447a48a85973eb12a2113b8723520177de9f3a5` |
| `full0.json` | 3133844 | `1e3109313d14fb2d1a429762b1f2ef544bed3bb38759e8b1bbd0d0f12f8ac683` |
| `full0.npz` | 3133844 | `1a5f0c2d9cb81e38cd8f11608440c237c032ec413efee4ad9675a9f7507b9a57` |
| `score.json` | 3133870 | `1100016fe85b769282b01d7df6bb374e5590f1707ca8375e2aa1e84ba7b4ae30` |
