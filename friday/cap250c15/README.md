# F-C15 · Robust outer rebalancing

The cap250 ablations in §400/400b first reversed a member's gain at the rebalancer, including when two arms used the same native multiplier. F-C15 changes the estimate of the past class baseline at just the emitted RouteB tail. The native shadow and routed gates, and the source of fine-tuning labels, still use the original rebalancers. [Preregistration](../../docs/LEDGER.md#f-c15--bound-the-past-class-estimate-in-the-actual-outer-rebalancer)

The robust arm caps each past probability vector's centered log norm at log(4), then averages 30 or fewer delivered rows with five neutral pseudo-rows. The scalar control uses the direction of the stock correction. Both use the smaller of the robust and stock norms at each delivery and the original exponent of 1.15. Only the current correction changes. The original rebalancer records the unmodified input in its session history. The actual outer SessionEM consumes the modified posterior.

## Matched smoke

Galvani `3132751` completed at 00:37:25 CET on 19 November. Scherer G visit 1 order 411 stock and identity reproduced all 120 archived decisions and counters. Identity was bit-exact at the hook and differed from independent stock by at most 2.49e-7 posterior. All truncated and interleaved prefixes matched decisions, with maximum posterior error 7.29e-7.

The robust correction differed from scalar by more than 0.01 on 78 windows. The two norms agreed within 5.56e-17. The largest robust output movement versus either stock or scalar at the rebalancer was 0.01151. Scalar and robust changed three and six final decisions. The two treatments preserved all 10 native refits, 200 optimizer steps, exact refit EEG buffer hashes and warm member decisions. The maximum protected outer input drift was 1.013e-5. Independent per-member warm numerical drift reached 1.619e-4 across the active arms.

An additional 31-window public P10 visit-2 replay started at global row 1238. Stock reproduced the original decisions. Each arm latched at age 10 and ran the real own-space stage on the subsequent 21 windows, with zero refits. The public subject appears in the pinned package's source. These are label-blind controls and activity tests.

The original smoke `3132722` stopped on the implementation's extra 2e-5 member-by-member warm tolerance. Stock and identity matched archived decisions. Their independent single-member warm posteriors drifted by up to 2.511e-4, while the consumed outer mix drift stayed within 2e-5. The registered retry retains the aggregate threshold, verifies exact effective refit inputs and decisions, and reports independently measured member drift.

## Full comparison

Galvani `3132752` depends on `3132751` and can run four people at once. It measures the three full arms with both visits of each of nine Scherer people on fresh delivery seeds 4411 and 5411. Scorer `3132753` verifies all nine banks before opening gold. The new orders reuse the people from earlier external work.

| Artifact in `ni26_friday_c15_rebalance_v2_20261119` | SHA-256 |
|---|---|
| `source.sha256` | `33a3557c9aba6f462923506e8f625f4062c2034a69ffb71ed412c13b900de9c4` |
| `scoring.sha256` | `725cfc77a68f15d0510d0d1b43ba640717b0414291522f1a54471e4523df8790` |
| `smoke.json` | `f87b1760dcd2aa9f08d1b1f558ded6deae9ec9a911b0ab0a6a36555c99513267` |
