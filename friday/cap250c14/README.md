# F-C14 · Raw cognitive odds × directional alignment

F-C13's raw-only S1C replacement helped both cognitive classes and hurt motor imagery on the same Scherer people. F-C14 tests whether that evidence still helps when the cap250 raw family's motor allocation is preserved. It crosses the raw-family substitution with F-C12's past-only cap on Euclidean alignment covariance. [Preregistration](../../docs/LEDGER.md#f-c14--cross-raw-cognitive-odds-with-past-only-directional-alignment)

The worker calls the original cap250 model for each new window. The odds adapter uses the one frozen S1C seed-0 posterior to split the incumbent cognitive mass between subtraction and word generation. It applies that split to the six native frozen raw FBCNet posteriors and to the live raw family contribution after the original refit correction. The post-refit arm still uses the actual refit's motor/cognitive allocation. Both checks preserve the current raw member/family total and motor mass. Earlier output changes can affect later refits and other session state. The directional adapter is the unmodified F-C12 `history.py`.

## Real worker smoke

Galvani A100 job `3132597` completed in 4m47s at 23:49:05 CET on 18 November. Stock and identity reproduced all 120 archived G visit-1 decisions and activity counters. The independent raw member posteriors differed by at most 3.013e-6. The consuming identity hook passed its exact forwarding checks. Recording-order identity and all three active 31-window prefixes passed bit for bit after interleaving another session.

| Arm | Final decisions changed vs stock | Covariance cap ages | Later inputs changed | Odds shifts above 0.01 | Live raw refit contributions projected |
|---|---:|---|---:|---:|---:|
| Directional | 1 | 48, 86, 90 | 71 | — | — |
| Cognitive | 5 | — | 0 | 118 / 120 | 100 |
| Combined | 4 | 48, 86, 90 | 71 | 118 / 120 | 100 |

All arms completed the original 10 refits and 200 steps. The first changed direction affected only later windows. Each active odds arm used all 120 actual actor calls. The smoke used no gold and gives no accuracy or runtime advancement decision.

## Full paired campaign

Galvani `3132604` runs four matched arms on both visits of all nine Scherer people, on full delivery seeds 2411 and 3411. Array concurrency is four. The conditional scorer `3132605` authenticates all worker banks, verifies archived stock controls and seals predictions before scoring. The people were already used to identify the class signal. This campaign tests new delivery orders on those same people.

Artifacts live in `/mnt/lustre/work/gehler/gfh527/ni26_friday_c14_factorial_v1_20261118/`.

| Receipt | SHA-256 |
|---|---|
| `source.sha256` | `f344d251b5fa2390e60e472515c03f4f391548a3a73baba6d104f16d2c5be49b` |
| `scoring.sha256` | `2a56f8296fa95393a810d08996810b7ac85e864b6cb7b41a3ede64df305b69ff` |
| `smoke.json` | `05a7b0a2b45ade3cb9b4398c02aeb56dd1835173a5ad8e1aad0e629e6b1e7621` |
