# F-C20e · Native RTX qualification stopped at repeat variation

19 November 2026 · Exact cap250 worker on Galvani RTX 2080 Ti · Label-blind qualification

Seven of nine independent person banks passed. The E and L banks exceeded the fixed **5e−5** tolerance for warm upstream probabilities. The score job canceled before gold access. The complete independent accuracy and runtime comparisons therefore remain unavailable. The later six-run diagnostic reproduced the E-scale drift between two untouched stock runs. [Preregistration](../../docs/LEDGER.md#f-c20e--native-outer-raw-replacement-on-two-new-scherer-deliveries)

| Independent fixed worker | Result |
|---|---|
| G order 0, 120 windows | Exact archived stock and native-mean identity decisions/work. Same-input outer identity exact. Warm stock/identity trace max 3.17e−5, stock/hard max 6.72e−6. Both real pupil exports agreed with CPU on all prefix decisions, with max probability error 6.01e−6. |
| G actual warm work | Both stock and Hard performed 10 refits and 200 optimizer updates. The 650 prior input rows, loss labels and weights matched exactly. RouteB consumed the replaced outer mixture on all 120 Hard windows, including 100 warm ones. |
| P10 visit 3, 31 public windows | Exact upstream probabilities across stock and Hard. Actual recognition at age 10 and CSP stopped at the same age. This is a conformance fixture because both pupils trained on P10. |
| E visit 1 order 9411 | Independent stock/Hard maximum 2.8394e−4 in the warm Sinc member. Inner mix, shadow, v5 output and pseudo-label probabilities were exact. Original guard stopped evaluation before an E person bank. |
| L visit 1 order 8411 | Independent maximum 5.9095e−5 in the warm FBLight member. Original guard stopped evaluation before an L person bank. |

The G smoke's same-allocation warm observation was **28.72 s stock and 31.05 s Hard**, or **1.081×**. The cold stock took 71.95 s. This single smoke does not qualify the registered median/worst complete-worker runtime.

## What caused the failed comparison

On E visit 1 order 9411, three pure stock repeats in job `3133946` showed bitwise Sinc outputs and at most 9.96e−7 drift in another warm member, with zero label changes. This did not meet the prespecified repeat-drift forecast.

The fixed stock / compute-only / Hard / Hard / compute-only / stock sequence in job `3133950` showed a **2.8127e−4 Sinc difference between its two untouched stock runs**. This first appeared at age 20, exceeded the bound at age 44 and peaked at age 112 after the age-110 refit. The compute-only and Hard runs differed by at most **1.60e−7** in Sinc from the final stock. Every comparison kept exact prior loss features, labels, weights and optimizer counters. The two stock runs' terminal decisions were identical. Both compute-only runs reproduced those decisions with unmodified outer inputs. Both Hard repeats made one different decision relative to stock. These are blind decision differences.

Both pupils executed on the default CUDA stream. The observed execution flags and cuFFT plan count stayed unchanged across sampled pupil calls. The sampled pupil and FT tensor storage was disjoint, and mixer arguments were unchanged after the native call. The diagnostic establishes native repeat variability of the observed order of magnitude. It does not isolate the numerical source, establish pupil compute as the cause, or waive the original threshold.

## Signed evidence

RTX v4 release `ni26_friday_c20e_native_v4_20261119`; warm correction/source `3b3575d4`, recognized anchor fix `fc9356a0`. Snapshots v1–v3 retained earlier failures without scoring. V1 caught an export-hash transcription error. V2 stopped on cold-versus-warm upstream difference before treatment. V3 passed the active G treatment but stopped on the signed P10 anchor schema. V4 checked P10 rows from the separate signed RTX F-C5r receipt.

| Receipt | Job | SHA-256 |
|---|---|---|
| V4 source manifest | — | `3d56e3f24611b220fa017de8a49728c4ddc619eb65194c0150ee8b165bb9e436` |
| V4 smoke | 3133920 | `06c4d1f1bf0e79cff25d2194cf4ced0dfe0b598386609419a579a23360177247` |
| Full independent array | 3133921 | Seven read-only `person_*.json` receipts. E task 3 and L task 8 failed. |
| Independent scorer | 3133922 | Canceled by dependency. |
| `ni26_friday_c20e_stock_repeat_v1_20261119/stock_repeat.json` | 3133946 | `7bb90e4027bb45ecc99c0375fa492a2552c9c8ff885bf8648c695bd3baed8a9e` |
| `ni26_friday_c20e_compute_repeat_v1_20261119/compute_repeat.json` | 3133950 | `a85859d553b9009399429d69434a783b5d9d10b313941e936d943b1d9e7510f0` |
