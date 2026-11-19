# F-C21gct · Share covariance and stop the alternate on recognition

The [combined class](combined_selector.py) subclasses the measured `SharedCovSelector` and sets `retire_on_recognition=True`. On each pre-recognition row it gives the alternate CSP an owned copy of the current original stock covariance. The two native tails keep their own probability histories, fitting rows, targets, models and clocks. On the actual recognition frame, the selector emits stock and stops both the alternate mixer and alternate tail. A fresh session gets a fresh selector. [Preregistered behavior and timing](../../../docs/LEDGER.md#f-c21gct--combine-shared-covariance-and-sticky-recognition-retirement)

The release pins F-C21gcv source SHA-256 `40ce7409bfc3f86ba53729b7e84c3b2ae9dd99f27392c51bf419abcfdd1c498d` and F-C21gt source SHA-256 `714add28327fb564cf2bb7b00b7f5420b58402d15ab74a95fc9b9ca5668de96c`. The exact covariance and live selector file digests are `24016a105a5efbdbd424738c2cf46b3126d6538f96fa42af1844d21e75e9fca9` and `6af1cb7229dae99834e15a1564b931d9f8610503722a510b10d245119dea3b12`.

| Galvani job | Scope |
| --- | --- |
| `3134548` | Conditional native P10 visit 2, late age-28 recognition, source-archive parity and reset. |
| `3134549` | Real RTX Scherer G and RTX P10 visit 3 full-dual comparisons on identical live inputs. |
| `3134550` | Warm paired RTX timing after both exact conformance jobs. |

Immutable release `ni26_friday_c21gct_combined_v2_20261119`, source SHA-256 `5c09a99069580b4ac894f4dbae7f5b4fea8349b0cef2bd380a289bbe861b49a6`.

The first release stopped on an observer assertion. Original stock CausalPers clears its covariance and target arrays at recognition, while the retired alternate retains its completed prefix. The corrected observer checks exact covariance values and ownership on each live share. It also checks the native stock flush on recognized sessions and the two retained whole covariance banks on unrecognized G. Both runtime parent modules and the selector subclass stayed byte-identical. V1 timing `3134544` was canceled.

## Signed conformance and runtime

The combined selector preserved every emitted vector and the complete pre-recognition alternate state on identical live inputs:

| Fixture | Covariances verified independently | FBL2 emissions eligible | Actual retirement frame |
| --- | ---: | ---: | --- |
| RTX Scherer G visit 1 | 120 | 100 | No recognition |
| RTX PROTEUS P10 visit 3 | 10 | 0 | Age 10 |
| Conditional native CPU P10 visit 2 | 28 | 8 | Age 28 |

G performed five alternate CSP refits and 30 fusions. P10 visit 2 retained the true warm ages 20–27 and 20 optimizer steps. The same-input prefix and first 20 independent RTX frames were bitwise. The distinct G fresh-prefix comparisons preserved all decisions, with maximum difference 1.106e−8 below measured original stock repeat 7.501e−8. Every verified covariance was bitwise and independently owned.

| Warm paired RTX fixture | Exact stock (s) | Combined selector (s) | Combined / stock mean | Worst pairing | Verdict |
| --- | --- | --- | ---: | ---: | --- |
| Scherer G visit 1, 120 windows | 31.756 / 30.828 | 34.854 / 35.493 | 1.12404× | 1.15130× | Missed both limits |
| PROTEUS P10 visit 3, 31 windows | 5.895 / 5.861 | 6.099 / 6.141 | 1.04121× | 1.04782× | Passed both limits |

The fixed boundaries were mean ≤1.12× and worst ≤1.15× on each fixture. The joint criterion failed. G missed its two bounds by 0.00404× and 0.00130×. P10's matched full-dual times were 7.058 / 7.046 seconds. The combined selector used 0.86785× their mean. G's combined per-window p95 was 0.791 / 0.779 seconds versus stock 0.740 / 0.739. P10's was 0.212 / 0.214 versus 0.192 / 0.191. The earlier standalone covariance-sharing timing remains a separate measurement. These are runtime and conformance results on reused RTX proxy fixtures.

| Receipt | Galvani job | SHA-256 |
| --- | --- | --- |
| `delayed.json` | `3134548` | `1bb1a7b47f393e40c3a827e7fcfcd8ffdacb19dc165e1dd9a0e3e3c22444b690` |
| `smoke.json` | `3134549` | `3be501d95012ca749d6a151e2aa1a5b69843273f330c6d4485f9a52857913511` |
| `timing_0.json` | `3134550_0` | `c9a87c94756e780405d69eb05938603f8d7fabb20f4ae1abb8c1b07f35f415bd` |
| `timing_1.json` | `3134550_1` | `f99c27adf07f61ff56f0c88fdd2a929a1b28dbbd751333ca7a9fe716c553425f` |

The final timing block completed at 07:26:09 CET on 19 November.
