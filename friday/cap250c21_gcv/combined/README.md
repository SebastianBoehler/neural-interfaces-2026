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
