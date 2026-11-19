# F-C21gpackD · Diagnose independent RTX package drift

The original independent G package comparison stopped at age 106 on a selected posterior difference of 5.84635e−5. The subsequent exact-reset check stopped on recognized P10 at 4.00819e−8. Both original vetoes are preserved. This diagnostic uses their original stock ZIP, deployed experimental ZIP, clean RTX 2080 Ti image, pinned fixture bytes and original archive reference. It uses the immutable prior test helpers and a separate harness. No task labels are read. [Preregistration](../../docs/LEDGER.md#f-c21gpackd--independent-stock-variation-and-shipped-same-input-reset-diagnostics)

The two stock-to-stock tasks run the unchanged stock Solver with a read-only stage observer. Independent full-dual references run afterward on each stock tape. The separate shipped tasks compare each actual archive output with independent downstream paths on its own live inputs. They check each reset against its own same-input path, and record numeric variation between independently calculated reset vectors.

| Galvani job | Scope |
| --- | --- |
| `3134870_0` | Original stock A/A and post-peer reset, recognized P10 visit 3. |
| `3134870_1` | Original stock A/A and post-peer reset, unrecognized Scherer G visit 1 order 0. |
| `3134870_2` | Shipped same-live and interleaved resets on P10. |
| `3134870_3` | Shipped same-live and interleaved resets on G. |

Immutable release `ni26_friday_c21gpack_drift_v1_20261119`, source SHA-256 `6249d01369c47e4498b5fc1cac57ca3af5ce8bb2b68dfc1165698a03a37406ba`. Stock ZIP `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`. Shipped trial ZIP `cf4a8deee46a25e59c2bb664bb0761254a6368467ac3eee5dff6c0bb572e447f`.
