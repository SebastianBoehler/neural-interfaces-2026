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

## Signed observations

All four independent jobs completed by 08:45:03 CET on 19 November. Original stock decisions, work and same-input native comparisons were exact. The shipped artifact matched stock, retained alternate and selected vectors on all 151 current inputs and on each of the four reset inputs. Both reset paths preserved the first class and native clocks on each fixture.

| Fixture | Untouched stock independent repeat and reset | Independent shipped reset delta |
| --- | --- | ---: |
| Recognized P10 | All checked repeat and reset stages bitwise | 4.00819e−8 |
| Unrecognized G | Repeat terminal max 9.39409e−6. Reset stages bitwise. | 3.28240e−8 |

Both shipped reset styles produced the same respective deltas, which were already present in the stock vector emitted at age zero. That leaves the reset's independent bitwise verdict failed on both fixtures. P10's delta reproduced the earlier observed value. Stock A/A and the shipped test ran in separate allocations, on cn122 and cn123 for P10. These observations locate the reset difference in the shipped stock path and leave its root cause unresolved.

Untouched stock G first moved at warm age 20. Among the captured protected stages, the immutable member stream, core, inner mix, labels shadow and three other warm families were bitwise. Warm FBLight moved up to 2.35941e−5 at age 87. Native stock terminal movement was 2.02953e−6 at age 106. Driving an independent full dual tail with each actual stock tape gave a selected maximum of 1.76811e−5 at age 116 and 4.28166e−6 at age 106. All classes and work counts matched. The A/A probe established base variation and fell short of the prior 5.84635e−5 selected failure at age 106. The independent G archive veto and earlier exact-reset veto still stand.

| Receipt | Job | SHA-256 |
| --- | --- | --- |
| `aa_recognized.json` | `3134870_0` | `0c4f863db783db2fd1b0d6ceb01b9da7962e503e3ddfee500eca364d68c0466b` |
| `aa_unrecognized.json` | `3134870_1` | `2fab908cfa1d97b60b8aca984dd219dd576535cf751b0f28f60e822cd7169500` |
| `same_recognized.json` | `3134870_2` | `ef50fdf08cdb7cb1c34788cc6fa2e89a593c1b028c9b123d8cf266483253cce0` |
| `same_unrecognized.json` | `3134870_3` | `13f5f51d92a96759a707e5a936faddfb3b1fa086977dd5dbe4bf27cad1282c9a` |
