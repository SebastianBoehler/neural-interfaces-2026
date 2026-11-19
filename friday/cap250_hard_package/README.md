# F-C25p · Direct hard waveform trial ZIP

The trial is a deterministic rebuild from exact cap250 and the two pinned F32e CPU hard models. Thirty-eight original non-code members are byte-identical. The new root entries are `hard_s1c_s0.pt` and `hard_s1c_s1.pt`. The original Solver factory selects an appended `HardWaveEnsemble` subclass. Each call feeds the current unchanged NetBank waveform through both bounded CPU actors. Their equally weighted probability mean replaces five frozen outer raw FBCNet members, using each original 1/90 weight. Native seed 47, its computed warm delta, the original protected inner, stateful RouteB, count filter and CSP remain in the circuit. Actor weights are immutable bytes and session state is instantiated privately and lazily. [Preregistered archive and timing tests](../../docs/LEDGER.md#f-c25p--direct-hard-only-waveform-trial-package-and-native-qualification)

| Artifact | SHA-256 |
| --- | --- |
| Stock cap250 | `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed` |
| Direct trial ZIP | `6a569180d343c038558fe7eccb7903a36e2bdf694af0df3687bff3b17049fbcc` |
| Exact F32x actor lock | `0292022abd39067cefebd0e3a0bd8f1834a04bfc67d7d41081cfe6abaae8ebd2` |
| F32e CPU seed 0 | `ec33684850e902479f3bf28e472a9ea590e64fa759b0b03704daf7aff1ea4018` |
| F32e CPU seed 1 | `ed83b6b5b2a2765f82afa4a2fcb3f2c526d24a32e8b789b527268bde67b5401e` |
| Trial source and manifests | `69abae0f51fa89f6dd420dad8b863d70ba530638288575e34f7c7a5fb2706489` |

Immutable Galvani release `ni26_friday_c25p_hard_trial_v2_20261119`, archive `v66_cap250_hard_trial.zip`. `3135066` exercises the recognized and unrecognized paths through the exact GPU Solver. `3135067` warms and times both paired profiles only after the smoke. The signed F32x and F-C25 raw hard-score fixtures matched bitwise across both original CPU hosts before the new direct tests. This run involves archive conformance and speed. It opens no task labels.

The v1 G smoke passed all 120 hard decisions, the exact signed pupil oracle and its same-input tail reconstruction. Its final independent comparison included the cold stock repeat, which differed from warmed stock by 0.00102144. The failure and canceled timing remain. V2 separates the authenticated cold stock observation and compares two genuinely warmed runs before live independent hard parity. Each fixture has its own real test. The trial ZIP bytes and all stated warm tolerances are identical to v1.

## Conformance evidence

| Profile | Worker | Scope and outcome | Signed receipt or log SHA-256 |
| --- | --- | --- | --- |
| G, v1 | `3135050` | Exact same-input proof and 120 signed decisions. Cold-included stock comparison exceeded 5e−5. Dependent timing canceled. | `b36d86cccde5588063035ad3a761a8735794002de3e743527fe647da9cacbcdc` |
| G, v2 | `3135066_0` | Exact same-input proof and 120 signed decisions. Genuinely warmed untouched stock A/A exceeded 5e−5 at six values, reaching 8.29270e−5 at age 106. Dependent timing canceled. | `9c7b856f26d7aafad3fb932d69e106be4ecc23e5432e4a990ea4806201c52b88` |
| P10 visit 3, v2 | `3135066_1` | Passed all 31 signed decisions, exact CPU vectors and same-input stages. Recognition at age 10. Independent protected maximum 3.89942e−6 and both reset maxima 7.96914e−8. | `e1b650082d00b0577453a4c4abb7e378b6993309d0c199c324878ba355d1d808` |
| G, same-input diagnosis | `3135121` | Reproduced all 120 hard decisions and CPU vectors. Verified the native mixer, independent private tail and its prefix bitwise. Both 17-row session resets reproduced CPU vectors and classes, with stage drift ≤3.37120e−8. | `ba3cc6f6d9bdf373ea065e76c293b2b7bee46b281ca729d71a12259e24e9a190` |

The v1 source manifest is `71df5c17bf6fedf05fa4e3bfabc5f7c29d9350d1aa5e4c944c134753c41781a3`. The G diagnostic release `ni26_friday_c25p_hard_diagnosis_v1_20261119` uses source manifest `012cb9f01cb88baa0cbfdbc311bf3470c4107d5759fc35006cf926d3ae3d8ce2`. It exercised 100 warmed raw rows, 200 protected optimizer steps and 30 CSP fusions. Warm stock A/A FBLight reached 5.88716e−5 at age 106. The independent stock/direct warm FB comparison reached 0.00102144 at age 116. The exact same-input G calculation and both reset checks passed. The independent G posterior comparison remains vetoed at 5e−5.

F-C25pt measures warm runtime of the same frozen archives with the P10 and G receipts pinned. This test keeps the independent G posterior findings and requires the actual signed CPU vectors, emitted decisions and full work on each timed session. [Preregistered runtime scope](../../docs/LEDGER.md#f-c25pt--scope-limited-direct-zip-runtime-on-two-authenticated-fixtures)
