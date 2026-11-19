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

## F-C25pt warm RTX result

Both profiles met the original limits of 1.12× mean and 1.15× worst paired elapsed. Each measured pass used the official Solver and a fresh session. All signed classes and native work matched. Each individual CPU vector matched the original signed oracle bitwise on every hard row. The G run executed its 200 optimizer steps and 30 CSP fusions. P10 recognized at age 10. Independent G posterior repeatability still exceeds 5e−5.

| Fixture | Stock elapsed | Trial elapsed | Mean ratio | Worst pair | Completed 19 Nov CET |
| --- | ---: | ---: | ---: | ---: | --- |
| G unrecognized | 30.050 / 30.045 s | 31.754 / 31.849 s | 1.05837× | 1.06004× | 09:34:23 |
| P10 visit 3 recognized | 4.721 / 4.752 s | 5.181 / 5.146 s | 1.09009× | 1.09729× | 09:31:23 |

| Fixture and archive | Restore per pass | First window per pass | Window p95 per pass | Peak GPU reserved |
| --- | ---: | ---: | ---: | ---: |
| G stock | 95.8–96.2 ms | 176.6–177.9 ms | 710.4–720.5 ms | 2,602 MiB |
| G trial | 89.0–94.1 ms | 218.1–219.7 ms | 722.9–726.4 ms | 2,594 MiB |
| P10 stock | 85.1–96.1 ms | 150.2–156.1 ms | 153.6–153.8 ms | 100 MiB |
| P10 trial | 90.1–90.7 ms | 195.4–196.3 ms | 165.1–166.6 ms | 100 MiB |

Immutable Galvani release `ni26_friday_c25pt_hard_runtime_v2_20261119` source `d1d4a39ff7aa2aab295fffb7c122d3fbab8abdcf0263d6ec02640250039b3776`. Job `3135142_0` G receipt `c94bfff42f87d4ddd3832d00fb433f7243544eeca44bc7ebc753469a99ede861`. Job `3135142_1` P10 receipt `ae050c0e7de043a43d5948c01201d96add7e8c0a1e6c89ba00f8f225c44582e3`. The earlier runtime harness `3135136` failed receipt preflight because it expected a newer diagnostic field in the older strict P10 receipt. The corrected harness verifies the distinct original schemas and exact hashes. No package bytes or checks changed.
