# F-C25p · Direct hard waveform trial ZIP

The trial is a deterministic rebuild from exact cap250 and the two pinned F32e CPU hard models. Thirty-eight original non-code members are byte-identical. The new root entries are `hard_s1c_s0.pt` and `hard_s1c_s1.pt`. The original Solver factory selects an appended `HardWaveEnsemble` subclass. Each call feeds the current unchanged NetBank waveform through both bounded CPU actors. Their equally weighted probability mean replaces five frozen outer raw FBCNet members, using each original 1/90 weight. Native seed 47, its computed warm delta, the original protected inner, stateful RouteB, count filter and CSP remain in the circuit. Actor weights are immutable bytes and session state is instantiated privately and lazily. [Preregistered archive and timing tests](../../docs/LEDGER.md#f-c25p--direct-hard-only-waveform-trial-package-and-native-qualification)

| Artifact | SHA-256 |
| --- | --- |
| Stock cap250 | `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed` |
| Direct trial ZIP | `6a569180d343c038558fe7eccb7903a36e2bdf694af0df3687bff3b17049fbcc` |
| Exact F32x actor lock | `0292022abd39067cefebd0e3a0bd8f1834a04bfc67d7d41081cfe6abaae8ebd2` |
| F32e CPU seed 0 | `ec33684850e902479f3bf28e472a9ea590e64fa759b0b03704daf7aff1ea4018` |
| F32e CPU seed 1 | `ed83b6b5b2a2765f82afa4a2fcb3f2c526d24a32e8b789b527268bde67b5401e` |
| Trial source and manifests | `71df5c17bf6fedf05fa4e3bfabc5f7c29d9350d1aa5e4c944c134753c41781a3` |

Immutable Galvani release `ni26_friday_c25p_hard_trial_v1_20261119`, archive `v66_cap250_hard_trial.zip`. `3135050` exercises the recognized and unrecognized paths through the exact GPU Solver. `3135051` warms and times both paired profiles only after the smoke. The signed F32x and F-C25 raw hard-score fixtures matched bitwise across both original CPU hosts before the new direct tests. This run involves archive conformance and speed. It opens no task labels.
