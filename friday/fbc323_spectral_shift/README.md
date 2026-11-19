# F37 source and held execution

[Prospective design](prereg-19nov.md) · [Source transform](spectral_shift.py) · [Transfer patch](transfer.patch)

The transfer patch applies to the original F17 source, held and relay entry points using `git apply --unidiff-zero` in a separate release directory. The source and held workers verify the immutable F17 actor manifest before reusing its selection tape and native input adapter. The original complete F16 hard control and published F17 assets stay in place. The release keeps the new actor identity separate.

| Cluster | Job | Scope |
| --- | --- | --- |
| Ferranti | 511781 | Four-epoch H100 source smoke, P01 excluded |
| Ferranti | 511782 | Twenty H100 training folds, cap three |
| Galvani | 3135762 | Independent, label-closed CPU held smoke |
| Galvani | 3135763 | Twenty native CPU held folds, cap six |
| Galvani | 3135764 | Reauthenticate complete predictions and score official cells |

Ferranti release `~/ni26/friday_s1c_spectral_shift_20261119`. Galvani release `ni26_friday_s1c_spectral_shift_20261119`. Actor `0575236e532b5858a28c0e510d8fda3030cc5f4584648aa657642347e6f833bd`, scorer `70deb2a3c25e1a2a569afd435aea726e9722ad5e442440dfa00fb7e79a90e13d`.

Source smoke passed at **10:40:12 CET** in 13 seconds. The real hard checkpoint matched original F16 bytes. Both active arms changed exactly 2,528 of 10,112 original row appearances. Mean normalized squared waveform displacement was 0.000874817, maximum row-wise difference 5.9062e−10 and maximum sampled spectral error 1.3648e−7. The controller transfers the exact source smoke files and holds the full source array until the separate CPU smoke passes.
