# F25e · Signed native modulation exports

19 November 2026, 09:03:11 CET. Both matched A100 source groups used all 2,655 public PROTEUS windows, the exact original F25 hard and two gate operators, and 1,720 native updates. The bare source hard fit reproduced the matched hard state and checkpoint bytes on each seed. The A100 recipes retained the original H100 initialization and schedule. The measured source scale differed by one float32 ULP and fitted hard fingerprints differed across devices. This is a matched A100 release. [Source and export protocol](prereg-19nov.md)

The pinned worker-image CPU exports matched all 576 original bounded CPU batch decisions over two real source sessions, with maximum posterior deviation 1.33910e−6. All 12 first-window, truncated-prefix and foreign-session reset checks passed, with active bounds on every model. The self-contained actors consume a contiguous CPU float32 `[1,41,480]`, emit normalized float64 `[1,3]` in official order, and clear the full causal history on zero-returning `reset()`. The validation uses only source waveforms, with labels closed during export.

The release is `/mnt/lustre/work/gehler/gfh527/ni26_friday_s1c_modulation_external_a100_20261119`. Use `exports/{hard,single,multi}/raw_s1c_s{0,1}.pt`. `single` is the matched conventional gate and `multi` is the multiresolution gate. The source and CPU smoke jobs `3135003/3135008` passed. Both source elements `3135009_0/1` passed in 86 seconds each, followed by both CPU exports `3135010_0/1` and the independent seal `3135011`. Scherer received zero exposure in training and export.

| Artifact | SHA-256 |
|---|---|
| Complete bundle manifest | `7c50ac96002ea129a1bffa6179f1a58174376502147ca2ed4534cb41dc406c50` |
| Source actor | `b9d64a585b6f75a806a0d27c8b04fd32d41c8baf565a46ac07e7835e2c673e4b` |
| CPU exporter | `65583a897b96f5ac9ae76a7aac260aadd1f257086412828fe6b9049a5f8ce0ae` |
| CPU seed 0 receipt | `4da45d4d15fcab4328ac6485b8cefa9c66ae32371aa68071e2608ae68215d3e7` |
| CPU seed 1 receipt | `0b710984965a7f1d9f372e12f42cb25f50fd6120050436a6f777899272e76a85` |
