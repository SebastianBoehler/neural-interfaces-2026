# Native CUDA replay of the coherent waveform decoders

Both frozen learned decoders reproduced every PyTorch decision on held public EEG. Across ten held people, eight pinned orders, two training seeds and two history conditions, the C++ implementation matched 84,960 of 84,960 labels. Its largest absolute logit difference was 0.0000011921. All 224 session starts produced identical genuine and blank logits. The registered parity criterion passed. The maximum error was smaller than the predicted 80% range.

| Training seed | Genuine history, individual-model dev | Same weights, blank | Native labels matching reference | Genuine decisions changed by blanking |
|---|---:|---:|---:|---:|
| 379 (§439) | 43.923% | 42.160% | 42,480/42,480 | 8,052/21,240 |
| 1441 (§441) | 45.629% | 41.774% | 42,480/42,480 | 8,533/21,240 |

An independent C++ implementation applied source RMS scaling, both waveform convolutions, the learned write and read, feature adaptation, pooling and classification. One native invocation received one session in delivery order. Each decision used earlier writes, then the current EEG updated the sum. Blank supplied a zero vector and zero age at every window. The host supplied the source-only scalar, unchanged EEG and frozen parameters. The comparison verified the checkpoint, original prediction archive, §443 reference logits, fold ownership, input and order hashes before reading outputs.

The table uses the existing official development metric, which follows directly from complete decision parity. The earlier matched-age donor scored 44.966% and 46.381%, respectively. Blank removes both age and content, so its performance does not identify a participant-specific advantage. The historical frozen full stack scored 76.279% on these same people.

Galvani completed the build (3126298), live P01 smoke (3126308), twenty A100 actors (3126309) and collector (3126310) by 11:58:05 CET on 18 November. The smoke checked 48 decisions, with eight of 24 genuine labels changing when memory was blanked. LibTorch 2.14.1+cu126 executed the diagnostic library. Training remained the original PyTorch training, and the production package did not execute this decoder. No sealed score was measured.

[Per-fold hashes, errors and reset counts](native452_result.json) · [Native decoder](../../studies/18nov/native452/decoder.cpp) · [First seed](gain439.md) · [Second seed](gain441.md)
