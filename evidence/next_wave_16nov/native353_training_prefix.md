# Six-update CPU training parity

## Initial failure

Galvani job `3122002` ended at 19:29:43 CET on 17 November 2026 after 139 seconds and 278 CPU core-seconds. Both paths completed six source updates on `galvani-cn001`. The original `SessionNet._train` reproduced all first-update oracle fields exactly. The native path reproduced the 1,898 prepared windows, 26 initialization fields, 1,240-batch plan, and OneCycle schedule. Both paths used Torch 2.14.1 and trained on the same source-only batches. The job did not run a full fit or score held data.

| Comparison | Update 1 | Update 2 | Update 6 |
|---|---:|---:|---:|
| Logits, max abs | 0 | 0 | 5.9605e−5 |
| Posteriors, max abs | 0 | 0 | 1.3345e−5 |
| Gradient, max abs | 0 | 0 | 2.9862e−5 |
| First AdamW moment, max abs | 0 | 7.4506e−9 | 8.0168e−6 |
| Second AdamW moment, max abs | 0 | 0 | 4.3772e−8 |
| Post-update state, max abs | 0 | 1.8626e−9 | 1.4289e−4 |

Updates 1 and 2 pass. At update 6, the spatial-convolution bias is the largest post-update state mismatch. Its first and second moments have relative L2 errors of 0.924 and 0.549. The harness captured every preregistered numeric field and reported 23 bound excesses. Bounds stay fixed at `1e-5` for logits/gradients, `1e-6` for posteriors/loss/state, `1e-7` absolute and `1e-6` relative L2 for each moment, and `1e-15` for the schedule. Counters must match exactly. Job times include a 35.73-second Python process and a 7.60-second native process. The processes do different diagnostic work and the native trajectory fails parity, so these times give no speed comparison.

## AdamW arithmetic evidence

The earliest compared mismatch is the first AdamW moment at update 2. The installed Python `torch/optim/adam.py` uses `exp_avg.lerp_(grad, 1 - beta1)` with decoupled weight decay for AdamW. The loaded `libtorch_cpu.so` (`3d793582452c4f0959e191b73fdfa1b592360735cc37cfe2362ab7308a52901d`) implements `AdamW::step` at `0x6b94b50`. Its first-moment call sites invoke `mul__Scalar` at `0x6b94e38` and `add__Tensor` at `0x6b94e66`. This is the same operation-order difference as the existing `FineTuneAdam` repair for Adam. That identifies a concrete candidate for the later divergence.

The arithmetic replay passed on one CPU thread in 6.62 seconds of measured process work. It used the saved update-1 parameters and moments, exact update-2 gradients, learning rate `4.001720919567721e-5` and beta `0.9499982073754503`. Multiply-add reproduced all six measured native first-moment max and relative errors and all six parameter max errors exactly. Changing only the first-moment formula to `lerp_` reproduced all six Python first-moment tensors and updated parameters exactly. The second moment matches exactly in both arms. The comparison uses native numeric summaries because the native update-2 arrays were never saved. The replay used only the 855,797-byte original trace and native log, with zero new Slurm allocations, forward/backward passes, or production edits.

This establishes the source of the observed update-2 error. A six-step production rerun is needed to check later gradients.

Snapshot `ni26_native353_source_prefix_v1_20261117`. Source manifest `22e2bad25195c38a44ebff8031e1d2b8fc14fb799de991bee3f07f0710870214`. Original trace `bf26fe55773afde012afb8f5e63c1e2b4aad8e6d517b34f1a2fd82573e868c36`. Native log `d7b32d92a690354b0584af797f5d6302f485a1885d6dba359a3de402063140b1`. Build proof `1750ee75a46466322ae3619ee743ca099b48e38129d9e4f1839913bd23d3500e`.

## Shared Adam repair passed

The native trainer and online fine-tuning now share one Adam implementation. Offline training passes the OneCycle learning rate and first-moment beta, and applies decoupled weight decay at `.001` before the moment update. The online call keeps its original learning rate, `.9` and `.999` betas, and zero weight decay.

Galvani job `3122019` completed at 19:39:48 CET on 17 November 2026. It took 171 seconds and 342 CPU core-seconds. The original reference trace matches the initial run byte-for-byte. At updates 1, 2 and 6, native logits, posteriors, loss, all six gradients, both moments, all 17 compared state tensors, and schedule fields matched exactly. Counters match, all source and initialization checks passed, and the source covariance-prior difference remains `2.956e-12`. The update-6 spatial-convolution bias error fell from `1.4289e-4` to zero. Each manifest and loaded library checksum passed before and after execution.

This qualifies the six-update CPU prefix on the existing source fixture. The full 1,240-update training trajectory, held predictions, CUDA training, and whole-pipeline speed still need their own evidence. The separate worker pilot uses its earlier pinned inference binary.

Snapshot `ni26_native353_source_prefix_v2_20261117`. Source manifest `b4f0e0c5d6fd7f0a6dd2b6a26423f37c8964515162f51d604829ebaf5be89903`. Native log `45e8483bd5d1c78d2fbe97dcb7921f08091767ffb4cdb89f4c4cd7b73283cad3`. Build proof `bc0c1a382570aa1f8ebc4a5b6ebf7bc3146b3107e89c8ab7e5a7cdda89ebe7e4`.
