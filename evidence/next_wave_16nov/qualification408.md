# §408 Whole-worker finite-prefix pruning

Pruning saved **3.06%** on the 199-window returning session and **2.40%** across the two naturally adapting external sessions (median paired whole-session wall). Both default-CUDA runs matched all 1,332 archived labels. An independent deterministic A100 pair matched all 3,996 post-personalization probability values byte for byte. Both timing predictions held for these streams.

| Stream | Stock / pruned EEG views | Stock then pruned, seconds | Pruned then stock, seconds (stock / pruned) |
|---|---:|---:|---:|
| P10 day 3, returning | 199 / 40 | 29.38 / 28.48 | 29.63 / 28.73 |
| Schirrmeister 1, naturally unknown | 320 / 150 | 115.18 / 113.80 | 117.65 / 117.28 |
| Schirrmeister 2, naturally unknown | 813 / 150 | 156.18 / 150.51 | 158.29 / 152.60 |
| **Actual complete process** | | **351.99 / 343.56** | **358.14 / 349.75** |

The private edit saved 8.43 s and 8.39 s per full worker process (2.39% and 2.34%). P10 latched at 40, recognized at 10 and ran no neural refits. Neither external recording latched or recognized. Each made 24 neural refits and 480 optimizer steps. They fused 230 and 723 personalization outputs. All model-work counters matched, and no stage reported an error or time guard. These two matched timing pairs cover three particular streams and cannot establish the private worker mix or wall time. §406's 57.76% measured the isolated CPU filter. [Stage measurement](filter406.md) · [Paired measurements, row and probability hashes](qualification408_result.json)

A fresh process loaded the original cap250 archive extracted in §385 or the private source-edited copy. All 38 other members have identical SHA-256 values. Both arms warmed on P01 (31 windows) and a genuine external prefix (35 windows, 40 optimizer steps). Each then used an official fresh model per complete session and the original §387 shuffled row lists. Default PyTorch CUDA handled timed runs. Session wall includes the model copy and all synchronized predictions. Process wall includes launching Apptainer, model loading, warm-up, measured sessions and shutdown.

The default stock/stock repeat matched all 195 archived labels and all model-work counters. Both external prefixes produced different posterior hashes across stock repeats even though neither had reached a pruning decision. P10 produced identical posterior bytes. The revised parity arm fixed NumPy and Torch seeds at 408, enabled deterministic Torch kernels, disabled cuDNN benchmarking and set the cuBLAS workspace to :4096:8. This kept timing on the default backend and made the independent posterior measurement reproducible on the smoke.

Frozen ZIP SHA-256 9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed, stock submission.py 37c666a8a63c043c290bd35c527c26c6cfaf028bbe55c746cb8c685dac9fb69f, private submission.py d1d6ebbfaa02e88e173d11c78dbd25ded3c490c7b4b6f6e94326947e41f81071. PyTorch 2.14.1+cu130 in worker image 7fc0389c46ac6bec7c4395a9815e9ffee04d207a2a3700aeb3aef318bf79e2d0. Galvani source directory ni26_qualification408_v2. Deterministic smoke 3124319, default stock/stock diagnostic 3124320_3, full array 3124321, CPU collector 3124379. Initial default smoke 3124280 failed its strict posterior guard; its dependent array 3124281 was canceled before launch. All six recording-order v3 replay reference scores also matched.
