# §281 fine-tuning implementation parity diagnostic

Pre-registered in `docs/LEDGER.md` at 12:35 UTC on 16 November 2026. The one real-data diagnostic passed on CPU under `scripts/run_capped.sh`, with one thread and the default 8 GB process-tree limit. It ran within0, base B2_s0, P10 session 2, refit t=20 and the shipped FBLight member 59. The pinned v6.6 cap-250 ZIP SHA-256 is `9c4d553823b5a51cd9bcfec02eb0991d4305eddd784a39782737799e7d9117ed`; the exact member trace SHA-256 is `a91b737bb4904e36fd158b890a8a3e6aa42c057c7f38edddea1e3688533e5079`. The matching prefix artifact SHA-256 is `9b131eac09b8bffc2d9ca9aebcee42232fe0e163a1a5d7b62cc013b46022321b`.

The real package `FineTune._refit` and reconstructed eager twin used five updates and independent, matched seed-0 augmentation generators. All augmented tensors and generator states matched. Initial logits, every loss, relative gradient L2 comparison and per-step parameter comparison had zero measured difference. The next-window probability maximum absolute difference was 3.127013925574573e-8. The largest package parameter update was 0.0050147101283073425; gradients and updates were nonzero. Trainable parameters, shapes and the frozen parameter set matched. The extracted input stream and the eager reconstruction also passed their existing package parity guards.

Source SHA-256: `s281_ft_parity.py` is `cf72313afd4a8fc3ca2ea5f0bdbfb5ac83b5f12a181eadc05a914d004acbc5a8`; `test_s281_ft_parity.py` is `980277ff0266120dd1d83a899b5d434da2ca4ec098153ec3bf7814a0ac226eba`. Three narrow guard tests passed. The log is `runs/s281_ft_parity_within0_B2_s0.log`. The first launch failed before loading data because isolated Python could not import the project package; the import path was fixed before the successful run.

Repository HEAD at the diagnostic was `920e83196fdda63a8a1170e083d323a6bba731eb`; the diagnostic and test were untracked additions identified by the hashes above. Commands:

```sh
.venv/bin/python -m unittest discover -s studies/14nov/newmember -p 'test_s281_ft_parity.py'
scripts/run_capped.sh runs/s281_ft_parity_within0_B2_s0.log .venv/bin/python -I studies/14nov/newmember/s281_ft_parity.py
```

This validates only one architecture, one refit, one CPU and one session input. The package weights were trained using the public data, so this is not a held-out score. It does not establish GPU, other-member, full-session, routed-tail, or end-to-end returning-session parity.
