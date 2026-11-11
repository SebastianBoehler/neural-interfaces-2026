# Project instructions

This project targets Track 2 BCI decoding in Neural Interfaces 2026.

- Read README.md, OFFICIAL_CONTRACT.md and BASELINE_RESULT.md first, then docs/STATE_11NOV.md (current state).
- Keep the official 2026-competition submodule pinned and unchanged.
- Keep our implementations and scripts outside the official submodule.
- Preserve the official participant split, preprocessing and metric for comparisons.
- Never commit datasets, environments, credentials or generated checkpoints.
- Track compact evidence with source revision, seed, metrics and validation limits.
- Validate locally before proposing a Codabench submission.
- Upload only with explicit user authorization; check live quota and phase first.
- Native CPU replay is not proof of Linux/A100 worker compatibility.
- Keep source files below approximately 300 lines and diffs surgical.
- Verify changes with the narrowest meaningful real-data check.
- Run local training only through `scripts/run_capped.sh` (one job at a time,
  process-tree RSS cap, MPS high watermark); never several MPS jobs in
  parallel. The Mac shares RAM with the GPU: parallel jobs froze it (Oct 22).
  Prefer Galvani for anything heavy.
