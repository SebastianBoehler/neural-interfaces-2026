# Project instructions

This project targets Track 2 BCI decoding in Neural Interfaces 2026.

- Read docs/STATE.md first (current state), then docs/LEDGER.md (what is closed) and OFFICIAL_CONTRACT.md.
- docs/archive/ is history; where it contradicts docs/STATE.md, the state file wins.
- Pre-register every run in docs/LEDGER.md (mechanism, numeric prediction, refutation rule) before scoring it.
- Check ideas against the ledger index first; score members and stages in the chain (`replay/`), never alone.
- No gate or closure is final: before citing one, check what actually ran (code path, model class, alone or in the
  chain, which system version). State a rule before the run; say plainly when and why you go past it.
- A cleanup or review must also hunt silent no-ops: options parsed but not applied, overrides that drop a parent's
  behaviour, cache keys that omit a setting, runs whose code path differs from their description. Make them fail
  loudly.
- `python replay/check.py` must print OK before any chain result is trusted.
- You own this codebase: edit and clean up freely, commit and push often, with no AI author line in commits.
- Never edit submissions/*/submission.py in place: their SHA-256 values are pinned to uploaded or staged packages.
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
