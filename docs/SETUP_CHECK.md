# Repository setup verification

Checked 17 October 2026 after moving the project to
`/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026`.

- Full `bash scripts/run_baseline.sh` completed successfully from the new directory.
- The real-data audit, training, packaging, ingestion and scoring all passed.
- Training and packaged replay matched within this run on all four metrics.
- Cell-average balanced accuracy was 37.8223392504%, compared with the historical
  run's 37.7543528303%. The cross-run difference is not yet explained; fixed-seed
  execution did not produce identical scores across these two runs.
- The official source revision remained unchanged.
- Shell syntax checks and Git whitespace checks passed.
- GitHub reported a private repository with default branch `main`.
- A fresh recursive clone retrieved the pinned official submodule successfully.
- Data, environments, logs and generated weights were excluded from Git.

The existing environment was preserved through the old workspace symlink.
A clean dependency installation was not repeated during this relocation check.
Fresh clones use `scripts/setup.sh`; that macOS ARM dependency snapshot has
not been validated on Linux/CUDA. No Codabench upload was performed.

The second run's validation record is tracked in
`evidence/relocation-2026-10-17/validation.json`. Historical evidence remains
unchanged under `evidence/baseline/`. Runtime files under `runs/baseline/` now
contain the second run, including its newly generated ZIP and logs.
