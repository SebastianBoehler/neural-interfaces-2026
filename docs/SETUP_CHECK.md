# Repository setup verification

Checked 17 October 2026 after moving the project to
`/Users/sebastianboehler/Documents/GitHub/neural-interfaces-2026`.

- Full `bash scripts/run_baseline.sh` completed successfully from the new directory.
- The real-data audit, training, packaging, ingestion and scoring all passed.
- All four metrics exactly matched the historical baseline evidence.
- The official source revision remained unchanged.
- Shell syntax checks and Git whitespace checks passed.
- GitHub reported a private repository with default branch `main`.
- A fresh recursive clone retrieved the pinned official submodule successfully.
- Data, environments, logs and generated weights were excluded from Git.

The existing environment was preserved through the old workspace symlink.
A clean dependency installation was not repeated during this relocation check.
Fresh clones use `scripts/setup.sh`; that macOS ARM dependency snapshot has
not been validated on Linux/CUDA. No Codabench upload was performed.
