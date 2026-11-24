# Public release audit · 24 November 2026

Status: **not cleared for publication**. This is a preliminary audit, not a
complete secrets, data provenance, or third-party license review.

- GitHub repository visibility was private. Its only advertised branch was
  `main` at `076ae6b80266decb204081c9615cbcda27ecd096`; there were no tags,
  releases, or pull requests. Local `main` was four commits ahead.
- The index contained 19 generated `.npz` files. They were removed from the
  index while preserving the local files. `.npy` and `.npz` are now ignored.
  `laneK2/chain/feat_k2.npz` contains PROTEUS labels, person codes, and derived
  features. The sampled lane H and B6 archives contain per-window predictions.
  These are public-data experiments; they are not evidence of sealed labels.
- The remote main history contains 36 matching NumPy archive objects. Removing
  files from the current index does not remove them from history. Review their
  contents and redistribution terms before choosing whether to publish the
  existing history or prepare a separate public history.
- A Gitleaks scan of the current indexed files and checked-out submodule found
  three false positives: two metric keys in the upstream submodule and one
  SHA-256 hash in `laneC4/braincodes_member.py`. This scan does not cover
  ignored files, all binary contents, Git history, GitHub metadata, or every
  possible credential pattern. The all-ref history scan was stopped before
  completion because its Git process used more than 2 GB of memory.
- `native/vendor/scipy_lbfgsb` retains its SciPy license and notice. Other
  third-party source, model and dataset terms need a complete provenance review.
  No top-level license was found; choosing one is an author decision.
- Local untracked files and ignored artifacts are not part of a Git push. Do
  not add them without review. The upstream submodule remains unchanged.

Before publication: complete a memory-bounded scan of every history that will
be exposed; inspect binary and data-derived artifacts and third-party terms;
review personal information and GitHub metadata; select a license if desired;
and verify the exact public refs. Rewriting existing shared history can disrupt
other checkouts and has not been performed.
