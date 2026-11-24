# History note

Before the public release, the repository history was rewritten once with `git filter-repo` to remove generated
`*.npz` and `*.npy` archives (PROTEUS labels, person codes and per-window predictions). Every commit keeps its message,
author and date, and the file tree at each commit is the same apart from those archives. Commit hashes changed.

Notes, ledger entries and evidence files written before the release cite the old hashes. To find the commit in this
history, look the old hash up in [COMMIT_MAP.txt](COMMIT_MAP.txt) (`old new`, full 40-character hashes), for example
`grep ^cb3e05a docs/COMMIT_MAP.txt`. The citations of `cb3e05a` (the frozen original ledger) were updated to `a5843492`.
Hash-pinned scripts were left untouched, so their embedded commit hashes need the same lookup.

The archives themselves are not published. Experiments whose evidence was an archive keep their logs, scripts and
result summaries, and can be re-run from the scripts.
