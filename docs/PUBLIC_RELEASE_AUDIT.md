# Public release audit · 24 November 2026

The cleaned main branch was pushed and the repository is public. The release uses [Apache 2.0](../LICENSE).
GitHub still serves at least one NumPy blob from the pre-rewrite history by object ID.

## Organiser rules

Upstream `2026-competition` main (`df31ee73`, 22 Nov) has no rule against publishing code. The Terms
(`codabench/pages/terms.md`) say original dataset and model licences remain in force, and the top three teams must
supply a method description and training and inference code for the reproducibility audit (Rules 7 and 8). Sealed
data stays private. The source tree contains no sealed data.

## What was checked

- **Secrets, current tree and submodule:** Gitleaks found three false positives (two metric keys upstream, one
  SHA-256 in `laneC4/braincodes_member.py`).
- **Secrets, all history:** Gitleaks over every ref (3,658 commits, 1.09 GB, 11 min, memory under 1 GB) found two
  false positives (a SHA-256 in `laneC4/braincodes_member.py`; prose in `docs/STATE.md`). No key, token or password.
  A manual grep for private keys, `ghp_`/`hf_`/`sk-`/AWS patterns, and e-mail addresses found nothing.
- **Generated data:** 19 `.npz` files (PROTEUS labels, person codes, per-window predictions) were removed from the
  index; `*.npz` and `*.npy` are ignored. The old history held 36 such blobs and was 1.5 GB. `git filter-repo` on a
  fresh clone removed them from every commit: 15 MB, 3,034 commits, same file tree as `main`, zero `.npz`/`.npy`
  objects, `git fsck` clean.
- **Weights and datasets:** none tracked. The 417 MB final `submission.zip` (trained weights, over GitHub's 100 MB
  file limit) is gitignored; README and `docs/STATE.md` no longer link it.
- **Personal paths:** 1,177 occurrences of the author's macOS home path in 34 files were replaced by `<repo>` and
  `<home>`. `experiments/ts_core.py` and `ts_core2.py` keep theirs because `.sha256` manifests pin them.
  The history still holds the old paths (not secret, not rewritten).
- **Local scratch:** `work/` (3.6 GB), `ssh_dummy/`, `moonM2/stage0.log` are untracked and now ignored.
- **Third party:** `native/vendor/scipy_lbfgsb` (SciPy licence and notices), `native/worker-parity/DUCC-LICENSE.md`.
  No other vendored source found. The upstream submodule is unchanged and points at its public URL.
- **Commit identity:** every commit uses the GitHub noreply address. 153 commit messages carry honest
  `Co-Authored-By: Claude` trailers.

## Accepted exposure (author's call)

- Cluster identifiers stay: Galvani account `gehler`, login `gfh527`, `/mnt/lustre/...` and `/leonardo_work/AIFAC_...`
  paths (about 3,100 occurrences, many in scripts pinned by `.sha256`). These are usernames and project names, not
  credentials. Scrubbing them would break the pins.
- Docs paraphrase organiser clarifications from the competition Discord and link to channel messages. The links need
  server membership to open.
- `AGENTS.md` and `.claude/` describe the author's agent workflow, usage limits and model choices.

## GitHub object retention

An unauthenticated GitHub API check returned HTTP 200 for the old commit and one removed NumPy
blob. The rewritten branch does not reference those objects. Purging the retained copy requires
GitHub-side cleanup.
