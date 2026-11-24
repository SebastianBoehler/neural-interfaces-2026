# Public release audit · 24 November 2026

Status: **ready to publish once the author picks a licence and pushes the cleaned history** (steps at the end).

## Organiser rules

Upstream `2026-competition` main (`df31ee73`, 22 Nov) has no rule against publishing code. The Terms
(`codabench/pages/terms.md`) say original dataset and model licences remain in force, and the top three teams must
supply a method description and training and inference code for the reproducibility audit (Rules 7 and 8). Sealed
data must never be used or released; this repository contains none.

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
- No top-level `LICENSE`. Without one, others have no right to reuse the code. The author chooses.

## To publish

1. Pick a licence and add `LICENSE` (recommend MIT or Apache-2.0 for code; third-party and dataset terms stay in force).
2. Rebuild the clean history from the final commit:
   `git clone --no-local . ../neural-interfaces-2026-public && cd ../neural-interfaces-2026-public && git remote remove origin && git filter-repo --force --invert-paths --path-glob '*.npz' --path-glob '*.npy'`
3. In that clone add the `origin` remote and `git push --force origin main`. This replaces the private remote's
   history. Other checkouts must re-clone.
4. Set the repository to public (Settings → Danger zone) and check the GitHub Insights tab for stray branches.
