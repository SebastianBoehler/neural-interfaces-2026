# Friday · Neural Interfaces 2026, Track 2 (BCI decoding)

Research code and experiment record for our entry in **Track 2 of the [Neural Interfaces 2026](https://github.com/neural-interfaces26/2026-competition) competition** (NeurIPS 2026 Brain and Body Workshop): decoding mental commands from short EEG windows, for people the decoder has never seen, with no labels at test time.

This is a research archive, not a library. It contains the decoder source, the pre-registered experiment ledger, the evidence behind every claim, and the cluster scripts used to train and score. Read it to see what was tried, what worked and what did not. Re-running everything needs the data, GPU clusters and weights described under [Reproducing](#reproducing), and several of those are not included.

## The task

One four-second, 41-channel EEG window at 120 Hz goes in, and one of three classes comes out immediately: **kinesthetic motor imagery, mental subtraction, or word generation**.

- The decoder gets no labels, no person ID and no guarantee about recording order. It starts fresh each session.
- The score is balanced accuracy, averaged equally over cells (participant, session, interface). Seen (returning) and unseen (new) people are scored separately and averaged unweighted.
- The sealed run has a 3,600 s budget on one RTX PRO 4500.
- The public warm-up data is the PROTEUS release ([NEMAR nm000290](https://doi.org/10.82901/nemar.nm000290)). The sealed recordings are private and were never available to us.

The full contract, with sources, is in [OFFICIAL_CONTRACT.md](OFFICIAL_CONTRACT.md).

## Results

| | Sealed balanced accuracy | Notes |
|---|---|---|
| Best recorded sealed upload (`v6.6 cap250`, submission 1058969) | **0.8299** | 2,555 s runtime, about rank 15–16 on the live table on 19 Nov |
| Earlier package (`v5.1b`, submission 1050185) | 0.8198 | 1,287 s |
| Final upload (arm c) | not recorded here | Qualified on public data, see [the decision](evidence/wave_20nov/arm_c_sprint_21nov.md). Official rankings stay provisional until the organisers' reproducibility audit. |

Development numbers use ten held-out PROTEUS people and two returning people. They are reading aids with wide intervals (about ±0.5 percentage points across people), not private-set predictions. Where a claim depends on a validity caveat, [docs/STATE.md](docs/STATE.md) says so, including the 18 Nov scoring audit and the 20 Nov red-team finding that some returning-person levels were inflated.

## Approach in brief

The decoder is a routed ensemble, not a single network:

- **Members:** several EEG architectures (for example FBCNet-style, shallow ConvNet-style and larger pretrained-backbone members) trained on PROTEUS and on additional public corpora, each declared in the ledger. The Scherer corpus was cleared for use by the organisers on 11 Nov.
- **Per-session adaptation without labels:** Euclidean-alignment style whitening, a session-level EM stage, a count filter, CSP and a causal personalisation step. State resets between sessions.
- **Routing:** a recognition step decides whether a stream looks like a person seen in training, and the stages and weights change accordingly.
- **Native runtime:** a C++/CUDA port of the inference path, built to fit the 3,600 s budget ([native/](native/)).

Every change was pre-registered in [docs/LEDGER.md](docs/LEDGER.md) with a numeric prediction and a refutation rule before it was scored, and was judged inside the full system under shuffled delivery orders. Most ideas were refuted or left undecided, and those records stay in.

## The experiments along the way

The full trail is here, not only the final system: about 3,000 commits since 10 Nov, the ledger with every pre-registered run (mechanism, numeric prediction, refutation rule, then result), the scripts and logs behind each entry under `evidence/`, `studies/`, `experiments/` and the `lane*/` directories, and the failures and closed directions. The original campaign's ledger sits in the history (`git show a5843492:docs/LEDGER.md`). Large result arrays were removed before release, so some entries keep only logs, scripts and summaries ([history note](docs/HISTORY.md)).

## Repository map

| Path | Contents |
|---|---|
| [docs/STATE.md](docs/STATE.md) | Current state, final submission, validity caveats. Start here. |
| [docs/LEDGER.md](docs/LEDGER.md) | The pre-registered experiment ledger |
| [OFFICIAL_CONTRACT.md](OFFICIAL_CONTRACT.md) | Competition rules and data contract, with sources |
| `submissions/` | Single-file `submission.py` packages for each uploaded version. Their SHA-256 values are pinned, so do not edit them. |
| `lean/` | The v5.1b inference source rewritten as small modules, with a bit-exact parity record |
| `cleanslate/` | From-scratch rebuilds of the cap250 system, harness and member families |
| `native/` | C++/CUDA training and inference port, plus vendored SciPy L-BFGS-B |
| `replay/` | Scorer that replays the packaged chain on public data. `python replay/check.py` must print `OK` before chain results are trusted. |
| `studies/`, `experiments/`, `scripts/` | Dated studies, experiment code, data export and Slurm helpers |
| `friday/`, `evidence/`, `lane*/`, `niv/`, `revisit/` | Per-experiment code, logs and result files, indexed from the ledger |
| [docs/HISTORY.md](docs/HISTORY.md) | Note on the one-time history rewrite, and a map from old to new commit hashes |
| `2026-competition/` | Official competition repository as a pinned submodule, unchanged |

## Reproducing

What you need:

1. **Environment.** Python 3.13 and the pins in [requirements/cloud.txt](requirements/cloud.txt) (a macOS arm64 baseline is in `requirements/baseline-macos-arm64.txt`).
2. **Official code.** `git submodule update --init` for `2026-competition/`.
3. **Data.** Download PROTEUS from NEMAR, then build the window export with `python scripts/export_windows.py`. It writes to `data/`, which is not tracked. Replay inputs are checked against [replay/data.sha256](replay/data.sha256).
4. **Check the replay.** `python replay/check.py` reproduces the six reference development scores of the packaged chain. This only guards the v3 chain in recording order. It does not measure private-set performance.

Limits you should know about:

- **Not included:** trained weights, datasets, and generated arrays (`*.npz`, `*.npy`). The final 417 MB `submission.zip` is not stored here. Its SHA-256 is in [docs/STATE.md](docs/STATE.md#final-track-2-submission).
- **Cluster paths:** training and scoring scripts target the Galvani (Tübingen) and Leonardo (CINECA) Slurm clusters. Paths such as `/mnt/lustre/...`, account names and job IDs belong to those systems and need editing for yours.
- **Records, not clean code:** many scripts are one-off experiment drivers kept as the evidence for a ledger entry. Some logs and notes show `<repo>` and `<home>` where a local path was removed before release.
- **Native CPU replay** is not proof that the Linux GPU worker gives the same outputs.

## Data and model licences

Original dataset and model licences remain in force. PROTEUS is distributed through NEMAR under its own terms, and the Scherer corpus under the terms declared in the ledger. Third-party code bundled here keeps its own notice: [native/vendor/scipy_lbfgsb](native/vendor/scipy_lbfgsb) (SciPy) and [native/worker-parity/DUCC-LICENSE.md](native/worker-parity/DUCC-LICENSE.md). The competition submodule is under the organisers' terms.

## Licence

No licence has been chosen for the code in this repository yet. Until a `LICENSE` file is added, all rights are reserved by the author.

## Contact

Open an issue on this repository.
