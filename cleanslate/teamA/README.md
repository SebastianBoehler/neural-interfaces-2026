# Team A clean-slate EEG decoder

Train a single pretrained encoder end to end. The package takes one 41-channel window and returns one label at a time.
The worker reads only `submission.py`, `weights.pt`, and the open-source notice where applicable.

## Build and evaluate

On a Slurm CUDA allocation with the competition checkout on `PYTHONPATH`:

```sh
python cleanslate/teamA/train.py --people P01,P02,P03,P04,P05,P06,P07,P08,P09,P10 \
    --data /path/to/proteus_windows.npz --out /path/to/package
```

The default recipe uses REVE-base, the seven Lane H corpora, Shin 2018 and Scherer. The required files must already
be staged. Pass `--external-dir` for a separate export root. Use `--external` with the preregistered corpus list when
including the B6 exports. A checkpoint override must match the pinned SHA-256.
The MANAS-2 build also requires `positions.json` and `LICENSE` from its pinned revision in `--assets`.

Use `--first-session-only P04` when holding that person's later visits out. To share only the external fit across
folds, run `--external-fit` once and give each fold that run's `weights.pt` via `--transfer`. Neither operation
reads held labels for learning. For NI-V, retrain the external checkpoint with Scherer omitted from the corpus
list before replaying its visits. The common harness reads the actual exported Solver.

The weights package contains an immutable source EEG gallery and first-visit personal heads. Inference stores the
alignment, feature-normalization state, pseudo-label updates, and recognition state on its session object.
The official deep-copy/reset path clears these between visits. The external head and external data stay on the cluster.

[Run registrations and results](../../evidence/wave_20nov/cleanslate_teamA.md)

The A2 research form uses `--arch manas --recipe A2`. Its full external list adds `ds007221g,ds007221lr,netbci`;
the loader checks all three B6 participant counts before fitting. The train command stores input and checkpoint hashes
with the package so fold provenance can be checked independently of the Slurm logs.
