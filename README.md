# Friday · Neural Interfaces 2026

**Final Track 2 upload** · the qualified arm c archive (`submission.zip`, 417 MB with trained weights) is not stored in this repository. [SHA-256 and verification](docs/STATE.md#final-track-2-submission)

Decode one four-second EEG window into motor imagery, mental subtraction, or word generation. The subject may be new. The decoder gets no labels and starts fresh each session. Friday's objective is to raise the sealed balanced accuracy within the 3,600 s sealed budget (one RTX PRO 4500).

Read [the current state](docs/STATE.md), [the Friday ledger](docs/LEDGER.md), then [the official contract](OFFICIAL_CONTRACT.md).

## What is in this repository

A research record and the code behind it: the decoder source (`submissions/`, `cleanslate/`, `native/`), the experiment ledger and evidence logs, and the Slurm scripts used on Galvani and Leonardo. Datasets, trained weights and generated arrays (`*.npz`, `*.npy`) are not included. Paths such as `/mnt/lustre/...` and Slurm account names in the scripts refer to the authors' clusters.

The public training data is the PROTEUS release ([NEMAR nm000290](https://doi.org/10.82901/nemar.nm000290)). The sealed evaluation recordings are private and were never available to this project. Original dataset and model licences remain in force; see `native/vendor/` for bundled third-party notices.
