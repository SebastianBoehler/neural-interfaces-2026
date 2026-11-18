# Friday starting point · 18 November 2026

Given one 41-channel EEG window at 120 Hz, predict one of three tasks immediately. The model sees the delivered session prefix, with no labels, person ID or recording-order guarantee. Each participant/session/interface cell contributes equal weight to the official balanced accuracy. All learned session state resets between sessions. The worker has one hour. See the [contract](../OFFICIAL_CONTRACT.md).

| System | Evidence available at checkout creation |
|---|---|
| v5.1b, sealed incumbent | Submission 1050185 scored **0.8197668759** in 1,287 seconds. |
| v6.6cap250, staged Python package | Adds prefix adaptation and personalization. Historical full-system public comparison against v5.1b gained 1.380 percentage points on ten held new people with matched auxiliary seed counts. Some component orders and returning routes differed from the packages. Exact external archive differences were +0.493 and +0.040 points on two order pairs over the same nine people. Both intervals crossed zero. Its sealed result was unknown at this snapshot. |
| C++/CUDA worker | The measured Galvani release reproduced cap250's labels and work counts on 2,655 ordinary shuffled public windows and 1,178 forced-adaptation windows. Timings covered one pair per workload. Source fitting for the complete production ensemble is still incomplete. See [worker source, image and manifest](../native/worker-parity/README.md) and [paired measurements](../evidence/next_wave_16nov/native366_worker.md). |

**Inputs.** The two frozen ZIPs live in `runs/sealed/final_zips/`. Their SHA-256 values and the official-window export hash are in [inputs.sha256](../friday/inputs.sha256). The [exporter](../scripts/export_windows.py) reads the pinned competition pipeline. The local PROTEUS export contains 2,655 windows from ten people and 14 sessions from NEMAR nm000290 v1.0.0. It retains person, session, interface, class, run and onset. See the [source data audit](../evidence/baseline/data_audit.json).

**Checks.** From this checkout run these commands with the existing project Python environment. The first checks both ZIPs, EEG and official revision, loads each real package, and predicts from two fresh sessions on the same public window. The second checks six historical v3 reference scores in recording order. A new shuffled scorer also needs a zero-change reference and a pinned-order check. Neither command measures private performance.

```sh
python scripts/friday_smoke.py
python replay/check.py
```

The prepared Mac checkout can use `../neural-interfaces-2026/.venv/bin/python`. On Galvani, obtain the measured native package and worker image from the pinned paths in the worker notes. The portable CMake sources alone do not reproduce that Linux build. Use the file manifest before running the native package. Recheck live quota, phase, and authorization before any upload.

**Checkout.** The active files are selected by [sparse-paths](../friday/sparse-paths). Existing jobs still use their original paths. The frozen history remains available with `git show cb3e05a:<path>`. To reconstruct this view in a second worktree, apply the profile with `git sparse-checkout set --no-cone --stdin < friday/sparse-paths`, initialize the pinned submodule and provide the hash-matched local ZIPs, EEG export, and replay inputs under `runs/replay_data/`. Their manifest is [replay/data.sha256](../replay/data.sha256).
