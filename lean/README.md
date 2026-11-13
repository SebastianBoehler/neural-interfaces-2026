# lean: the inference source of the best package, by responsibility

`lean/src/` is the inference code of `cand_v51b_own` (sealed 0.8198) rewritten as ten small modules that hold only
what that package executes. `lean/tools/bundle.py` turns the tree into the single file a package may contain,
`submissions/lean_v5/submission.py`. With the unchanged data files of `cand_v51b_own`, the bundled file gives the
same outputs as `submissions/ensemble_v5b/submission.py`, bit for bit on CPU ([PARITY.md](PARITY.md)).
What was removed and what was kept, with reasons: [INVENTORY.md](INVENTORY.md). What an audit of the training
side would still need: [AUDIT_TRAINING_SIDE.md](AUDIT_TRAINING_SIDE.md). Evidence files: `lean/evidence/`.

## Layout

| Module (bundle order) | Holds |
|---|---|
| `numerics.py` | matrix functions by eigendecomposition, softmax, LDA posterior, zero-phase filtering without per-call setup |
| `tangent.py` | `TangentCore`: core1 and the Hilbert core |
| `inputs.py` | montage mapping, sanitising of the harness batch, `InputStream` (per-session whitening or global scale) |
| `nets.py` | `Shallow`, `FBC`, `Jit` (teacher), `SJit` and `SharedFIRSJit` (SessionBN traces), `NetBank` (grouping and mix order) |
| `stages.py` | `Rebalance`, `SessionEM`, `TSEM`, and `Tail` = the three in a chain |
| `gates.py` | `ContrastGate` (class directions, from window 60), `IdentityGate` (spectral shape, from window 10) |
| `personal.py` | `OwnStage`: the three own-space members per public person |
| `routing.py` | `Route`: gate decision, personal head into the mix, routed tail, own-space stage, per-session log line |
| `ensemble.py` | `Ensemble`: member mix, unrouted tail, routing; the worker thread for the cores |
| `solver.py` | `Solver.load_model` and `check_package` |

Per window: `Ensemble.step` computes the cores (worker thread) and the nets, mixes them, runs the unrouted `Tail`
(its output and the 256-PC view feed the gates), then `Route.step`, whose output is the answer.

## Rules of the tree

- **One file per package.** Modules import each other with relative imports (`from .numerics import softmax`), only
  from modules earlier in `MODULES` (`lean/tools/bundle.py`). Top-level names are unique across the tree. The
  bundler checks both and fails otherwise.
- **Never edit `submissions/lean_v5/submission.py`.** Edit the tree, then bundle. `bundle.py --check` and the
  builder fail when the file is stale.
- **No silent configuration.** The code implements one configuration. `check_package` compares the flags in
  `weights.npz` with it at load time and raises on a mismatch; an unsupported net export raises in `build_net`.
- **Per-window interface.** Every member and stage has `reset_state()` and a per-window `step` (or `__call__` /
  `proba` for inputs and nets). The evaluator sends one window per `predict` call; a larger batch is handled
  window by window.
- `Ensemble.trace` holds the last window's internals (mix, core1, view, unrouted and final posterior, both gates'
  decisions, the stage). The parity tools read it; nothing else depends on it.

## Commands

`MAIN` is the main checkout (data, environment and the package's data files live there). Local jobs go through
`scripts/run_capped.sh`, one at a time.

```bash
PY=$MAIN/.venv/bin/python
python3 lean/tools/bundle.py                      # lean/src -> submissions/lean_v5/submission.py
python3 lean/tools/bundle.py --check              # is the bundled file current?
scripts/run_capped.sh LOG $PY lean/tools/build.py $MAIN/runs/sealed/cand_v51b_own runs/lean/pkg
D="--public $MAIN/data/proteus_windows.npz --external $MAIN/data/scherer2015/windows.npz"
scripts/run_capped.sh LOG $PY lean/tools/record.py $MAIN/runs/sealed/cand_v51b_own runs/lean/orig.npz $D
scripts/run_capped.sh LOG $PY lean/tools/record.py runs/lean/pkg runs/lean/lean.npz $D
scripts/run_capped.sh LOG $PY lean/tools/parity.py runs/lean/orig.npz runs/lean/lean.npz --out runs/lean/parity.json
scripts/run_capped.sh LOG $PY lean/tools/timing.py $MAIN/runs/sealed/cand_v51b_own runs/lean/pkg \
    --public $MAIN/data/proteus_windows.npz --out runs/lean/timing.json
```

`record.py` takes `--cases a,b` to run a subset (a full recording takes about 10 minutes on the Mac). In zsh,
pass `$D` as an array or as `${=D}`. The official replay (`scripts/replay_package.py PKG OUT cpu`, with
`BENCHOPT_DATA_HOME=$MAIN/data` and `$MAIN/.venv/bin` on `PATH`) runs the warm-up phase's ingestion and scoring
programs on a package directory; in a fresh worktree the `2026-competition` submodule must be initialised first.

## Adding a feature

1. Put it in the module that owns the responsibility, or add a module and list it in `MODULES` after the modules
   it imports from.
2. New data goes into `weights.npz` (or a new data file) under new keys; add the key to `check_package` if the code
   depends on a flag.
3. Bundle, build a package directory with the new data files, and record it. `parity.py` against the recording of
   the previous package shows exactly which windows and which internal values changed; with the feature switched
   off by its data, the recording must stay bitwise equal.
