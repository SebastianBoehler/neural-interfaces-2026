# ttt_base: TTT-able base candidates for team C's interface

`ref/` = team C's `submission.py`, `train.py`, `stream_eval.py`, `score.py` vendored from commit 819765ac (HEAD when this
lane started; team C kept editing the originals). Only edits: the repo-root `parents[]` index (one level deeper).
`train.py` here builds a package with the same layout (`weights.pt` members `{arch, kw, seed, state}`, `train.json`,
`submission.py`). `submission.py` of the package = `ref/submission.py` + `archs.py` appended + BatchNorm generalised
(`nn.BatchNorm2d` -> any `_BatchNorm`, 3 places, asserted), so every team C variant (A0..A3f) runs unchanged on it.
Drop-in for team C: copy the package's `submission.py` (adds arch `bd` = braindecode net with `.fc`) and `weights.pt`.
