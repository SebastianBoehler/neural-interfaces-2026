"""Lane R4 wrapper around ret_worker.py for the arms that keep CSP active (R2-R5). With CSP fused after recognition the package's `scores`
are log-scores (count-filter log posterior + cap * log CSP posterior; the label is their argmax), not a normalised posterior, so the pinned inner
worker's "tape sums to 1" check raised. The labels were never affected. Fix, loud and narrow: the inner worker source is re-compiled with one inserted
statement that softmax-normalises the score rows of a session only if its CSP fused windows (model.cp.fused > 0 and not dead); everything else (the
finite check, the labels == argmax check, every other assertion) stays as it was, and a bad tape of any other session still raises. Printed per session."""
import importlib.util
import sys
from pathlib import Path

import dose_worker as D
import inputs as I
import ret_worker as R

OLD = "    tape = {k: np.asarray(v) for k, v in tape.items()}\n"
NEW = OLD + ('    if model.cp is not None and model.cp.fused > 0 and not model.cp.dead:  # lane R4 wrapper: fused CSP scores are log-scores\n'
             '        _s = tape["scores"].astype(np.float64); _s = np.exp(_s - _s.max(1, keepdims=True))\n'
             '        tape["scores"] = _s / _s.sum(1, keepdims=True)\n'
             '        print("[r4-norm] softmax of fused log-scores, rows", len(_s), "fused", model.cp.fused, file=__import__("sys").stderr, flush=True)\n')


def native_worker():
    p = Path(I.__file__).with_name("worker.py")
    src = p.read_text()
    assert src.count(OLD) == 1
    m = importlib.util.module_from_spec(importlib.util.spec_from_file_location("native_inner_worker", p))
    exec(compile(src.replace(OLD, NEW), str(p), "exec"), m.__dict__)
    return m


D.native_worker = R.native_worker = native_worker
if __name__ == "__main__":
    R.main(sys.argv[1], sys.argv[2], "--smoke" in sys.argv[3:])
