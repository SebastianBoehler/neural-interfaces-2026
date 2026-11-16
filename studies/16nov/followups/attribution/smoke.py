"""Verify recomputed control filter against pinned original CP input."""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import capture
import metrics
OLD = Path('/mnt/lustre/work/gehler/gfh527/ni26_lanes292_20261116/lane1/chain')
filtered = capture.original_filtered(OLD, 'within0', 'B2_pool')
import package as K
with np.load(OLD / 'runs/chainout/s281_v66p_control_within0.npz') as z:
    raw = z['B2_pool|V6Fr'].astype(np.float64)
hc = K.J.hyp_cache('within0', 'wprior', K.X, K.LAM, sessions=K.SESS, proto=K.PROTO)
recomputed = K.gate(raw, K.J.filt_joint(raw, hc, 'urn'), 'within0', K.latch('within0'))
metrics.probability(raw); metrics.probability(recomputed)
if not np.array_equal(recomputed, filtered): raise ValueError('original CP filtered input differs')
print('OK: pinned original CP filtered control input matches bitwise; valid probabilities', flush=True)
