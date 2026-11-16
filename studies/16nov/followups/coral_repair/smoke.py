"""Real GPU activity and unchanged-wrapper parity; never score held labels."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from produce import ROOT, SB, infer, install, make, protocol, weights


def main():
    d = protocol.load(); held, train, test = list(protocol.folds(d, 'all_loso'))[0]
    fit = np.flatnonzero(train)
    start = time.monotonic()
    original = make(d, 'unconditional10', 0, epochs=1)
    original.fit(fit); original.set_k(2.)
    expected, baseline_hash = infer(original, d, test)
    install(SB)
    wrapped = make(d, 'unconditional10', 0, epochs=1)
    wrapped.fit(fit); wrapped.set_k(2.)
    actual, wrapped_hash = infer(wrapped, d, test)
    if baseline_hash != wrapped_hash or any(not np.array_equal(expected[k], actual[k]) for k in expected):
        raise ValueError('unchanged unconditional wrapper altered training or stream outputs')
    del original, wrapped
    torch.cuda.empty_cache()
    conditional = make(d, 'conditional10', 0, epochs=1)
    conditional.fit(fit); conditional.set_k(2.)
    if (conditional.repair_stats['gradient_norm'] <= 0 or
            min(conditional.repair_stats['class_pairs']) <= 0 or
            conditional.dg_stats['coral_pen'] <= 0):
        raise ValueError('conditional training mechanism inactive')
    output, conditional_hash = infer(conditional, d, test)
    if conditional_hash == wrapped_hash or all(np.array_equal(expected[k], output[k]) for k in expected):
        raise ValueError('conditional mechanism made no change')
    if set(conditional.fit_subjects) & set(d['subject'][test]):
        raise ValueError('held person entered training')
    report = dict(held=held, epochs=1, seed=0, original_wrapper_bitwise=True,
                  all_eight_order_prefix_reset=True, stats=conditional.repair_stats,
                  penalty=conditional.dg_stats, seconds=time.monotonic()-start,
                  gpu=torch.cuda.get_device_name(0), metrics_computed=False)
    path = ROOT / 'runs/coral304/smoke.json'; path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(report, f, sort_keys=True, indent=2)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
