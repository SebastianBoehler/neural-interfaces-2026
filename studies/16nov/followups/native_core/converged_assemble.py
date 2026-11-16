"""Assemble §305 arms with the existing complete-matrix provenance checks."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from converged import replace_once


def patch(source):
    changes = (
        ("ARMS = ('Rzero', 'Nfixed', 'Ntrain')", "ARMS = ('Rzero', 'Nfixed', 'Rconverged', 'Nconverged')"),
        ("'runs/s301_native/deltas'", "'runs/s305_converged/deltas'"),
        ("f'runs/s301_native_shards/{person}.npz'", "f'runs/s305_converged_shards/{person}.npz'"),
        ("source_sha = sha(ROOT / 'producer.sha256')", "source_sha = sha(ROOT / 'producer.sha256')\n    converged_sha = sha(ROOT / 'converged.sha256')"),
        ("meta['source_sha256'] != source_sha or meta['input_sha256'] != input_hash or", "meta['source_sha256'] != source_sha or meta['converged_source_sha256'] != converged_sha or meta['input_sha256'] != input_hash or"),
        ("abs(meta['native_fit_loss'] - meta['reference_loss']) > 1e-5 or meta['max_gradient'] > 1e-5):",
         "abs(meta['native_fit_loss'] - meta['strict_loss']) > 1e-5 or max(meta['max_gradient'], meta['strict_max_gradient']) > 1e-5 or set(meta['strict_vs_native_by_order']) != set(ORDERS) or any(not np.isfinite([v['probability_max_abs'], v['logit_max_abs']]).all() for v in meta['strict_vs_native_by_order'].values())):"),
        ("for arm in ('Nfixed', 'Ntrain'):", "for arm in ('Nfixed', 'Rconverged', 'Nconverged'):"),
        ("                    arrays[arm, order][rows] = value", "                    if arm == 'Nfixed' and np.max(np.abs(value), initial=0) > 1e-5:\n                        raise ValueError(f'{person}/{order}: fixed native parity differs')\n                    arrays[arm, order][rows] = value"),
        ("'producer_source_sha256': source_sha}", "'producer_source_sha256': converged_sha}"),
        ("'producer_source_sha256': source_sha,", "'producer_source_sha256': converged_sha,"),
        ("assembled 24 locked native member deltas", "assembled 32 locked converged member deltas"),
    )
    for old, new in changes:
        source = replace_once(source, old, new)
    return source


def main():
    path = Path(__file__).with_name('assemble.py')
    exec(compile(patch(path.read_text()), str(path), 'exec'), {'__name__': '__main__', '__file__': str(path)})


if __name__ == '__main__':
    main()
