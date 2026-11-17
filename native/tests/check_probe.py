"""Compare native component output to source-only frozen Python reference."""
import sys
import torch
from torch.jit._pickle import restore_type_tag

reference = torch.load(sys.argv[1], weights_only=True)
with torch.serialization.safe_globals([restore_type_tag]):
    native = torch.load(sys.argv[2], weights_only=True)
for key in ('p1', 'p2', 'tail', 'reapply', 'features'):
    actual, expected = native[key], reference[key]
    if actual.shape != expected.shape or not torch.isfinite(actual).all():
        raise ValueError(f'invalid native output: {key}')
    error = (actual-expected).abs().max().item()
    # Declared before the first native source run; eigen/BLAS roundoff is expected.
    limit = 1e-4 if key == 'features' else 1e-7
    changed = 0 if key == 'features' else int((actual.argmax(-1) != expected.argmax(-1)).sum())
    print(f'{key}: max_abs={error:.12g}, limit={limit:g}, changed_labels={changed}')
    if error > limit or changed:
        raise ValueError(f'native source parity failed: {key}')
print('OK: source component parity; no accuracy or complete-worker claim')
