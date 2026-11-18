"""Move the locked Wang outputs to a common average across the 41 inputs."""
from pathlib import Path
import json
import sys

import numpy as np

SOURCE = Path('../ni26_wang429_v1_20261118')


def main():
    proofs = []
    for tag in ['smoke'] + [f'sub-{p:02d}' for p in range(2,8)]:
        with np.load(SOURCE / (tag+'.npz')) as z:
            original, names, ranks = z['x'], z['names'], z['ranks']
        ref = names.tolist().index('FCz')
        assert np.all(original[:,:,ref] == 0)
        x = (original - original.mean(axis=2, keepdims=True, dtype=np.float64)).astype(np.float32)
        err = np.max(np.abs((x - x[:,:,ref:ref+1]).astype(np.float64) - original))
        assert err < 5e-5 and x.std(-1).min() > 1e-5
        proofs.append(dict(tag=tag, max_reference_reconstruction_error_uv=float(err)))
        np.savez_compressed(tag+'.npz', x=x, names=names, ranks=ranks)
        (Path(tag+'.json')).symlink_to(SOURCE/(tag+'.json'))
    Path('reference_proof.json').write_text(json.dumps(proofs))


if __name__ == '__main__':
    main()
