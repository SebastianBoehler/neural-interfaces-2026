"""Observe the exact §435 inference logits while retaining all original training and checks."""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path('/mnt/lustre/work/gehler/gfh527')
OLD = BASE/'ni26_phase435_20261118'
sys.path.insert(0, str(OLD))
from score379 import source_manifest, sha

assert source_manifest(OLD/'source.sha256') == '96feb485e22bd58d90c154e13bf24a1f9d99469c7ab47b009e1894403cdccf71'
import experiment


def main(held, smoke):
    logits = []

    class Capture(experiment.Decoder):
        def sequence(self, features, *args, **kwargs):
            result = super().sequence(features, *args, **kwargs)
            if not self.training and len(features) > 1:
                logits.append(result[0].detach().cpu().numpy().copy())
            return result

    experiment.Decoder = Capture
    experiment.main(held, smoke)
    assert len(logits) == 2
    if smoke:
        return
    path = Path(held+'.npz')
    proof = json.loads(Path(held+'.json').read_text())
    original = json.loads((OLD/(held+'.json')).read_text())
    assert original['episode_sha'] == proof['episode_sha']
    assert original['train_rows'] == proof['train_rows']
    assert sha(OLD/path) == original['output_sha'] and sha(path) == proof['output_sha']
    with np.load(path) as new, np.load(OLD/path) as old:
        assert np.array_equal(new['index'], old['index'])
        for arm, values in zip(experiment.ARMS, logits):
            assert np.array_equal(values.argmax(1), new[arm]) and np.array_equal(new[arm], old[arm])
        np.savez_compressed(held+'_logits.npz', index=new['index'], **dict(zip(experiment.ARMS, logits)))
    proof['logits'] = sha(Path(held+'_logits.npz'))
    proof['archive'] = original['output_sha']
    Path(held+'.json').write_text(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:] == ['smoke'])
