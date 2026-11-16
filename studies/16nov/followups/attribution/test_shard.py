import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
for name in ('metrics', 'shard_guard'):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    globals()[name] = module

class Score:
    @staticmethod
    def sha256(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    @staticmethod
    def verify_inputs(files, cache, root):
        for name, digest in files.items():
            if Score.sha256(name) != digest: raise ValueError('input mismatch')

class ShardTests(unittest.TestCase):
    def test_missing_array_and_changed_input_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); folder = root / 'runs/s300_shards'; folder.mkdir(parents=True)
            source = root / 'source'; source.write_text('original')
            path = folder / 'within0.npz'; p = np.array([[.5,.3,.2]])
            data = {f'control|{stage}|B2_pool': p for stage in ('V6Fr','count','CP')}
            np.savez_compressed(path, **data)
            manifest = {'order':'within0','smoke':False,'sha256':Score.sha256(path),'keys':sorted(data),
                        'input_sha256':{str(source):Score.sha256(source)}}
            side = path.with_suffix('.json'); side.write_text(json.dumps(manifest))
            shard_guard.load(root,'within0',('control',),('B2_pool',),Score,metrics.probability)
            manifest['keys'].pop(); side.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError): shard_guard.load(root,'within0',('control',),('B2_pool',),Score,metrics.probability)
            manifest['keys']=sorted(data); side.write_text(json.dumps(manifest)); source.write_text('changed')
            with self.assertRaises(ValueError): shard_guard.load(root,'within0',('control',),('B2_pool',),Score,metrics.probability)

if __name__ == '__main__': unittest.main()
