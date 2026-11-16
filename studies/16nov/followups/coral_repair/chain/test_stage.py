import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('stage304', Path(__file__).with_name('stage.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class StageTests(unittest.TestCase):
    def test_exact_matrix_and_changed_fold_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root / 'source'; (source / 'runs/coral304').mkdir(parents=True)
            (root / 'target/data').mkdir(parents=True); (root / 'target/runs').mkdir()
            (source / 'coral304.sha256').write_text('source')
            (root / 'target/data/proteus_windows.npz').write_text('data')
            (root / 'target/runs/shuffle_perms.npz').write_text('permutation')
            hashes = {}
            for arm in m.PRODUCERS:
                for seed in range(6):
                    for person in range(1,11):
                        name = f'{arm}_s{seed}_P{person:02d}.npz'; p = source / 'runs/coral304' / name
                        p.write_text(name); hashes[name] = m.sha(p)
            v = {'counts': {'folds':240,'streams':192}, 'producer_hashes':hashes,
                 'stream_hashes':{n:'x' for n in m.expected_names()},
                 'producer_manifest_sha256':m.sha(source / 'coral304.sha256'),
                 'data_sha256':m.sha(root / 'target/data/proteus_windows.npz'),
                 'permutation_sha256':m.sha(root / 'target/runs/shuffle_perms.npz')}
            previous = m.ROOT
            try:
                m.ROOT = root / 'target'; m.verify_validation(v,source)
                victim = next(iter(hashes)); (source / 'runs/coral304' / victim).write_text('changed')
                with self.assertRaises(ValueError): m.verify_validation(v,source)
                v['producer_hashes'].pop(victim)
                with self.assertRaises(ValueError): m.verify_validation(v,source)
            finally: m.ROOT = previous

if __name__ == '__main__': unittest.main()
