"""Recipe and input provenance guards for the §281 chain runner."""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).with_name('s281_provenance.py')


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(PATH.is_file(), 'provenance module missing')
        spec = importlib.util.spec_from_file_location('s281_provenance', PATH)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_registered_recipes_and_distinct_zero(self):
        recipes = (
            ('control', 'none', 'none', ''),
            ('S1', 'fb_coral', 'none', ''),
            ('S2', 'fb_am', 'none', ''),
            ('S3', 'fb_coral,fb_am', 'none', ''),
            ('S4', 'none', 'fb_coral', ''),
            ('S5', 'none', 'fb_am', ''),
            ('zero_coral', 'fb_coral', 'none', '1'),
        )
        for name, add, swap, zero in recipes:
            with self.subTest(name=name):
                recipe = self.module.validate_recipe(name, add, swap, zero, '0,1,2,3,4,5')
                self.assertEqual(recipe['name'], name)

    def test_mislabel_and_missing_seeds_are_rejected(self):
        bad = (('S1', 'fb_am', 'none', ''), ('S3', 'fb_am,fb_coral', 'none', ''),
               ('S1', 'fb_coral', 'none', '1'), ('zero_coral', 'fb_coral', 'none', ''),
               ('control', 'fb_coral', 'none', ''), ('S6', 'fb_am', 'none', ''))
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.module.validate_recipe(*args, '0,1,2,3,4,5')
        with self.assertRaises(ValueError):
            self.module.validate_recipe('S1', 'fb_coral', 'none', '', '0')

    def test_isolated_runner_rejects_mislabel_before_input_lookup(self):
        env = dict(os.environ, NM_NAME='S1', ADD='fb_am', SWAP='none')
        env.pop('ZERO', None)
        runner = PATH.with_name('run_s_v66.py')
        result = subprocess.run([sys.executable, '-I', str(runner), 'within999'], env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('recipe', result.stderr)

    def test_input_hashes_are_verified_and_manifest_is_exclusive(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); f = d / 'stream.npz'; f.write_bytes(b'original')
            tracker = self.module.InputTracker(); tracker.track(f)
            result = tracker.manifest(); self.assertEqual(len(result), 1)
            manifest = d / 'manifest.json'
            self.module.write_manifest(manifest, {'inputs': result})
            with self.assertRaises(FileExistsError):
                self.module.write_manifest(manifest, {'inputs': result})
            f.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                tracker.manifest()


if __name__ == '__main__':
    unittest.main()
