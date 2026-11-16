import unittest
from pathlib import Path

from score import patch_scorer


class ScorerTests(unittest.TestCase):
    def test_patch_reuses_existing_guarded_scorer(self):
        root = Path(__file__).resolve().parents[4]
        source = (root / 'studies/14nov/newmember/score_s_v66p.py').read_text()
        patched = patch_scorer(source)
        self.assertIn("'geo_zero'", patched)
        self.assertIn('def verify_zero(', patched)
        self.assertIn('def load_output(', patched)
        compile(patched, '<patched-score>', 'exec')
        with self.assertRaises(ValueError):
            patch_scorer(patched)


if __name__ == '__main__':
    unittest.main()
