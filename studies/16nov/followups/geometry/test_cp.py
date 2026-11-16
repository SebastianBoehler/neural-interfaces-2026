import unittest
from pathlib import Path

from cp import patch_source


class PersonalizationWrapperTests(unittest.TestCase):
    def test_private_source_patches_are_guarded_and_add_probability_checks(self):
        src = '''assert arm in {"control", "zero_coral", "S1", "S2", "S3", "S4", "S5"}
    tracker = InputTracker()
    assert raw.shape == (K.C.N, 3)
    filtered = K.gate(raw, K.J.filt_joint(raw, hc, "urn"), order, lat)
'''
        out = patch_source(src)
        self.assertIn('geo_d50', out)
        self.assertIn('raw >= 0', out)
        self.assertIn('filtered >= 0', out)
        with self.assertRaises(ValueError):
            patch_source(src + src)
        root = Path(__file__).resolve().parents[4]
        source = (root / 'studies/14nov/newmember/s281_cp.py').read_text()
        compile(patch_source(source), '<actual-cp>', 'exec')


if __name__ == '__main__':
    unittest.main()
