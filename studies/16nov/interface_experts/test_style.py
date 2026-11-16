import unittest
import numpy as np
import style


class StyleTest(unittest.TestCase):
    def test_per_window_and_reference_invariance(self):
        rng = np.random.default_rng(11); x = rng.normal(size=(2, 41, 480))
        expected = style.features(x)
        np.testing.assert_allclose(expected[:1], style.features(x[:1]), atol=1e-12, rtol=0)
        common = rng.normal(size=(2, 1, 480))
        np.testing.assert_allclose(expected, style.features(x + common), atol=1e-11, rtol=0)
        with self.assertRaises(ValueError): style.features(x[:, :, :-1])


if __name__ == '__main__': unittest.main()
