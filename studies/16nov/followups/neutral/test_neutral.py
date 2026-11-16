import importlib.util
import unittest
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('neutral', Path(__file__).with_name('transform.py'))
m = importlib.util.module_from_spec(spec)

class NeutralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): spec.loader.exec_module(m)
    def test_output_is_exact_uniform_and_independent_of_input(self):
        a=np.array([[.9,.05,.05],[.1,.2,.7]])
        b=np.array([[.1,.2,.7],[.9,.05,.05]])
        one=m.apply(a); two=m.apply(b)
        np.testing.assert_array_equal(one,two)
        np.testing.assert_array_equal(one,np.full_like(a,1./3.))
    def test_invalid_input_fails(self):
        with self.assertRaises(ValueError): m.apply(np.array([[np.nan,.4,.6]]))

if __name__ == '__main__': unittest.main()
