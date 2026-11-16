import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

class PatchTests(unittest.TestCase):
    def test_only_registered_recipes_change_and_original_is_untouched(self):
        spec = importlib.util.spec_from_file_location('fbl_patch', HERE / 'patch.py')
        p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
        source = (HERE.parent / 'system/recipes.py').read_text()
        changed = p.patch(source)
        self.assertEqual(source[source.index('ZERO ='):], changed[changed.index('ZERO ='):])
        with tempfile.TemporaryDirectory() as d:
            module = Path(d) / 'recipes.py'; module.write_text(changed)
            spec = importlib.util.spec_from_file_location('fbl_recipes', module)
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            self.assertEqual(set(m.ARMS), {'FBL05','FBL15','FBL2','FBL3'})
            self.assertEqual([m.ARMS[n][3] for n in ('FBL05','FBL15','FBL2','FBL3')], [.5,1.5,2.,3.])
            extra = [('sc_fb',1.),('sc_fbl',2.),('sc_sinc',3.),('sc_fbraw',4.)]
            self.assertEqual(dict(m.modify_extra(extra,m.ARMS['FBL3']))['sc_fbl'],6.)

if __name__ == '__main__': unittest.main()
