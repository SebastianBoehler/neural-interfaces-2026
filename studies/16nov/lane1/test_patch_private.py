import importlib.util
import ast
import unittest
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('lane1_patch', HERE / 'patch_private.py')


class PrivatePatchTests(unittest.TestCase):
    def test_recipes_are_exact_and_include_six_matched_arms(self):
        self.assertTrue(SPEC.loader is not None)
        self.assertTrue((HERE / 'patch_private.py').exists(), 'private patcher not implemented')
        module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(module)
        recipes = module.RECIPES
        self.assertEqual(recipes['E'], ('eegn', 'none', False))
        self.assertEqual(recipes['D'], ('eegn_dist2', 'none', False))
        self.assertEqual(recipes['C3S'], ('none', 'fb_coral3', False))
        self.assertEqual(recipes['C10S'], ('none', 'fb_coral10', False))
        self.assertEqual(len(recipes), 8)

    def test_guarded_replacement_rejects_missing_and_duplicate_anchors(self):
        self.assertTrue((HERE / 'patch_private.py').exists(), 'private patcher not implemented')
        module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(module)
        self.assertEqual(module.replace_once('a x b', 'x', 'y'), 'a y b')
        for src in ('a b', 'x x'):
            with self.assertRaises(ValueError):
                module.replace_once(src, 'x', 'y')

    def test_actual_sources_patch_and_parse(self):
        module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(module)
        root = HERE.parents[2] / 'studies/14nov/newmember'
        files = ('s281_provenance.py', 'run_s.py', 's281_cp.py', 'score_s_v66p.py')
        patched = module.patch_sources({name: (root / name).read_text() for name in files})
        for name, source in patched.items():
            with self.subTest(name=name):
                ast.parse(source)
        self.assertIn("('D','E')", patched['score_s_v66p.py'])
        self.assertIn('lane1_inputs/streams.sha256', patched['score_s_v66p.py'])
        self.assertIn('fb_coral10', patched['run_s.py'])
        with self.assertRaises(ValueError):
            module.patch_sources(patched)

    def test_cell_weighted_summary_uses_cell_counts_not_equal_people(self):
        module = importlib.util.module_from_spec(SPEC)
        SPEC.loader.exec_module(module)
        root = HERE.parents[2] / 'studies/14nov/newmember'
        files = ('s281_provenance.py', 'run_s.py', 's281_cp.py', 'score_s_v66p.py')
        source = module.patch_sources({name: (root / name).read_text() for name in files})['score_s_v66p.py']
        matches = [x for x in ast.parse(source).body
                   if isinstance(x, ast.FunctionDef) and x.name == 'cell_weighted_summary']
        self.assertEqual(len(matches), 1, 'cell-weighted summary is not implemented')
        node = matches[0]
        ns = {'np': np}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<test>', 'exec'), ns)
        result = ns['cell_weighted_summary'](np.array([[0., 4.]] * 3),
                                              ('within0', 'full0', 'runs0'),
                                              np.array([1, 3]), bootstrap=10)
        self.assertEqual(result['mean'], 3.0)


if __name__ == '__main__':
    unittest.main()
