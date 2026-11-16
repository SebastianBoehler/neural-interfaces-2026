import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).with_name('recipes.py')


class RecipeTests(unittest.TestCase):
    def test_fifteen_exact_registered_arms_and_matched_weights(self):
        self.assertTrue(PATH.exists(), 'registered recipes are not implemented')
        spec = importlib.util.spec_from_file_location('recipes', PATH)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertEqual(len(module.ARMS), 15)
        self.assertEqual(module.ARMS['E05CS'], ('eegn', .5, 'swap', 1.0))
        self.assertEqual(module.ARMS['D1CA'], ('eegn_dist2', 1., 'add', 1.0))
        self.assertEqual(module.ARMS['FBL15'], ('none', 0., 'none', 1.5))

    def test_swap_add_and_fbl_use_original_weights_independently(self):
        self.assertTrue(PATH.exists(), 'registered recipes are not implemented')
        spec = importlib.util.spec_from_file_location('recipes', PATH)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        extra = [('other', 2.), ('sc_fb', 1.), ('sc_fbl', 2.), ('sc_sinc', 3.), ('sc_fbraw', 4.)]
        out = dict(module.modify_extra(extra, ('eegn', .5, 'swap', 1.5)))
        self.assertNotIn('sc_fb', out)
        self.assertEqual(out['nm_fb_coral10'], 1.)
        self.assertEqual(out['nm_eegn'], 1.25)
        self.assertEqual(out['sc_fbl'], 3.)
        self.assertEqual(out['other'], 2.)
        self.assertEqual(extra[1], ('sc_fb', 1.))

    def test_zero_added_slot_and_each_ablation_preserves_other_knobs(self):
        self.assertTrue(PATH.exists(), 'registered recipes are not implemented')
        spec = importlib.util.spec_from_file_location('recipes', PATH)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        extra = [('sc_fb', 1.), ('sc_fbl', 2.), ('sc_sinc', 3.), ('sc_fbraw', 4.)]
        full = dict(module.modify_extra(extra, ('eegn', 1., 'swap', 1.5)))
        no_eeg = dict(module.modify_extra(extra, ('none', 0., 'swap', 1.5)))
        no_swap = dict(module.modify_extra(extra, ('eegn', 1., 'none', 1.5)))
        no_fbl = dict(module.modify_extra(extra, ('eegn', 1., 'swap', 1.)))
        self.assertNotIn('nm_eegn', no_eeg)
        self.assertEqual(no_eeg['nm_fb_coral10'], full['nm_fb_coral10'])
        self.assertEqual(no_swap['sc_fb'], 1.)
        self.assertEqual(no_swap['nm_eegn'], full['nm_eegn'])
        self.assertEqual(no_fbl['sc_fbl'], 2.)
        self.assertEqual(no_fbl['nm_eegn'], full['nm_eegn'])
        zero = dict(module.modify_extra(extra, ('eegn', 0., 'none', 1.)))
        self.assertEqual(zero['nm_eegn'], 0.)
        self.assertTrue(all(zero[n] == w for n, w in extra))


if __name__ == '__main__':
    unittest.main()
