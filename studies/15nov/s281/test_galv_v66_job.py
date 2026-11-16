"""Slurm task mapping and child-failure checks; no scientific workload."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PATH = Path(__file__).with_name('galv_v66_job.py')


class TaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PATH.exists():
            raise AssertionError('Galvani task driver is missing')
        spec = importlib.util.spec_from_file_location('leo_v66_job', PATH)
        cls.m = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.m)

    def test_arrays_cover_each_registered_arm_and_order_once(self):
        for phase, count, arms in (('control',8,('control',)), ('treatment',48,('zero_coral','S1','S2','S3','S4','S5')),
                                   ('cp',56,('control','zero_coral','S1','S2','S3','S4','S5'))):
            pairs = [self.m.mapping(phase, i) for i in range(count)]
            self.assertEqual(set(pairs), {(a,o) for a in arms for o in self.m.ORDERS})
            self.assertEqual(len(set(pairs)), count)
            with self.assertRaises(ValueError): self.m.mapping(phase, count)
            with self.assertRaises(ValueError): self.m.mapping(phase, -1)
        with self.assertRaises(ValueError): self.m.mapping('unknown', 0)

    def test_registered_recipes_map_to_the_right_arm(self):
        self.assertEqual(self.m.RECIPES['control'], ('none','none','0'))
        self.assertEqual(self.m.RECIPES['zero_coral'], ('fb_coral','none','1'))
        self.assertEqual(self.m.RECIPES['S1'], ('fb_coral','none','0'))
        self.assertEqual(self.m.RECIPES['S2'], ('fb_am','none','0'))
        self.assertEqual(self.m.RECIPES['S3'], ('fb_coral,fb_am','none','0'))
        self.assertEqual(self.m.RECIPES['S4'], ('none','fb_coral','0'))
        self.assertEqual(self.m.RECIPES['S5'], ('none','fb_am','0'))

    def test_cap_boundary_requires_each_session_below_250(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'bases.npz'
            np.savez(path, **{'sess|all_loso':np.repeat([0,1],[199,249])})
            self.m.check_session_lengths(path)
            np.savez(path, **{'sess|all_loso':np.repeat([0,1],[199,250])})
            with self.assertRaisesRegex(ValueError, '250'):
                self.m.check_session_lengths(path)

    def test_private_inputs_reject_symlinks_and_hardlinks(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve(); original=root/'source'; original.write_bytes(b'x')
            symlink=root/'sym'; symlink.symlink_to(original)
            hardlink=root/'hard'; os.link(original, hardlink)
            with self.assertRaisesRegex(ValueError, 'links'):
                self.m.require_private(symlink, root)
            with self.assertRaisesRegex(ValueError, 'links'):
                self.m.require_private(hardlink, root)

    def test_tree_check_rejects_a_linked_unlisted_input(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve(); tree = root/'runs'; tree.mkdir()
            (tree/'visible').write_bytes(b'x')
            self.m.check_private_tree(tree, root)
            (tree/'hidden_link').symlink_to(tree/'visible')
            with self.assertRaisesRegex(ValueError, 'links'):
                self.m.check_private_tree(tree, root)

    def test_all_children_are_waited_for_and_failure_propagates(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            cmds = [(str(i), [sys.executable, '-c', f'print({i}); raise SystemExit({2 if i == 1 else 0})']) for i in range(3)]
            with self.assertRaisesRegex(RuntimeError, 'child'):
                self.m.run_children(cmds, directory, {})
            for i in range(3):
                self.assertEqual((directory / f'{i}.log').read_text().strip(), str(i))
            with self.assertRaises(FileExistsError):
                self.m.run_children(cmds, directory, {})


if __name__ == '__main__':
    unittest.main()
