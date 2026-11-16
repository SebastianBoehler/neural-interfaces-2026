import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


class ChainPatchTests(unittest.TestCase):
    def test_patch_is_stable_across_python_hash_seeds(self):
        script = """import hashlib, importlib.util
from pathlib import Path
root=Path(%r)
def load(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
first=load(root/'studies/16nov/lane1/patch_private.py','first')
second=load(root/'studies/16nov/followups/coral_repair/chain/patch.py','second')
names=('s281_provenance.py','run_s.py','s281_cp.py','score_s_v66p.py')
source={n:(root/'studies/14nov/newmember'/n).read_text() for n in names}
print(hashlib.sha256(second.patch_sources(first.patch_sources(source))['s281_cp.py'].encode()).hexdigest())
""" % str(ROOT)
        hashes = []
        for seed in ('1','2'):
            env = dict(os.environ, PYTHONHASHSEED=seed)
            hashes.append(subprocess.check_output([sys.executable, '-c', script], env=env, text=True).strip())
        self.assertEqual(hashes[0], hashes[1])

    def test_eight_matched_recipes_and_original_guards(self):
        first = load(ROOT / 'studies/16nov/lane1/patch_private.py', 'lane1_patch')
        new = load(HERE / 'patch.py', 'coral304_patch')
        names = ('s281_provenance.py','run_s.py','s281_cp.py','score_s_v66p.py')
        initial = {n:(ROOT / 'studies/14nov/newmember' / n).read_text() for n in names}
        before = first.patch_sources(initial)
        after = new.patch_sources(before)
        self.assertEqual(len(new.RECIPES),10)
        self.assertEqual(new.RECIPES['CCA10'], ('fb_cc10','none',False))
        self.assertEqual(new.RECIPES['CCS30'], ('none','fb_cc30',False))
        self.assertEqual(new.RECIPES['zero304'], ('fb304','none',True))
        self.assertIn('all(f in ("fb304", "fb_uc304", "fb_cc10", "fb_cc30")',after['run_s.py'])
        self.assertIn('runs/coral304_inputs/streams.sha256',after['score_s_v66p.py'])
        self.assertIn("('CCA10', 'UCA')",after['score_s_v66p.py'])
        self.assertIn('verify_zero(control, zero)',after['score_s_v66p.py'])
        self.assertIn('assembler validation manifest changed',after['score_s_v66p.py'])
        self.assertIn('producer fold changed',after['score_s_v66p.py'])
        self.assertNotEqual(before['run_s.py'],after['run_s.py'])

if __name__ == '__main__': unittest.main()
