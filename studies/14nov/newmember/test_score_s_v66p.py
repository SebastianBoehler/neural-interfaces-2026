"""Checks for the §281 post-personalisation scorer; no treatment data."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

PATH = Path(__file__).with_name('score_s_v66p.py')


class ScorerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not PATH.exists():
            raise AssertionError('§281 scorer is missing')
        spec = importlib.util.spec_from_file_location('score_s_v66p', PATH)
        cls.s = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.s)

    def test_labels_are_taken_before_cell_and_person_averages(self):
        # Two cells for P01; the first is perfect and the second has one
        # class wrong. P02 has one perfect cell. Scores deliberately mix
        # probability and log-score scales.
        scores = np.array([[.8,.1,.1], [1.,-9.,-9.], [-9.,1.,-9.],
                           [-1.,-2.,-3.], [.1,.8,.1], [.1,.1,.8]])
        y = np.array([0, 0, 1, 2, 1, 2]); cell = np.array([0, 0, 0, 1, 1, 2])
        person = np.array(['P01','P01','P01','P01','P01','P02'])
        got = self.s.person_accuracy(scores, y, cell, person, ('P01','P02'))
        np.testing.assert_allclose(got, [75., 100.])

    def test_cells_cannot_cross_people_and_nonfinite_scores_fail(self):
        y = np.array([0, 1]); c = np.array([0, 0]); p = np.array(['P01', 'P02'])
        with self.assertRaisesRegex(ValueError, 'cell'):
            self.s.person_accuracy(np.ones((2,3)), y, c, p, ('P01','P02'))
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.s.person_accuracy(np.array([[0,np.nan,1],[0,1,0]]), y, np.array([0,1]), p, ('P01','P02'))

    def test_order_weighting_is_by_scheme_not_by_order_count(self):
        # Four within orders + two each of full/runs: three schemes count equally.
        orders = ('within0','within1','within2','within3','full0','full1','runs0','runs1')
        d = np.array([[3., 3.]]*4 + [[0., 0.]]*2 + [[0., 0.]]*2)
        out = self.s.summarize(d, orders, ('P04','P10'), bootstrap=200, seed=2)
        self.assertAlmostEqual(out['mean'], 1.0)
        self.assertEqual(out['people']['P04'], 1.0)
        self.assertEqual(out['people']['P10'], 1.0)
        np.testing.assert_allclose(out['interval'], [1.0, 1.0])

    def test_zero_check_catches_changed_full_stage_scores(self):
        x = np.array([[0., -1., -3.], [.2,.3,.5]])
        self.s.verify_zero(x, x.copy())
        x2 = x.copy(); x2[1, 0] += 1e-4
        with self.assertRaisesRegex(ValueError, 'zero'):
            self.s.verify_zero(x, x2)

    def test_routing_requires_matching_archive_and_no_route_on_every_base(self):
        bases = ('B2_s0','B2_s1')
        route = {'order':'within0','archive_sha256':'abc','bases':{
            b:{'sessions':2,'routed_sessions':0,'contrast_windows':0,'bio_windows':0,'routed':[]} for b in bases}}
        self.s.verify_route(route, 'within0', 'abc', bases, 2)
        with self.assertRaisesRegex(ValueError, 'archive'):
            self.s.verify_route(route, 'within0', 'wrong', bases, 2)
        route['bases']['B2_s0']['bio_windows'] = 1
        with self.assertRaisesRegex(ValueError, 'rout'):
            self.s.verify_route(route, 'within0', 'abc', bases, 2)

    def test_manifest_rejects_modified_input_and_wrong_recipe(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d); source=d/'source.npz'; source.write_bytes(b'source')
            digest=self.s.sha256(source)
            manifest={'order':'within0', 'recipe':{'name':'S1','add':'fb_coral','swap':'none','zero':False,'seeds':'0,1,2,3,4,5'},
                      'ft_variant':'r15c250','ft_tag':'ttt18within0LPC','inputs_sha256':{str(source):digest}}
            self.s.verify_manifest(manifest, 'S1', 'within0', {})
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                self.s.verify_manifest(manifest, 'S1', 'within0', {})
            manifest['recipe']['name']='S2'
            with self.assertRaisesRegex(ValueError, 'recipe'):
                self.s.verify_manifest(manifest, 'S1', 'within0', {})


    def test_remote_input_paths_resolve_locally_by_content(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve(); f = root/'data'/'windows.npz'
            f.parent.mkdir(); f.write_bytes(b'local copy')
            remote = '/remote/work/ni26/data/windows.npz'
            self.s.verify_inputs({remote: self.s.sha256(f)}, {}, root)
            reference = root/'data'/'reference.npz'; reference.write_bytes(b'local copy')
            f.write_bytes(b'wrong copy')
            with self.assertRaisesRegex(ValueError, 'changed'):
                self.s.verify_inputs({remote: self.s.sha256(reference)}, {}, root)

    def test_package_digest_rejects_modified_zip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'package.zip'; p.write_bytes(b'candidate')
            expected = self.s.sha256(p)
            self.s.verify_package(p, expected)
            p.write_bytes(b'different')
            with self.assertRaisesRegex(ValueError, 'package'):
                self.s.verify_package(p, expected)

    def test_stream_manifest_requires_each_seed_and_correct_remote_hash(self):
        order = 'within0'
        files = {f'/tmp/ttt18{order}_fb_coral_base-all_loso-s{i}.npz': str(i) for i in range(6)}
        remote = {Path(p).name: h for p, h in files.items()}
        self.s.verify_streams({'inputs_sha256': files}, 'S1', order, remote)
        bad = dict(remote); bad[next(iter(bad))] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'stream'):
            self.s.verify_streams({'inputs_sha256': files}, 'S1', order, bad)
        del files[next(iter(files))]
        with self.assertRaisesRegex(ValueError, 'stream'):
            self.s.verify_streams({'inputs_sha256': files}, 'S1', order, remote)

    def test_output_validates_sidecar_sources_and_identity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d).resolve()
            def put(rel, text=b'x'):
                p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(text); return p
            arm, order, base = 'S1', 'within0', 'B2_s0'
            chain = put(f'runs/chainout/s281_v66p_{arm}_{order}.npz')
            audit = put(f'runs/newmember/s281_v66p_{arm}_{order}_gate.npz')
            stage = put('lean/src66/cpers.py')
            required = [chain, audit, stage, put('studies/14nov/newmember/s281_cp.py'),
                        put('studies/14nov/rundetect/package.py'), put('studies/14nov/rundetect/jointcount.py'),
                        put('studies/14nov/rundetect/common.py'), put('studies/14nov/rundetect/detect.py'),
                        put('data/proteus_windows.npz'), put('runs/shuffle_perms.npz')]
            manifest = {'order': order, 'recipe': {'name':arm,'add':'fb_coral','swap':'none','zero':False,
                        'seeds':'0,1,2,3,4,5'}, 'ft_variant':'r15c250','ft_tag':f'ttt18{order}LPC',
                        'inputs_sha256':{str(chain):self.s.sha256(chain)}}
            prov = put(f'runs/newmember/s281_v66p_{arm}_{order}_provenance.json', json.dumps(manifest).encode())
            cp = root / f'runs/s281_v66_cp/{arm}_{order}_{base}.npz'; cp.parent.mkdir(parents=True)
            side = cp.with_suffix('.json')
            side.write_text(json.dumps({'arm':arm,'order':order,'base':base,
                            'inputs_sha256':{str(p):self.s.sha256(p) for p in required}}))
            def save(**updates):
                dct = dict(scores=np.zeros((2,3)), arm=arm, order=order, base=base,
                           source_sha256=self.s.sha256(chain), stage_sha256=self.s.sha256(stage),
                           provenance_sha256=self.s.sha256(prov), input_manifest_sha256=self.s.sha256(side),
                           routing_checked=False, scores_are_probabilities=False)
                dct.update(updates); np.savez(cp, **dct)
            save()
            np.testing.assert_array_equal(self.s.load_output(root, arm, order, base, 2, {}), np.zeros((2,3)))
            save(base='B2_s1')
            with self.assertRaisesRegex(ValueError, 'base'):
                self.s.load_output(root, arm, order, base, 2, {})
            save(input_manifest_sha256='wrong')
            with self.assertRaisesRegex(ValueError, 'input_manifest'):
                self.s.load_output(root, arm, order, base, 2, {})
            inputs = json.loads(side.read_text())
            duplicate = put('data/other/package.py')
            inputs['inputs_sha256'][str(duplicate)] = self.s.sha256(duplicate)
            side.write_text(json.dumps(inputs)); save()
            with self.assertRaisesRegex(ValueError, 'missing required'):
                self.s.load_output(root, arm, order, base, 2, {})
            del inputs['inputs_sha256'][str(duplicate)]
            del inputs['inputs_sha256'][str(root/'data/proteus_windows.npz')]
            side.write_text(json.dumps(inputs)); save()
            with self.assertRaisesRegex(ValueError, 'missing required'):
                self.s.load_output(root, arm, order, base, 2, {})
            inputs['inputs_sha256'][str(root/'data/proteus_windows.npz')] = self.s.sha256(root/'data/proteus_windows.npz')
            side.write_text(json.dumps(inputs)); save(); audit.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                self.s.load_output(root, arm, order, base, 2, {})


if __name__ == '__main__':
    unittest.main()
