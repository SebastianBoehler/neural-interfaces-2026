"""Fail-closed fold provenance, routing and probability checks."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import validate as V


class FoldValidation(unittest.TestCase):
    def test_fold_rejects_wrong_order_and_bad_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); folder = root / 'runs/coral304'; folder.mkdir(parents=True)
            subjects = np.array(['held'] + [f'train{i}' for i in range(9)])
            train = np.array([False] + [True] * 9); test = ~train
            folds = {'P01': (train, test)}
            d = {'subject': subjects}; perms = {o: np.arange(10) for o in V.ORDERS}
            meta = dict(arm='conditional10', seed=0, held='P01', fit_people=sorted(subjects[train]),
                        manifest_sha256=V.MANIFEST, data_sha256=V.DATA, permutation_sha256=V.PERMS,
                        train_rows_sha256=hashlib.sha256(np.flatnonzero(train).astype(np.int64).tobytes()).hexdigest(),
                        parameter_sha256='a'*64, dg_stats=dict(coral_pairs=1, coral_pen=1.),
                        penalty=dict(batches=1, class_pairs=[1, 1, 1], gradient_norm=1.))
            arrays = {'metadata': json.dumps(meta)}
            for order in V.ORDERS:
                arrays[f'{order}|index'] = np.array([0]); arrays[f'{order}|proba'] = np.array([[.2, .3, .5]])
            path = folder / 'conditional10_s0_P01.npz'
            np.savez(path, **arrays)
            V.fold(root, d, folds, perms, 'conditional10', 0, 'P01')
            arrays['full0|index'] = np.array([1]); np.savez(path, **arrays)
            with self.assertRaises(ValueError): V.fold(root, d, folds, perms, 'conditional10', 0, 'P01')
            arrays['full0|index'] = np.array([0]); arrays['full0|proba'] = np.array([[.2, .3, .4]])
            np.savez(path, **arrays)
            with self.assertRaises(ValueError): V.fold(root, d, folds, perms, 'conditional10', 0, 'P01')
            arrays['full0|proba'] = np.array([[.2, .3, .5]])
            meta['penalty']['class_pairs'][2] = 0; arrays['metadata'] = json.dumps(meta); np.savez(path, **arrays)
            with self.assertRaises(ValueError): V.fold(root, d, folds, perms, 'conditional10', 0, 'P01')


if __name__ == '__main__': unittest.main()
