"""Regression checks for the separately defined common-anchor intervention."""
from pathlib import Path

import pytest
import numpy as np

from s316_target import patch_source
from s316_anchor import load_common_anchor
from s316_assemble import patch_source as patch_assembly


def test_common_anchor_patch_is_active_and_fail_closed():
    source = Path(__file__).with_name('target.py').read_text()
    patched = patch_source(source)
    assert 'load_common_anchor(shard, refmeta, cache_hashes' in patched
    assert 'fit_native(native,train,labels)' not in patched
    assert "torch.device('cpu')" in patched
    assert "'anchor_policy':'s305_common'" in patched
    with pytest.raises(ValueError):
        patch_source(source.replace('theta,loss,grad,calls = fit_native(native,train,labels)', 'pass'))


def test_assembly_requires_common_anchor_for_every_person():
    source = Path(__file__).with_name('assemble_target.py').read_text()
    patched = patch_assembly(source)
    assert "m.get('anchor_policy')!='s305_common'" in patched
    with pytest.raises(ValueError):
        patch_assembly(source.replace("m['held']!=held", 'False'))


def test_common_anchor_checks_provenance_and_convergence(tmp_path):
    shard = tmp_path / 'P09.npz'
    np.savez(shard, Nconverged_coef=np.zeros((3, 44239)), Nconverged_intercept=np.array([1., 2., 3.]))
    meta = {'input_sha256': {'P01.npz':'cache'}, 'data_sha256':'data', 'permutation_sha256':'perm',
            'source_sha256':'source', 'held_accuracy_computed':False, 'native_fit_loss':.1,
            'max_gradient':1e-8, 'objective_evaluations':8}
    theta, loss, grad, calls = load_common_anchor(shard,meta,{'P01':'cache'},'data','perm','source')
    assert theta.shape == (3, 44240) and tuple(theta[:,-1]) == (1., 2., 3.)
    assert (loss,grad,calls) == (.1,1e-8,8)
    with pytest.raises(ValueError):
        load_common_anchor(shard,meta,{'P01':'wrong'},'data','perm','source')
    meta['max_gradient'] = 1e-3
    with pytest.raises(ValueError):
        load_common_anchor(shard,meta,{'P01':'cache'},'data','perm','source')
