"""Real C++ LibTorch bridge parity; S308_LIB must name the built library."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export import structured_export  # noqa: E402
from heads import Structured, fit_scale  # noqa: E402
from native import NativeHead  # noqa: E402

pytestmark = pytest.mark.skipif('S308_LIB' not in os.environ, reason='native library is built on Galvani')


def test_native_checkpoint_and_stateless_batch(tmp_path):
    rng = np.random.default_rng(17)
    x = rng.normal(size=(5, 44239)).astype('float64')
    mean, scale = fit_scale(x)
    model = Structured(True).eval()
    path = tmp_path / 'model.pt'
    structured_export(model, mean, scale, x[:1], path)
    expected = torch.jit.load(str(path))(torch.from_numpy(x)).detach().numpy()
    for device in map(int, os.environ.get('S308_TEST_DEVICES', '0').split(',')):
        head = NativeHead(os.environ['S308_LIB'], path, device)
        result = head.predict(x)
        assert np.allclose(result, expected, atol=2e-5, rtol=2e-5)
        assert np.allclose(head.predict(x[:1]), result[:1], atol=2e-5, rtol=2e-5)
        head.close()
        try:
            head.predict(x)
        except RuntimeError:
            pass
        else:
            raise AssertionError('closed native handle accepted prediction')
