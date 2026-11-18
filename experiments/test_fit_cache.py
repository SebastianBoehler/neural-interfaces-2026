"""Real-EEG regression checks for training-cache isolation; no accuracy is scored."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deep import AlignedNet
from session_bn import SessionBN, SessionNet


class FitCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        path = Path(__file__).resolve().parents[1] / "data/proteus_windows.npz"
        with np.load(path, allow_pickle=False) as data:
            cls.data = {k: data[k][:24] for k in ("X", "y", "subject")}
            cls.data["stream"] = np.char.add(cls.data["subject"], data["session"][:24])
        cls.rows = np.arange(24)

    def setUp(self):
        self.saved_cache = AlignedNet._fitted
        AlignedNet._fitted = {}
        self.addCleanup(setattr, AlignedNet, "_fitted", self.saved_cache)

    def model(self, cls=AlignedNet, **kwargs):
        model = cls(self.data, arch="eegnet", pretrained=False, epochs=1,
                    batch=8, adabn=0, **kwargs)
        model.dev = "cpu"
        return model

    def test_inference_only_adabn_change_reuses_base_fit(self):
        first = self.model()
        first.fit(self.rows)
        second = self.model()
        second.adabn = 5
        with patch.object(second, "_fit", side_effect=AssertionError("unexpected refit")):
            second.fit(self.rows)
        self.assertIsNot(first.net, second.net)
        for name, tensor in first.net.state_dict().items():
            self.assertTrue(torch.equal(tensor, second.net.state_dict()[name]), name)

    def test_session_training_cannot_reuse_plain_batchnorm(self):
        self.model().fit(self.rows)
        session = self.model(SessionNet, group=4, causal=True, k_train=7)
        session.fit(self.rows)
        layers = [m for m in session.net.modules() if isinstance(m, SessionBN)]
        self.assertTrue(layers, "plain fitted net silently replaced SessionNet training")
        self.assertTrue(all(m.causal and m.k_train == 7 and m.m == 4 for m in layers))

    def test_cotraining_change_cannot_skip_fit(self):
        self.model().fit(self.rows)
        changed = self.model()
        changed.cotrain = .25
        # Co-training needs external corpora. Inspect cache dispatch without loading them.
        with patch.object(changed, "_fit", side_effect=RuntimeError("requested co-training")):
            with self.assertRaisesRegex(RuntimeError, "requested co-training"):
                changed.fit(self.rows)

    def test_source_weights_cannot_be_accepted_then_ignored(self):
        model = self.model()
        model.d = {**self.data, "w": np.linspace(.5, 1.5, len(self.rows))}
        with self.assertRaisesRegex(AssertionError, "does not apply"):
            model.fit(self.rows)


if __name__ == "__main__":
    unittest.main()
