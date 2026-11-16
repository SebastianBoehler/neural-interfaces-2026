import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import gate
import head
from source_labels import read_selected, read_training_labels


class ExpertTest(unittest.TestCase):
    def test_nested_gate_excludes_each_held_person(self):
        rng = np.random.default_rng(3)
        people = np.repeat(np.arange(9), 12)
        labels = np.tile(np.repeat(np.arange(3), 4), 9)
        domain = np.tile([0, 0, 1, 1], 27)
        x = rng.normal(size=(len(people), 246))
        oof, model, traces = gate.nested(x, people, labels, domain, 9)
        self.assertTrue(np.isfinite(oof).all())
        self.assertEqual(gate.predict(model, x[:3]).shape, (3,))
        for person, trace in traces.items():
            self.assertNotIn(int(person), trace['gate_fit_people'])
            self.assertNotIn(int(person), trace['calibration_people'])
            for other, fit_ids in trace['inner_fits'].items():
                self.assertNotIn(int(person), fit_ids); self.assertNotIn(int(other), fit_ids)
        with self.assertRaises(ValueError): gate.nested(x, people, labels, domain, 0)

    def test_zero_deviation_and_objective_gradient(self):
        rng = np.random.default_rng(5); x = rng.normal(size=(12, 4))
        theta = rng.normal(size=(2, 3, 5)); theta[1] = 0
        np.testing.assert_allclose(head.predict(theta, x, np.ones(12)),
                                   head.predict(theta, x, -np.ones(12)), rtol=0, atol=0)
        t = torch.tensor(theta, requires_grad=True); X = torch.tensor(x)
        loss = head.objective(t, X, torch.tensor(np.arange(12) % 3),
                              torch.tensor(np.linspace(-1, 1, 12)), torch.ones(12), 10.)
        loss.backward(); self.assertTrue(torch.isfinite(t.grad).all())
        self.assertGreater(float(t.grad[1].abs().max()), 0.)

    def test_source_label_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.npz'; y = np.arange(20) % 3
            interfaces = np.array(['Graz', 'BrainHero'] * 10)
            np.savez_compressed(path, y=y, interface=interfaces)
            rows = np.array([0, 3, 7, 18], np.int64)
            np.testing.assert_array_equal(read_training_labels(path, rows, len(y)), y[rows])
            np.testing.assert_array_equal(read_selected(path, 'interface', rows, len(y)), interfaces[rows])
            with self.assertRaises(ValueError): read_training_labels(path, rows[::-1], len(y))


if __name__ == '__main__': unittest.main()
