import unittest

import numpy as np

from run import task_config
from score import validate_status


class ScoreTests(unittest.TestCase):
    def test_exact_task_mapping(self):
        self.assertEqual(len({task_config(i) for i in range(80)}), 80)
        with self.assertRaises(ValueError):
            task_config(80)

    def test_status_rejects_silent_adaptation(self):
        event = {'static': {'windows': 61, 'updates': 0},
                 'adaptive': {'windows': 61, 'updates': 2, 'prefix_count': 60,
                              'max_grad': .1, 'delta': .05}}
        validate_status(event, 61)
        bad = {**event, 'adaptive': {**event['adaptive'], 'max_grad': 0}}
        with self.assertRaises(ValueError):
            validate_status(bad, 61)
        bad = {**event, 'adaptive': {**event['adaptive'], 'delta': .2}}
        with self.assertRaises(ValueError):
            validate_status(bad, 61)


if __name__ == '__main__':
    unittest.main()
