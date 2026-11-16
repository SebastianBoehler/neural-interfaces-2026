import unittest

from assemble import validate_record


class RecordTests(unittest.TestCase):
    def test_exact_schedule_and_active_bounded_state(self):
        record = {'session': 'P01|1', 'windows': 49, 'updates': 9, 'grad_max': .1, 'bias_norm': .2}
        validate_record(record, 'P01|1', 49, .25)
        for changed in ({'updates': 6}, {'bias_norm': .3}, {'grad_max': 0}, {'session': 'other'}):
            with self.assertRaises(ValueError):
                validate_record({**record, **changed}, 'P01|1', 49, .25)


if __name__ == '__main__':
    unittest.main()
