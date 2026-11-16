import unittest
from ids import person_ids


class PersonIdsTest(unittest.TestCase):
    def test_protocol_names_match_fold_ids(self):
        names = [f'Dreyer2026Proteus/P{i:02d}' for i in range(1,11)]
        self.assertEqual(person_ids(names).tolist(),[f'P{i:02d}' for i in range(1,11)])
        with self.assertRaises(ValueError): person_ids(names[:-1])
