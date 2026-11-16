"""A rolling score cannot proceed with a partial arm or sidecar set."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('rolling_test', HERE / 'rolling_v2.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


class CompleteArmTests(unittest.TestCase):
    def test_requires_every_order_base_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            oldroot = r.root
            try:
                r.root = Path(temp); directory = r.root / 'runs/s281_v66_cp'; directory.mkdir(parents=True)
                with self.assertRaises(ValueError): r.prerequisites('E05')
                for order in r.s.ORDERS:
                    for base in r.s.BASES:
                        p = directory / f'E05_{order}_{base}.npz'; p.touch(); p.with_suffix('.json').touch()
                r.prerequisites('E05')
                (directory / f'E05_{r.s.ORDERS[-1]}_{r.s.BASES[-1]}.json').unlink()
                with self.assertRaises(ValueError): r.prerequisites('E05')
                with self.assertRaises(ValueError): r.prerequisites('not_registered')
            finally:
                r.root = oldroot


if __name__ == '__main__': unittest.main()
