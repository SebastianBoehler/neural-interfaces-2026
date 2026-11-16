import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from rolling_audit import record


class AuditTests(unittest.TestCase):
    def test_records_exact_hashes_and_previous_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); item = root / 'output'; item.write_bytes(b'first')
            reg = root / 'ledger'; reg.write_text('registered')
            digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            path, _, previous = record('294', 'E05', [item], reg, digest, root / 'journal')
            self.assertEqual(previous, [])
            _, _, previous = record('296', 'EG06', [item], reg, digest, root / 'journal')
            self.assertEqual(previous, ['294:E05'])
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(records[0]['input_sha256'][str(item.resolve())], digest(item))
            with self.assertRaises(FileNotFoundError):
                record('294', 'E2', [root / 'missing'], reg, digest, root / 'journal')


if __name__ == '__main__': unittest.main()
