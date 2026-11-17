"""Control routing must reject changed archived or copied artifacts."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from control_source import verify_pair


class ControlSourceTest(unittest.TestCase):
 def test_both_control_copies_must_match_locked_digest(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'new';old=Path(tmp)/'old';root.mkdir();old.mkdir();name='control.npz'
   (root/name).write_bytes(b'control');(old/name).write_bytes(b'control')
   digest=hashlib.sha256(b'control').hexdigest()
   verify_pair(root,old,name,digest)
   (root/name).write_bytes(b'changed')
   with self.assertRaises(ValueError):verify_pair(root,old,name,digest)
   (root/name).write_bytes(b'control');(old/name).write_bytes(b'changed')
   with self.assertRaises(ValueError):verify_pair(root,old,name,digest)


if __name__=='__main__':unittest.main()
