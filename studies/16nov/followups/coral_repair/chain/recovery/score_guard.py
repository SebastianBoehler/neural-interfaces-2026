"""Run the unchanged §304 scorer with validated historical controls."""
import importlib.util
import os
import sys
from pathlib import Path

root=Path(os.environ['NI26_WS']).resolve(strict=True)
here=Path(__file__).resolve().parent
sys.path[:0]=[str(here),str(root/'studies/14nov/newmember')]
from control_source import install
path=root/'studies/14nov/newmember/score_s_v66p.py'
spec=importlib.util.spec_from_file_location('s304_recovery_score',path)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
install(s,root)
s.main()
