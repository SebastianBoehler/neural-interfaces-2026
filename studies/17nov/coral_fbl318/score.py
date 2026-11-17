"""Run the original complete-matrix scorer with verified historical controls."""
import importlib.util
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(root/'studies/14nov/newmember'),str(root/'studies/16nov/followups/coral_repair/chain/recovery')]
from control_source import install
path=root/'studies/14nov/newmember/score_s_v66p.py'
spec=importlib.util.spec_from_file_location('score318',path)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
install(s,root)
s.main()
