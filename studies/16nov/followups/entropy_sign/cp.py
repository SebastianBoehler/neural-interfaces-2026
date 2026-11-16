"""Reuse the registered original causal-personalization implementation."""
import os
import sys
from pathlib import Path
root=Path(os.environ['NI26_WS']).resolve(strict=True)
here=Path(__file__).resolve()
sys.path.insert(0,str(root/'studies/16nov/followups/geometry'))
from cp import patch_source,replace_once
path=root/'studies/14nov/newmember/s281_cp.py'
source=patch_source(path.read_text())
old='assert arm in {"control", "geo_zero", "geo_s50", "geo_d25", "geo_d50", "geo_d100"}'
source=replace_once(source,old,'assert arm in {"Rzero", "Anchor", "Sharpen", "Flatten"}')
anchor='    tracker.track(Path(os.environ["S295_CP_WRAPPER"]))'
source=replace_once(source,anchor,anchor+'\n    tracker.track('+repr(str(root/'studies/16nov/followups/geometry/cp.py'))+')')
os.environ['S295_CP_WRAPPER']=str(here)
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path),'os':os})
