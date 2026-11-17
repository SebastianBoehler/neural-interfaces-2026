"""Run the unchanged §304 smoke with historically correct control provenance."""
import os
import sys
from pathlib import Path

root=Path(os.environ['NI26_WS']).resolve(strict=True)
here=Path(__file__).resolve().parent
path=root/'studies/16nov/followups/coral_repair/chain/smoke_check.py'
source=path.read_text();old='spec.loader.exec_module(s)'
new=old+'\nfrom control_source import install\ninstall(s, ROOT)'
if source.count(old)!=1:raise ValueError('unexpected smoke scorer loading')
sys.path.insert(0,str(here))
exec(compile(source.replace(old,new,1),str(path),'exec'),{'__name__':'__main__','__file__':str(path)})
