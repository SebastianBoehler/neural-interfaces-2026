"""Score all complete §310 arms using the existing guarded metric path."""
import os
import sys
from pathlib import Path
root=Path(os.environ['NI26_WS']).resolve(strict=True)
here=Path(__file__).resolve().parent
sys.path[:0]=[str(here),str(root/'studies/16nov/followups/native_core'),str(root/'studies/16nov/followups/geometry')]
import score
from entropy_chain import ARMS
path=root/'studies/16nov/followups/geometry/score.py'
source=score.patch_score(path.read_text())
old='from member_chain import ARMS, ORDERS'
if source.count(old)!=1: raise ValueError('unexpected scorer arm source')
source=source.replace(old,'from entropy_chain import ARMS, ORDERS',1)
old="pairs = [('Nfixed', 'control'), ('Ntrain', 'control'), ('Ntrain', 'Nfixed')]"
pairs=[('Anchor','control'),('Sharpen','Anchor'),('Flatten','Anchor'),('Flatten','Sharpen'),('Flatten','control')]
if source.count(old)!=1: raise ValueError('unexpected score contrasts')
source=source.replace(old,'pairs = '+repr(pairs),1).replace('native_core_s301_results.json','entropy_sign_s310_results.json')
ctx={'__name__':'s307_score','__file__':str(path)}
exec(compile(source,str(path),'exec'),ctx)
score.ARMS=ARMS;ctx['verify_geometry']=score.verify_native;ctx['main']()
