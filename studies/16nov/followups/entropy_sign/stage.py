"""Stage the complete §310 delta matrix using the original native validator."""
import os
import sys
from pathlib import Path
root=Path(os.environ['NI26_WS']).resolve(strict=True)
path=root/'studies/16nov/followups/native_core/stage.py'
source=path.read_text()
replacements={
 'from member_chain import ARMS, ORDERS, validate_delta':'from entropy_chain import ARMS, ORDERS\nfrom member_chain import validate_delta',
 "source / 'producer.sha256'":"source / 's310-src.sha256'",
 "source / 'runs/s301_native_shards'":"source / 'runs/s310/target'"}
for old,new in replacements.items():
 if source.count(old)!=1: raise ValueError(f'unexpected native staging anchor {old}')
 source=source.replace(old,new,1)
sys.path.insert(0,str(Path(__file__).resolve().parent))
exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path),'os':os,'sys':sys})
