"""Native clean CPU worker check of original frozen raw tangent/PCA source fold."""
import json
import sys
from pathlib import Path

import torch

import returning_view_guard as G

if torch.__version__!='2.14.1+cu130' or 'neural-interfaces-2026/.venv/' in Path('/proc/self/maps').read_text():
    raise ValueError('actual isolated qualified worker image required')
G.main(sys.argv[1])
p=Path('raw_view_guard.json');proof=json.loads(p.read_text());proof['executor']='actual_clean_worker_SIF_CPU'
proof['worker_image_sha256']=Path('worker_image.sha256').read_text().split()[0];p.write_text(json.dumps(proof,indent=2))
