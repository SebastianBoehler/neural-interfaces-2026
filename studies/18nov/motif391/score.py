import json
from pathlib import Path
import numpy as np
from fit import ARMS
import score379 as base

base.ARMS=ARMS
base.CONTRASTS=[('transition',a) for a in ('power','marginal','scrambled','frozen','JS')]+[
    ('marginal','power'),('d1','marginal'),('d3','marginal'),('d8','marginal')]
base.main(Path.cwd())
Path('result379.json').rename('result391.json')
