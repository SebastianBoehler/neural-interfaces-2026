"""Read shared immutable full-dimensional source7/source6 static feature banks."""
import json
import os
from pathlib import Path
import numpy as np
from data import sha,digest,metadata


def load(name):
    root=Path(os.environ['FIXED_ROOT'])
    if sha(root/'manifest.json')!='c8ebeac134d610dc78111700ae84f5e61d74fdf0f500510ba079fca2fdc7bf9e':raise ValueError('pinnedfixedmanifest changed')
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['source_sha256']!='f80c9285bf09fa95e51badeb2485724f300b2f45b72f87270d0a5854ff2cdae9':raise ValueError('fixedproducer source identity')
    path=root/f'{name}.npz';record=manifest['records'][name]
    if sha(path)!=record['file_sha256'] or manifest['labels_opened'] or manifest['P01_used']:
        raise ValueError('fixed-feature input identity')
    with np.load(path,allow_pickle=False) as z:bank={k:z[k] for k in z.files}
    if bank['features'].shape!=(len(bank['rows']),44239) or digest(bank['rows'][bank['fit']])!=record['source_rows_sha256'] or digest(bank['reference'])!=record['reference_sha256'] or digest(bank['slow_mean'])!=record['slowmean_sha256']:
        raise ValueError('reference/rows/layout differs')
    fitpeople=sorted(set(bank['people'][bank['fit']]))
    expected=[f'P{i:02d}' for i in range(4,11)]
    if name!='source7':expected.remove(name.split('_')[-1])
    if fitpeople!=expected or record['fit_people']!=expected or 'P01' in set(bank['people']):raise ValueError('sourceexclude')
    return bank,record,sha(root/'manifest.json')
