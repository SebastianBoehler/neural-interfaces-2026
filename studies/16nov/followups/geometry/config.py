"""Registered treatment definitions and their historical core streams."""
ARMS = {'geo_s50': ('scalar', .5), 'geo_d25': ('diag', .25),
        'geo_d50': ('diag', .5), 'geo_d100': ('diag', 1.)}
ARCHIVES = {'within0': 'coremember.npz', 'within1': 'coremember.npz',
            'full0': 'coremember.npz', 'runs0': 'coremember.npz',
            'within2': 'coremember_within2_within3_full1_runs1.npz',
            'within3': 'coremember_within2_within3_full1_runs1.npz',
            'full1': 'coremember_within2_within3_full1_runs1.npz',
            'runs1': 'coremember_within2_within3_full1_runs1.npz'}
