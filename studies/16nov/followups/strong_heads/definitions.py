"""Fixed registered heads, independent seeds and pinned delivery orders."""

ARMS = ('Slinear', 'Stanh', 'Rlinear', 'Rquadratic', 'Rfourier')
ORDERS = ('within0', 'within1', 'within2', 'within3', 'full0', 'full1', 'runs0', 'runs1')
SEED_ARMS = ('Rzero', *(f'{arm}{seed}' for arm in ARMS for seed in (0, 1)))
