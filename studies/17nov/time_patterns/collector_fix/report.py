"""Build the §314 report with explicit, JSON-compatible arm/order keys."""
import numpy as np

ARMS=('TF','TFD','reference')


def build_report(official,pp,orders,people,folds,source_sha,summarize):
 expected={(a,o) for a in ARMS for o in orders}
 if set(official)!=expected or set(pp)!=expected:raise ValueError('incomplete scored arm/order matrix')
 if not all(np.isfinite(official[k]) and np.isfinite(pp[k]).all() for k in expected):raise ValueError('nonfinite metric')
 def contrast(a,b):return summarize(np.stack([pp[a,o]-pp[b,o] for o in orders]),orders,people)
 return {'official_equal_cell_by_order':{a:{o:official[a,o] for o in orders} for a in ARMS},
         'official_equal_cell_scheme_equal':{a:float(np.mean([np.mean([official[a,o] for o in orders if o.startswith(s)]) for s in ('within','full','runs')])) for a in ARMS},
         'absolute_people_order':{a:summarize(np.stack([pp[a,o] for o in orders]),orders,people) for a in ARMS},
         'people_order_intervals':{'TFD_minus_TF':contrast('TFD','TF'),'TF_minus_reference':contrast('TF','reference'),'TFD_minus_reference':contrast('TFD','reference')},
         'fold_sha256':folds,'source_sha256':source_sha,
         'seed_aggregation':'argmax each independently trained seed, score, then average',
         'interpretation':'whole static decoder; external reference is new-person routed-chain emulation, not a matched baseline or package proof'}
