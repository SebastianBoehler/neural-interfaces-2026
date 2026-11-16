"""§294 fixed outer-slot recipes and explicit per-knob ablations."""
import math

ARMS = {
    'E05CS': ('eegn', .5, 'swap', 1.), 'E1CS': ('eegn', 1., 'swap', 1.),
    'E2CS': ('eegn', 2., 'swap', 1.), 'D05CS': ('eegn_dist2', .5, 'swap', 1.),
    'D1CS': ('eegn_dist2', 1., 'swap', 1.), 'D2CS': ('eegn_dist2', 2., 'swap', 1.),
    'E1CA': ('eegn', 1., 'add', 1.), 'D1CA': ('eegn_dist2', 1., 'add', 1.),
    'E05': ('eegn', .5, 'none', 1.), 'E2': ('eegn', 2., 'none', 1.),
    'D05': ('eegn_dist2', .5, 'none', 1.), 'D2': ('eegn_dist2', 2., 'none', 1.),
    'FBL15': ('none', 0., 'none', 1.5),
    'EFBL': ('eegn', 1., 'none', 1.5), 'DFBL': ('eegn_dist2', 1., 'none', 1.5),
}
ZERO = ('eegn', 0., 'none', 1.)
SLOTS = ('sc_fb', 'sc_fbl', 'sc_sinc', 'sc_fbraw')


def get(name):
    if name == 'zero':
        return ZERO
    if name not in ARMS:
        raise ValueError(f'unknown registered recipe {name}')
    return ARMS[name]


def modify_extra(extra, recipe):
    family, scale, coral, fbl = recipe
    if family not in ('none', 'eegn', 'eegn_dist2') or coral not in ('none', 'swap', 'add'):
        raise ValueError('invalid family or CORAL operation')
    if not math.isfinite(scale) or scale < 0 or not math.isfinite(fbl) or fbl <= 0:
        raise ValueError('invalid slot weight')
    if family == 'none' and scale != 0:
        raise ValueError('absent family cannot carry weight')
    names = [name for name, _ in extra]
    if len(names) != len(set(names)) or any(names.count(name) != 1 for name in SLOTS):
        raise ValueError('missing or duplicate existing slot')
    weights = dict(extra)
    base = sum(float(weights[name]) for name in SLOTS) / len(SLOTS)
    fb_weight = float(weights['sc_fb'])
    out = [(name, float(w) * (fbl if name == 'sc_fbl' else 1.))
           for name, w in extra if not (coral == 'swap' and name == 'sc_fb')]
    if coral != 'none':
        out.append(('nm_fb_coral10', fb_weight if coral == 'swap' else base))
    if family != 'none':
        out.append((f'nm_{family}', scale * base))
    return out
