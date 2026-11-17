"""Build the registered §318 runner from an isolated, locked §304 copy."""
import re
import sys
from pathlib import Path

RECIPES = {
    'control': ('none','none',False), 'zero318': ('fb_cc30','none',True),
    'SA': ('fb_cc30','none',False), 'SS': ('none','fb_cc30',False),
    'FBL2': ('none','none',False), 'JA': ('fb_cc30','none',False),
    'JS': ('none','fb_cc30',False),
}
FACTORS = {'control':1.,'zero318':1.,'SA':1.,'SS':1.,'FBL2':2.,'JA':2.,'JS':2.}
CONTRASTS = [('JA','SA'),('JA','FBL2'),('JS','SS'),('JS','FBL2')]


def replace(src, old, new):
    if src.count(old) != 1:
        raise ValueError(f'expected one patch anchor, found {src.count(old)}: {old[:80]!r}')
    return src.replace(old, new, 1)


def patch(sources):
    out = dict(sources)
    p = out['s281_provenance.py']
    a,b = p.index('RECIPES = {'),p.index('\n\n\ndef validate_recipe')
    p = p[:a] + 'RECIPES = ' + repr(RECIPES) + '\nFACTORS = ' + repr(FACTORS) + p[b:]
    p = replace(p, "'zero': zero == '1', 'seeds': seeds}", "'zero': zero == '1', 'seeds': seeds, 'fbl_factor': FACTORS[name]}")
    out['s281_provenance.py'] = p
    p = out['run_s.py']
    p = replace(p, 'if FAMS_:\n    _mix = A.mix', 'FACTOR = {"FBL2": 2., "JA": 2., "JS": 2.}.get(NAME, 1.)\nif FAMS_ or FACTOR != 1.:\n    _mix = A.mix')
    p = replace(p, '        m = _mix(bn, p, fams, extra, *a, **k); d = float(np.abs(m[0] - m0[0]).max()); LOG.append((bn, w, d))', '''        prior = list(extra)
        assert sum(n == "sc_fbl" for n, _ in prior) == 1
        extra = [(n, x * FACTOR if n == "sc_fbl" else x) for n, x in prior]
        m = _mix(bn, p, fams, extra, *a, **k)
        if FACTOR != 1.:
            ref = _mix(bn, p, fams, prior, *a, **k)
            assert not np.array_equal(m[0], ref[0]), "FBL factor has no effect"
        if FAMS_ and not ZERO:
            without = [(n, x) for n, x in extra if n != f"nm_{FAMS_[0]}"]
            if SW: without.append(("sc_fb", sw["sc_fb"]))
            ref = _mix(bn, p, fams, without, *a, **k)
            assert not np.array_equal(m[0], ref[0]), "CC30 member has no effect"
        d = float(np.abs(m[0] - m0[0]).max()); LOG.append((bn, w, d))''')
    p = replace(p, 'if FAMS_:\n    assert LOG,', 'if FAMS_ or FACTOR != 1.:\n    assert LOG,')
    out['run_s.py'] = p
    p,count = re.subn(r'    assert arm in \{[^\n]+\}', '    assert arm in ' + repr(set(RECIPES)), out['s281_cp.py'], count=1)
    if count != 1: raise ValueError('CP arm guard anchor changed')
    out['s281_cp.py'] = p
    p = out['score_s_v66p.py']
    p = re.sub(r"^ARMS = .*?$", 'ARMS = ' + repr(tuple(RECIPES)), p, count=1, flags=re.M)
    p = p.replace('zero304','zero318')
    marker = "    verify_inputs(manifest.get('inputs_sha256'), cache, root)"
    p = replace(p, marker, "    if recipe.get('fbl_factor') != " + repr(FACTORS) + "[arm] and arm != 'control':\n        raise ValueError('incorrect FBL factor')\n" + marker)
    a = p.index(' + [(\'CCA10\', \'UCA\')', p.index('def main():'))
    b = p.index(':\n',a)
    p = p[:a] + ' + ' + repr(CONTRASTS) + p[b:]
    anchor = "\n\nif __name__ == '__main__':"
    extra = '''
    for joint, single in (("JA", "SA"), ("JS", "SS")):
        d = np.stack([metric[joint,o] - metric[single,o] - metric["FBL2",o] + metric["control",o] for o in ORDERS])
        print(f"{joint} interaction: {summarize(d, ORDERS, PEOPLE)} cell-equal: {cell_weighted_summary(d, ORDERS, cell_counts)}")
'''
    out['score_s_v66p.py'] = replace(p,anchor,extra+anchor)
    return out


def main(root):
    base = Path(root).resolve(strict=True) / 'studies/14nov/newmember'
    names = ('run_s.py','s281_provenance.py','s281_cp.py','score_s_v66p.py')
    paths = {n:base/n for n in names}
    if any(not p.is_file() or p.is_symlink() or p.stat().st_nlink != 1 for p in paths.values()):
        raise ValueError('requires a private regular-file snapshot')
    for n, contents in patch({n:p.read_text() for n,p in paths.items()}).items():
        paths[n].write_text(contents)

if __name__ == '__main__': main(sys.argv[1])
