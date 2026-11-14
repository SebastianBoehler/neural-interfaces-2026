"""Bundle lean/src into one submission.py (a package may hold a single Python file).

Usage: python lean/tools/bundle.py [OUT]     write the bundle (default submissions/lean_v5/submission.py)
       python lean/tools/bundle.py --check   exit 1 if that file is not the bundle of the current tree

The modules are concatenated in MODULES order. Imports from outside the tree are merged at the top; imports between
modules are dropped after checking that they point to an earlier module and to names defined there. Top-level
names must be unique across the tree. The output depends on the sources only.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC, OUT = ROOT / "lean/src", ROOT / "submissions/lean_v5/submission.py"
MODULES = ("numerics", "tangent", "inputs", "nets", "stages", "gates", "personal", "routing", "ensemble", "solver")
if "--v6" in sys.argv:  # the v6 tree (lean/src6) and its bundle; lean/src and lean_v5 are untouched
    SRC, OUT = ROOT / "lean/src6", ROOT / "submissions/lean_v6/submission.py"
    MODULES = MODULES[:-1] + ("extras", "finetune", "v6chain", "solver")
    sys.argv.remove("--v6")
if "--v61" in sys.argv:  # v6.1 tree (lean/src61); lean/src6 and lean_v6 untouched
    SRC, OUT = ROOT / "lean/src61", ROOT / "submissions/lean_v61/submission.py"
    MODULES = MODULES[:-1] + ("extras", "finetune", "v6chain", "solver")
    sys.argv.remove("--v61")
if "--v62" in sys.argv:  # v6.2 tree (lean/src62); src6, src61 and their bundles untouched
    SRC, OUT = ROOT / "lean/src62", ROOT / "submissions/lean_v62/submission.py"
    MODULES = MODULES[:-1] + ("extras", "finetune", "v6chain", "solver")
    sys.argv.remove("--v62")
if "--v62g" in sys.argv:  # v6.2 with fine-tuning cost guards (lean/src62g); the other trees and bundles untouched
    SRC, OUT = ROOT / "lean/src62g", ROOT / "submissions/lean_v62g/submission.py"
    MODULES = MODULES[:-1] + ("extras", "finetune", "v6chain", "solver")
    sys.argv.remove("--v62g")
if "--v62h" in sys.argv:  # v6.2g with the exception guard in step62 (lean/src62h); the other trees and bundles untouched
    SRC, OUT = ROOT / "lean/src62h", ROOT / "submissions/lean_v62h/submission.py"
    MODULES = MODULES[:-1] + ("extras", "finetune", "v6chain", "solver")
    sys.argv.remove("--v62h")
if "--v62i" in sys.argv:  # v6.2h with a per-session fine-tuning limit instead of the process clock (lean/src62i)
    SRC, OUT = ROOT / "lean/src62i", ROOT / "submissions/lean_v62i/submission.py"
    MODULES = MODULES[:-1] + ("extras", "finetune", "v6chain", "solver")
    sys.argv.remove("--v62i")
if "--v63" in sys.argv:  # v6.3: v6.2i with bitwise savings (lean/src63); the other trees and bundles untouched
    SRC, OUT = ROOT / "lean/src63", ROOT / "submissions/lean_v63/submission.py"
    MODULES = MODULES[:-1] + ("extras", "finetune", "v6chain", "solver")
    sys.argv.remove("--v63")


def top_names(tree):
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names += [n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)]
    return names


def bundle():
    plain, froms, defined, parts = set(), {}, {}, []
    for name in MODULES:
        text = (SRC / f"{name}.py").read_text()
        tree, lines = ast.parse(text), text.splitlines()
        drop = set()
        for i, node in enumerate(tree.body):
            if i == 0 and ast.get_docstring(tree) is not None:
                drop.update(range(node.lineno, node.end_lineno + 1))
            elif isinstance(node, ast.ImportFrom) and node.level:
                missing = [a.name for a in node.names if a.name not in defined.get(node.module, ())]
                assert not missing, f"{name}: {missing} not defined in an earlier module '{node.module}'"
                drop.update(range(node.lineno, node.end_lineno + 1))
            elif isinstance(node, ast.ImportFrom):
                assert all(a.asname is None for a in node.names), f"{name}: 'from x import y as z' is not supported"
                froms.setdefault(node.module, set()).update(a.name for a in node.names)
                drop.update(range(node.lineno, node.end_lineno + 1))
            elif isinstance(node, ast.Import):
                plain.update(ast.unparse(ast.Import([a])) for a in node.names)
                drop.update(range(node.lineno, node.end_lineno + 1))
        names = top_names(tree)
        clash = [n for n in names if any(n in d for d in defined.values())]
        assert not clash, f"{name}: top-level names {clash} are already defined in an earlier module"
        defined[name] = names
        body = "\n".join(line for i, line in enumerate(lines, 1) if i not in drop).strip("\n")
        title = ast.get_docstring(tree).splitlines()[0]
        parts.append(f"# {'=' * 100}\n# {name}: {title}\n# {'=' * 100}\n\n{body}\n")
    doc = ast.get_docstring(ast.parse((SRC / "__init__.py").read_text()), clean=False).strip()
    head = (f'"""{doc}\n\nGenerated by lean/tools/bundle.py from lean/src; edit the tree and bundle again, '
            f'never this file.\n"""\n')
    imports = sorted(plain) + [f"from {m} import {', '.join(sorted(n))}" for m, n in sorted(froms.items())]
    out = head + "\n" + "\n".join(imports) + "\n\n\n" + "\n\n".join(parts)
    compile(out, "submission.py", "exec")
    return out


def main():
    out = bundle()
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text() != out:
            sys.exit(f"{OUT.relative_to(ROOT)} is not the bundle of lean/src: run lean/tools/bundle.py")
        print("OK: bundle is current")
        return
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out)
    print(f"wrote {path} ({len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()
