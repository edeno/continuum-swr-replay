"""For each part, count commits since a date that change the docstring-stripped AST of
any module the part's builder loads (module list taken at HEAD)."""

import ast, importlib, os, subprocess, sys


def norm(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            b = node.body
            if (
                b
                and isinstance(b[0], ast.Expr)
                and isinstance(getattr(b[0], "value", None), ast.Constant)
                and isinstance(b[0].value.value, str)
            ):
                node.body = b[1:] or [ast.Pass()]
    return ast.dump(tree)


def show(rev, path):
    r = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


part = sys.argv[1]
importlib.import_module(f"continuum_swr_replay.reported.part{part}")
files = sorted(
    os.path.relpath(m.__file__)
    for k, m in sys.modules.items()
    if k.startswith("continuum_swr_replay") and getattr(m, "__file__", None)
)
commits = subprocess.run(
    ["git", "log", "--since=2026-10-10", "--format=%H", "--", *files],
    capture_output=True,
    text=True,
).stdout.split()
code = 0
for c in commits:
    changed = subprocess.run(
        ["git", "diff", "--name-only", f"{c}^", c, "--", *files],
        capture_output=True,
        text=True,
    ).stdout.split()
    if any(norm(show(f"{c}^", f) or "") != norm(show(c, f) or "") for f in changed):
        code += 1
days = len(
    set(
        subprocess.run(
            [
                "git",
                "log",
                "--since=2026-10-10",
                "--format=%ad",
                "--date=short",
                "--",
                *files,
            ],
            capture_output=True,
            text=True,
        ).stdout.split()
    )
)
print(
    f"part{part}: {len(commits)} commits, {code} change code (docstrings/comments stripped), over {days} days"
)
