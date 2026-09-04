"""Tests for the mutation-check tool's enumeration and mutation mechanics.

The full killed/survived behavior is exercised whenever the check runs against
the builders (its recorded score lives in docs/design/builder_test_floor.md);
these fast tests pin the deterministic core: which sites it finds and that a
mutation actually changes the operation it claims to.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

_TOOL = Path(__file__).resolve().parents[2] / "scripts/qa/mutation_check.py"
_spec = importlib.util.spec_from_file_location("mutation_check", _TOOL)
mutation_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mutation_check)


_SNIPPET = '''
import numpy as np
def f(x, a, b):
    i = np.searchsorted(a, x, side="right")
    if x < 3:
        return a - b + i
    return 0
'''


def _descs(code: str):
    return {desc for _lineno, desc, _payload in mutation_check._enumerate(ast.parse(code))}


def test_enumerates_expected_sites():
    descs = _descs(_SNIPPET)
    assert 'side "right"->"left"' in descs   # searchsorted side
    assert "Lt->LtE" in descs                # the `x < 3` comparison
    assert "Sub->Add" in descs               # the `a - b`
    assert "Add->Sub" in descs               # the `... + i`


def test_apply_side_flip_changes_source():
    tree = ast.parse(_SNIPPET)
    muts = mutation_check._enumerate(tree)
    idx = next(i for i, (_l, d, _p) in enumerate(muts) if d.startswith("side"))
    kind, node, payload = muts[idx][2]
    mutation_check._apply(kind, node, payload)
    mutated = ast.unparse(ast.fix_missing_locations(tree))
    assert "side='left'" in mutated and "side='right'" not in mutated


def test_apply_comparison_swap_changes_operator():
    tree = ast.parse("y = x < 3")
    kind, node, payload = mutation_check._enumerate(tree)[0][2]
    mutation_check._apply(kind, node, payload)
    assert ast.unparse(tree).strip() == "y = x <= 3"
