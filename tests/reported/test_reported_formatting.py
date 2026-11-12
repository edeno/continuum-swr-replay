"""The rounding rules: every style of ``reported.formatting``, one row per rule.

Each expected text is worked by hand from the rule in the module's docstring,
so a change to a rule fails here and not only in the generated files.
"""

from __future__ import annotations

import pytest

from continuum_swr_replay.reported.formatting import formatted
from continuum_swr_replay.reported.values import Value

MINUS = "−"


def _value(x: float, kind: str = "quantity", style: str = "sig2") -> Value:
    return Value(
        key="x",
        value=x,
        kind=kind,
        style=style,
        unit="",
        description="d",
        questions=("2d.2",),
        figures=(),
        source="s",
        function="f",
    )


@pytest.mark.parametrize(
    "x, kind, style, text",
    [
        # count: exact, with thousands separators
        (75675, "count", "count", "75,675"),
        (7, "count", "count", "7"),
        # pct: a whole percent with its sign; a half rounds up
        (0.57, "fraction", "pct", "57 %"),
        (0.565, "fraction", "pct", "57 %"),
        (0.564, "fraction", "pct", "56 %"),
        (0.005, "fraction", "pct", "1 %"),
        (1.0, "fraction", "pct", "100 %"),
        # pct1: a percent to one decimal; a half rounds up
        (0.004, "fraction", "pct1", "0.4 %"),
        (0.0125, "fraction", "pct1", "1.3 %"),
        (0.78, "fraction", "pct1", "78.0 %"),
        # pctnum: the same number without the sign
        (0.47, "fraction", "pctnum", "47"),
        # int: a whole number; a half rounds away from zero
        (37.5, "quantity", "int", "38"),
        (36.5, "quantity", "int", "37"),
        (-2.5, "quantity", "int", MINUS + "3"),
        (-0.4, "quantity", "int", "0"),
        # sig2: two significant figures
        (0.51, "quantity", "sig2", "0.51"),
        (1.1, "quantity", "sig2", "1.1"),
        (96, "quantity", "sig2", "96"),
        (1234, "quantity", "sig2", "1200"),
        (0.125, "quantity", "sig2", "0.13"),
        (0.0, "quantity", "sig2", "0"),
        (-0.071, "quantity", "sig2", MINUS + "0.071"),
        # sig2 where rounding up gains a digit: still two figures
        (0.0995, "quantity", "sig2", "0.10"),
        (9.95, "quantity", "sig2", "10"),
        (99.5, "quantity", "sig2", "100"),
        # dec2: two decimal places; a value that rounds to zero has no sign
        (0.6, "quantity", "dec2", "0.60"),
        (0.125, "quantity", "dec2", "0.13"),
        (-0.004, "quantity", "dec2", "0.00"),
        (-0.071, "quantity", "dec2", MINUS + "0.07"),
        # sig2s: the sign always written
        (0.73, "quantity", "sig2s", "+0.73"),
        (-0.071, "quantity", "sig2s", MINUS + "0.071"),
        # pval: two figures; scientific notation below 0.001
        (0.034, "quantity", "pval", "0.034"),
        (0.0012, "quantity", "pval", "0.0012"),
        (0.00099996, "quantity", "pval", "0.0010"),
        (0.00096, "quantity", "pval", "9.6e-4"),
        (1.1e-11, "quantity", "pval", "1.1e-11"),
        (9.96e-5, "quantity", "pval", "1.0e-4"),
    ],
)
def test_each_style_prints_by_its_rule(x, kind, style, text):
    assert formatted(_value(x, kind, style)) == text


@pytest.mark.parametrize(
    "x, kind, style, text",
    [
        (75675, "count", "count", "75{,}675"),
        (0.57, "fraction", "pct", "57"),
        (0.004, "fraction", "pct1", "0.4"),
        (-0.071, "quantity", "sig2s", "\\ensuremath{-}0.071"),
        (-0.071, "quantity", "dec2", "\\ensuremath{-}0.07"),
        (1.1e-11, "quantity", "pval", "\\ensuremath{1.1\\times10^{-11}}"),
    ],
)
def test_latex_leaves_the_percent_sign_to_the_prose_and_writes_a_true_minus(
    x, kind, style, text
):
    assert formatted(_value(x, kind, style), latex=True) == text


def test_a_marker_s_style_replaces_the_value_s_own():
    assert formatted(_value(0.47, "fraction", "pct"), "pctnum") == "47"
    assert formatted(_value(1234, "quantity", "sig2"), "int") == "1234"


@pytest.mark.parametrize(
    "x, kind, style, error",
    [
        (12, "count", "pct", "prints fractions"),
        (2.5, "quantity", "pctnum", "prints fractions"),
        (2.5, "quantity", "pct1", "prints fractions"),
        (2.5, "quantity", "count", "needs a whole number"),
    ],
)
def test_a_style_that_does_not_suit_the_value_is_refused(x, kind, style, error):
    with pytest.raises(ValueError, match=error):
        formatted(_value(x, kind), style)
