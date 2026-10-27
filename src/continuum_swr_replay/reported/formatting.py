r"""How a reported value is printed: the project's only rounding rules.

- ``count``: an exact whole number with thousands separators (``75,675``).
- ``pct``: a fraction as a whole percent with its sign (``57 %``).
- ``pctnum``: the same percent without the sign (``57``), for ranges such as
  ``47–65 %`` where the sign is written once.
- ``int``: a quantity rounded to a whole number (a median of 37.5 spikes reads
  ``38``).
- ``sig2``: a quantity to two significant figures (``0.51``, ``1.1``, ``96``
  stays ``96``; ``1234`` reads ``1200``).
- ``dec2``: a quantity to two decimal places (``0.60``).
- ``sig2s``: ``sig2`` with its sign always written (``+0.73``, ``−0.071``), for
  coefficients whose sign is the claim. The minus is U+2212 in markdown and
  ``\ensuremath{-}`` in LaTeX.
- ``pval``: a p- or q-value, two significant figures; below 0.001 in scientific
  notation (``0.034``, ``1.1e-11``; LaTeX ``\ensuremath{1.1\times10^{-11}}``).

Rounding is half away from zero, so a printed number does not depend on
binary floating point's ties. Rounding is a communication choice, not a
statement of uncertainty: where variability matters to a claim, it is a value
of its own (an interval's bounds, a count of animals).
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

from continuum_swr_replay.reported.values import Value


def _round(x: float, places: int) -> Decimal:
    return Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP)


def _sig2(x: float) -> str:
    if x == 0:
        return "0"
    places = 1 - math.floor(math.log10(abs(float(x))))
    rounded = _round(x, places)
    if abs(rounded) >= Decimal(10) ** (2 - places):
        # rounding up gained a digit (0.0995 -> 0.100): two figures are 0.10
        rounded = _round(x, places - 1)
    return format(rounded, "f")


def formatted(value: Value, style: str | None = None, *, latex: bool = False) -> str:
    """``value`` printed in ``style`` (its own by default).

    With ``latex``, the percent sign is left out (the prose writes ``\\%``) and
    the thousands separator is ``{,}``.

    Raises
    ------
    ValueError
        ``style`` does not suit the value's kind, for example a percent of a
        count, or a count style for a value that is not whole.
    """
    style = style or value.style
    x = value.value
    if style in ("pct", "pctnum") and value.kind != "fraction":
        raise ValueError(f"{value.key}: {style} prints fractions, not {value.kind}s")
    if style == "count":
        if float(x) != int(x):
            raise ValueError(f"{value.key}: count style needs a whole number, got {x}")
        text = f"{int(x):,}"
        return text.replace(",", "{,}") if latex else text
    if style in ("pct", "pctnum"):
        number = format(_round(100 * float(x), 0), "f")
        return number if (latex or style == "pctnum") else f"{number} %"
    if style == "int":
        return format(_round(x, 0), "f")
    if style == "dec2":
        return format(_round(x, 2), "f")
    if style == "sig2":
        return _sig2(x)
    if style == "sig2s":
        text = _sig2(x)
        if text.startswith("-"):
            return ("\\ensuremath{-}" if latex else "\u2212") + text[1:]
        return "+" + text
    if style == "pval":
        if float(x) >= 0.001 or x == 0:
            return _sig2(x)
        exponent = math.floor(math.log10(float(x)))
        mantissa = Decimal(repr(float(x))).scaleb(-exponent)
        mantissa = mantissa.quantize(Decimal("0.1"), ROUND_HALF_UP)
        if mantissa >= 10:
            # rounding up gained a digit (9.96e-5 -> 10.0e-5): it is 1.0e-4
            mantissa, exponent = (mantissa / 10).quantize(Decimal("0.1")), exponent + 1
        if latex:
            return f"\\ensuremath{{{mantissa}\\times10^{{{exponent}}}}}"
        return f"{mantissa}e{exponent}"
    raise ValueError(f"{value.key}: unknown style {style!r}")
