"""Per-bin model terms: one (term, columns) list gives the formula, columns and window.

``PerBinRateModelSpec.terms`` replaced three hand-kept restatements (at
2d0ef3f9): ``PerBinRateModelSpec.formula``, ``per_bin_rate_model._design_columns``
and the name-prefix guess in ``longest_history_window_s``; the is_swr-contrast
ladder's parallel ``_BLOCK_F`` / ``_BLOCK_COLS`` dicts became one ``_BLOCKS``
of terms. These tests pin the new code to
``data/per_bin_terms_2d0ef3f9.json`` -- the answers the old code gave, captured
at 2d0ef3f9:

* every shipped per-bin rung (``MODEL_PER_BIN_RATE``, ``MODEL_UNCERTAINTY_RATE``,
  ``MODEL_CROSS_HISTORY_RATE``): formula string, design columns in order, and
  history window -- exactly;
* every ``SWR_CONTRAST_RATE_G_SPECS`` / ``_S_SPECS`` rung: formula, need list in
  order (repeats included) and window, plus ``SWR_CONTRAST_RATE_STOP_COVARS``
  and ``POST_OUTCOME_RATE_RUNGS`` -- exactly;
* a grid of 288 flag combinations no rung ships: formula and window exactly,
  and columns exactly except where ``reward_time_interaction`` is on without
  one of its main effects (the time spline or the reward basis). There the old
  code listed the interaction's columns at their main-effects' positions and
  the term list lists them at the interaction's, so the order differs and the
  set does not. No rung, runner or test builds such a spec, and the columns are
  only read by name.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from continuum_swr_replay.analysis.replay_models import model_specs as ms
from continuum_swr_replay.analysis.replay_models.rate.per_bin_rate_model import (
    _design_columns,
)

ANSWERS_2D0EF3F9 = json.loads(
    (
        Path(__file__).parents[3] / "support" / "data" / "per_bin_terms_2d0ef3f9.json"
    ).read_text()
)

PER_BIN_LADDERS = {
    "MODEL_PER_BIN_RATE": ms.MODEL_PER_BIN_RATE,
    "MODEL_UNCERTAINTY_RATE": ms.MODEL_UNCERTAINTY_RATE,
    "MODEL_CROSS_HISTORY_RATE": ms.MODEL_CROSS_HISTORY_RATE,
}
CONTRAST_LADDERS = {
    "SWR_CONTRAST_RATE_G_SPECS": ms.SWR_CONTRAST_RATE_G_SPECS,
    "SWR_CONTRAST_RATE_S_SPECS": ms.SWR_CONTRAST_RATE_S_SPECS,
}
# The grid's self- and cross-history settings: (lags, basis).
HISTORY = {
    "none": (0, 0),
    "lags": (ms.N_HISTORY_LAGS, 0),
    "basis": (ms.N_HISTORY_LAGS, ms.N_HISTORY_BASIS),
}


# Ladders whose self-history moved to event ends on the at-risk design
# (``PER_BIN_HISTORY_ANCHORS``) after the snapshot: their answer is the
# snapshot's with each onset self-history column renamed to its end twin.
END_ANCHORED = {"MODEL_PER_BIN_RATE", "MODEL_UNCERTAINTY_RATE", *CONTRAST_LADDERS}


def _end_anchored(answer):
    """``answer`` with the onset self-history columns renamed to their end twins."""
    rename = lambda text: re.sub(r"(?<![a-z_])hist_rcos_", "offhist_rcos_", text)  # noqa: E731
    if isinstance(answer, str):
        return rename(answer)
    if isinstance(answer, list):
        return [_end_anchored(a) for a in answer]
    if isinstance(answer, dict):
        return {k: _end_anchored(v) for k, v in answer.items()}
    return answer


def _answer(spec: ms.PerBinRateModelSpec) -> dict:
    return {
        "formula": spec.formula(),
        "columns": _design_columns(spec),
        "history_window_s": ms.longest_history_window_s([spec]),
    }


def _grid_spec(key: str) -> ms.PerBinRateModelSpec:
    flags = dict(part.split("=") for part in key.split("|"))
    self_lags, self_basis = HISTORY[flags["sh"]]
    cross_lags, cross_basis = HISTORY[flags["ch"]]
    return ms.PerBinRateModelSpec(
        name="grid",
        time_spline=bool(int(flags["ts"])),
        reward_basis=int(flags["rb"]),
        reward_rate=bool(int(flags["rr"])),
        self_history_lags=self_lags,
        switch=False,
        reward_time_interaction=bool(int(flags["rti"])),
        description="",
        parent=None,
        self_history_basis=self_basis,
        cross_history_lags=cross_lags,
        cross_history_basis=cross_basis,
        time_clock=flags["clock"],
    )


def test_every_shipped_per_bin_rung_is_reproduced() -> None:
    shipped = {
        n: (s, ladder_name in END_ANCHORED)
        for ladder_name, ladder in PER_BIN_LADDERS.items()
        for n, s in ladder.items()
    }
    assert set(shipped) == set(ANSWERS_2D0EF3F9["specs"])
    for name, (spec, end_anchored) in shipped.items():
        old = ANSWERS_2D0EF3F9["specs"][name]
        assert _answer(spec) == (_end_anchored(old) if end_anchored else old), name


def test_every_contrast_rung_is_reproduced() -> None:
    rungs = {n: r for ladder in CONTRAST_LADDERS.values() for n, r in ladder.items()}
    assert set(rungs) == set(ANSWERS_2D0EF3F9["contrast"])
    for name, (formula, columns) in rungs.items():
        assert {
            "formula": formula,
            "columns": list(columns),
            "history_window_s": ms.longest_history_window_s([(formula, columns)]),
        } == _end_anchored(ANSWERS_2D0EF3F9["contrast"][name]), name
    assert (
        list(ms.SWR_CONTRAST_RATE_STOP_COVARS)
        == ANSWERS_2D0EF3F9["SWR_CONTRAST_RATE_STOP_COVARS"]
    )
    assert (
        sorted(ms.POST_OUTCOME_RATE_RUNGS)
        == ANSWERS_2D0EF3F9["POST_OUTCOME_RATE_RUNGS"]
    )


def test_every_ladder_history_window_is_reproduced() -> None:
    windows = {
        name: ms.longest_history_window_s(ladder.values())
        for name, ladder in {**PER_BIN_LADDERS, **CONTRAST_LADDERS}.items()
    }
    windows["G+S"] = ms.longest_history_window_s(
        [*ms.SWR_CONTRAST_RATE_G_SPECS.values(), *ms.SWR_CONTRAST_RATE_S_SPECS.values()]
    )
    assert windows == ANSWERS_2D0EF3F9["ladder_windows"]


@pytest.mark.parametrize("key", sorted(ANSWERS_2D0EF3F9["grid"]))
def test_every_flag_combination_is_reproduced(key: str) -> None:
    spec = _grid_spec(key)
    old = ANSWERS_2D0EF3F9["grid"][key]
    new = _answer(spec)
    assert new["formula"] == old["formula"]
    assert new["history_window_s"] == old["history_window_s"]
    interaction_lacks_a_main_effect = spec.reward_time_interaction and not (
        spec.time_spline and spec.reward_basis
    )
    if interaction_lacks_a_main_effect:
        assert sorted(new["columns"]) == sorted(old["columns"])
        assert len(new["columns"]) == len(old["columns"])
    else:
        assert new["columns"] == old["columns"]


def test_formula_and_columns_come_from_the_same_terms() -> None:
    """Each rung's formula is its terms joined, its columns their columns."""
    for ladder in PER_BIN_LADDERS.values():
        for name, spec in ladder.items():
            terms = spec.terms()
            rhs = " + ".join(t.term for t in terms) or "1"
            assert spec.formula() == f"count ~ {rhs}", name
            term_columns = {c for t in terms for c in t.columns}
            support = {"count"}
            if spec.time_clock == "arrival":
                support.add(ms.ARRIVAL_CLOCK_COLUMN)
            assert set(spec.design_columns()) == term_columns | support, name
