"""How much do Part 4's mPFC values depend on the firing-rate gate?

Part 4 reads its unit population through Figure 4's loader
(:func:`~continuum_swr_replay.analysis.pfc.load_population.load_population_data`)
with no gate on ``immobility_avg_firing_rate``, the rate over the bins the unit
GLM was fit on, and repeats a few values with a 1 Hz gate as a check. The
bundles themselves hold every curated mPFC unit with at least one spike in the
epoch: nothing is gated at fit time. So the gate can be moved without a refit.
This script loads the population once with no gate, keeps the units at or above
each gate
(:func:`~continuum_swr_replay.analysis.pfc.population_statistics.units_at_or_above_rate`)
and recomputes Part 4's per-unit statistics with the functions the summary
calls (:mod:`~continuum_swr_replay.analysis.pfc.population_statistics`), with
the summary's settings. It also counts, per gate, the units the loader drops
for another reason (no usable fit, a non-finite modulation curve), and gives the
called fractions per firing-rate band, so a low gate can be checked for units
with almost no spikes.

The gate ``none`` keeps every unit with at least one spike on the fitted bins
(rate > 0); a unit with none has no usable fit and the loader drops it anyway.

Writes, under ``--out-dir``:

- ``part4_rate_threshold_sensitivity.csv``: one row per gate, one column per
  value, plus the ``funnel_*`` counts and ``extended_sessions`` (the sessions of
  the extended event set). A value's name is the statistic's own: the function,
  the row labels of its result, and the statistic, joined by dots (for example
  ``called_fractions.non_local_SWR.fraction``, or
  ``swr_without_replay_unit_statistics.all.called_fraction``).
- ``part4_rate_threshold_sensitivity_long.csv``: the same values in long form,
  plus the per-band rows (``selection`` = ``band [lo, hi) Hz``).

Usage
-----
JAX_PLATFORMS=cpu uv run python scripts/pfc/part4_rate_threshold_sensitivity.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.pfc import population_statistics
from continuum_swr_replay.analysis.pfc.bundle_files import MPFC_STEM, read_sidecars
from continuum_swr_replay.analysis.pfc.load_population import (
    CALLED_ALPHA,
    HELDOUT_FIT_COLUMN,
    at_or_above_rate,
    load_population_data,
    significant_fraction,
)
from continuum_swr_replay.analysis.pfc.modulation import THREE_CLASS_MODEL
from continuum_swr_replay.analysis.pfc.population_statistics import (
    BEST_MODELS,
    EFFECT_CURVES,
    SWR_WITHOUT_REPLAY_CURVES,
    SWR_WITHOUT_REPLAY_MIN_EVENTS,
)
from continuum_swr_replay.analysis.pfc.timing import (
    TIMING_BOOTSTRAP_DRAWS,
    TIMING_BOOTSTRAP_SEED,
)
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY
from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.mpfc.inputs import default_inputs

#: The gates compared, by label. ``none`` is the smallest positive rate: a
#: unit with at least one spike on the fitted bins.
GATES = {
    "none": float(np.nextafter(0.0, 1.0)),
    "0.1": 0.1,
    "0.5": 0.5,
    "1.0": 1.0,
}

#: Firing-rate bands (Hz, ``[low, high)``) for the per-band called fractions.
BANDS = ((0.0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, np.inf))

#: The values shown first in the wide table.
HEADLINE = (
    "funnel_units_kept",
    "called_fractions.non_local_SWR.fraction",
    "called_fractions.non_local_no_SWR.fraction",
    "swr_without_replay_unit_statistics.all.called_fraction",
    "called_fractions.SWR.fraction",
    "called_fractions.non_local.fraction",
    "called_fractions.engagement.fraction",
    "replay_class_overlap_statistics.odds_ratio",
    "replay_class_overlap_statistics.rank_correlation",
    "replay_split_statistics.improves_fraction",
    "peak_modulation_statistics.non_local_SWR.abs_peak_median",
    "peak_modulation_statistics.non_local_no_SWR.abs_peak_median",
    "peak_modulation_statistics.SWR.abs_peak_median",
    "peak_modulation_statistics.matched_control.abs_peak_median",
)

DEFAULT_OUT_DIR = ROOT_DIR / ".claude/research/pfc_response_followups_outputs"


def _named(statistics: dict | pd.DataFrame, *prefix: str) -> dict[str, float]:
    """A statistic's result under its own names.

    Parameters
    ----------
    statistics : dict or pandas.DataFrame
        A dict of named numbers, or a table with one row per group (a plain or
        a multi-level index).
    *prefix : str
        Labels put first, the function's name and its arguments.

    Returns
    -------
    dict
        One entry per number: ``prefix``, the row labels and the statistic's
        name, joined by dots.
    """
    if isinstance(statistics, pd.DataFrame):
        statistics = {
            (*(row if isinstance(row, tuple) else (row,)), name): value
            for row, values in statistics.to_dict(orient="index").items()
            for name, value in values.items()
        }
    return {
        ".".join(
            str(label)
            for label in (*prefix, *(key if isinstance(key, tuple) else (key,)))
        ): value
        for key, value in statistics.items()
    }


def gate_values(
    population: pd.DataFrame,
    extended: pd.DataFrame,
    extended_curves: dict[str, np.ndarray],
    lag_time: np.ndarray,
    sessions: pd.DataFrame,
) -> dict[str, float]:
    """Part 4's per-unit statistics over the units of one gate.

    Each is the function the summary calls
    (:mod:`~continuum_swr_replay.analysis.pfc.population_statistics`), with the
    summary's settings, named by :func:`_named`.

    Parameters
    ----------
    population : pandas.DataFrame
        The canonical population at the gate.
    extended : pandas.DataFrame
        The extended-event-set population at the gate.
    extended_curves : dict of str to numpy.ndarray, each shape (n_lags, n_units)
        Its curves.
    lag_time : numpy.ndarray, shape (n_lags,)
        Lag of each curve row, in seconds.
    sessions : pandas.DataFrame
        The extended event set's sessions
        (:func:`~continuum_swr_replay.analysis.pfc.population_statistics.swr_without_replay_sessions`).

    Returns
    -------
    dict
        The statistics, by name.
    """
    stats = population_statistics
    settings = {"alpha": CALLED_ALPHA, "min_events": SWR_WITHOUT_REPLAY_MIN_EVENTS}
    return {
        **_named(stats.called_fractions(population, CALLED_ALPHA), "called_fractions"),
        **_named(
            stats.called_fraction_by_animal(population, "engagement", CALLED_ALPHA),
            "called_fraction_by_animal",
            "engagement",
        ),
        "bh_called_fraction.engagement": stats.bh_called_fraction(
            population, "engagement", CALLED_ALPHA
        ),
        **_named(
            stats.best_model_fractions(population, list(BEST_MODELS.values())),
            "best_model_fractions",
        ),
        **_named(
            stats.replay_class_overlap_statistics(population, CALLED_ALPHA),
            "replay_class_overlap_statistics",
        ),
        **_named(stats.replay_split_statistics(population), "replay_split_statistics"),
        **_named(
            stats.peak_modulation_statistics(population, list(EFFECT_CURVES.values())),
            "peak_modulation_statistics",
        ),
        "extended_sessions": len(sessions),
        **_named(
            stats.swr_without_replay_event_statistics(
                sessions, SWR_WITHOUT_REPLAY_MIN_EVENTS
            ),
            "swr_without_replay_event_statistics",
        ),
        **_named(
            stats.swr_without_replay_unit_statistics(
                extended, sessions, three_class_model=THREE_CLASS_MODEL, **settings
            ),
            "swr_without_replay_unit_statistics",
        ),
        **_named(
            stats.swr_without_replay_called_by_animal(extended, sessions, **settings),
            "swr_without_replay_called_by_animal",
        ),
        **_named(
            stats.swr_without_replay_timing_statistics(
                extended,
                extended_curves["swr_no_non_local"],
                lag_time,
                sessions,
                n_boot=TIMING_BOOTSTRAP_DRAWS,
                seed=TIMING_BOOTSTRAP_SEED,
                **settings,
            ),
            "swr_without_replay_timing_statistics",
        ),
    }


def spikes_on_fitted_bins(frame: pd.DataFrame, bundle_dir: Path) -> pd.Series:
    """Each unit's spike count on its session's fitted bins.

    Parameters
    ----------
    frame : pandas.DataFrame
        Units, with ``nwb_file_name``, ``epoch_name`` and
        ``immobility_avg_firing_rate``.
    bundle_dir : Path
        Directory of the bundles, whose sidecars give each session's fitted
        bins (``cohort_counts.n_bins_fit``).

    Returns
    -------
    pandas.Series, shape (n_units,)
        Rate times fitted duration, on ``frame``'s index.
    """
    bins = {
        (s["nwb_file_name"], s["epoch_name"]): s["cohort_counts"]["n_bins_fit"]
        for s in read_sidecars(bundle_dir)
    }
    n_bins = [
        bins[key]
        for key in zip(frame["nwb_file_name"], frame["epoch_name"], strict=True)
    ]
    return frame["immobility_avg_firing_rate"] * n_bins / SAMPLING_FREQUENCY


def raw_units(bundle_dir: Path) -> pd.DataFrame:
    """Every unit in the bundles, unfiltered.

    Parameters
    ----------
    bundle_dir : Path
        Directory holding ``*_mPFC_modulation.pkl``.

    Returns
    -------
    pandas.DataFrame, shape (n_units, 4)
        ``nwb_file_name``, ``epoch_name``, ``immobility_avg_firing_rate`` and
        ``has_fit`` (a finite ``SWR: heldout_deviance``, the loader's rule).
    """
    frame = pd.concat(
        [pd.read_pickle(p) for p in sorted(bundle_dir.glob(f"*{MPFC_STEM}.pkl"))],
        ignore_index=True,
    )
    out = frame[["nwb_file_name", "epoch_name", "immobility_avg_firing_rate"]].copy()
    out["has_fit"] = frame[HELDOUT_FIT_COLUMN].notna().to_numpy()
    return out


def funnel(raw: pd.DataFrame, gate: float, n_kept: int) -> dict[str, float]:
    """Why units leave the population at one gate.

    Parameters
    ----------
    raw : pandas.DataFrame
        From :func:`raw_units`.
    gate : float
        Minimum ``immobility_avg_firing_rate`` (Hz); a unit exactly at it
        passes
        (:func:`~continuum_swr_replay.analysis.pfc.load_population.at_or_above_rate`).
    n_kept : int
        Units of the loaded population (a usable fit and finite curves) at or
        above this gate.

    Returns
    -------
    dict
        ``funnel_units_in_bundles``, ``funnel_units_rate_pass``,
        ``funnel_units_no_fit`` (of the rate-passing units, those with no usable
        fit), ``funnel_units_nonfinite_curve`` (of the rate-passing, fitted
        units, those the loader drops for a non-finite modulation curve; the
        remainder, since that is the loader's only other rule) and
        ``funnel_units_kept``.
    """
    rate_pass = at_or_above_rate(raw, gate)
    fitted = rate_pass & raw["has_fit"].to_numpy()
    return {
        "funnel_units_in_bundles": len(raw),
        "funnel_units_rate_pass": int(rate_pass.sum()),
        "funnel_units_no_fit": int((rate_pass & ~raw["has_fit"].to_numpy()).sum()),
        "funnel_units_nonfinite_curve": int(fitted.sum()) - n_kept,
        "funnel_units_kept": n_kept,
    }


def band_rows(
    population: pd.DataFrame, extended: pd.DataFrame, spikes: pd.Series
) -> list[dict[str, object]]:
    """Called fractions and spike counts per firing-rate band.

    The fractions of the five canonical tests are those of
    :func:`~continuum_swr_replay.analysis.pfc.population_statistics.called_fractions`
    on the band's units. The fraction for SWRs without replay is counted the
    same way
    (:func:`~continuum_swr_replay.analysis.pfc.load_population.significant_fraction`),
    NaN when no unit of the band is tested.

    Parameters
    ----------
    population : pandas.DataFrame
        Canonical population at the lowest gate.
    extended : pandas.DataFrame
        Extended-event-set population at the lowest gate (same units).
    spikes : pandas.Series
        Spikes on the fitted bins per unit of ``population``, same order.

    Returns
    -------
    list of dict
        Long-form rows (``selection``, ``key``, ``value``).
    """
    rows: list[dict[str, object]] = []
    rate = population["immobility_avg_firing_rate"]
    ext_rate = extended["immobility_avg_firing_rate"]
    for low, high in BANDS:
        selection = f"band [{low:g}, {high:g}) Hz"
        keep = (rate >= low) & (rate < high)
        values: dict[str, float] = {
            "units": int(keep.sum()),
            "spikes_fit_median": float(spikes[keep].median()),
            "spikes_fit_min": float(spikes[keep].min()),
        }
        for name, fraction in population_statistics.called_fractions(
            population[keep], CALLED_ALPHA
        ).iterrows():
            values[f"{name}_units_tested"] = fraction["n_tested"]
            values[f"{name}_called_fraction"] = fraction["fraction"]
        n_called, n_tested = significant_fraction(
            extended[(ext_rate >= low) & (ext_rate < high)],
            CALLED_ALPHA,
            population_statistics.SWR_WITHOUT_REPLAY_P_COLUMNS,
        )[population_statistics.SWR_WITHOUT_REPLAY]
        values["SWR_no_non_local_units_tested"] = n_tested
        values["SWR_no_non_local_called_fraction"] = (
            n_called / n_tested if n_tested else np.nan
        )
        rows += [
            {"selection": selection, "key": k, "value": v} for k, v in values.items()
        ]
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    inputs = default_inputs()
    extended_dir = inputs.swr_without_replay_dir
    sessions = population_statistics.swr_without_replay_sessions(
        read_sidecars(extended_dir)
    )
    raw = raw_units(inputs.bundle_dir)
    # Every unit is loaded (no gate), whatever gate the summary itself uses.
    every_unit, _, lag_time = load_population_data(
        inputs.bundle_dir, min_firing_rate=0.0
    )
    every_extended_unit, every_extended_curve, _ = load_population_data(
        extended_dir, min_firing_rate=0.0, extra_curves=SWR_WITHOUT_REPLAY_CURVES
    )

    wide: dict[str, dict[str, float]] = {}
    long_rows: list[dict[str, object]] = []
    for label, gate in GATES.items():
        population, _ = population_statistics.units_at_or_above_rate(every_unit, gate)
        extended, extended_curves = population_statistics.units_at_or_above_rate(
            every_extended_unit, gate, every_extended_curve
        )
        if len(extended) != len(population):
            raise ValueError(
                f"gate {label}: {len(population)} canonical units against "
                f"{len(extended)} extended ones"
            )
        values = {
            **funnel(raw, gate, len(population)),
            **gate_values(population, extended, extended_curves, lag_time, sessions),
        }
        wide[label] = values
        long_rows += [
            {"selection": f"gate {label}", "key": k, "value": v}
            for k, v in values.items()
        ]
        if label == "none":
            spikes = spikes_on_fitted_bins(population, inputs.bundle_dir)
            long_rows += band_rows(population, extended, spikes)

    table = pd.DataFrame(wide).T
    table.index.name = "gate"
    ordered = [*HEADLINE, *(c for c in table.columns if c not in HEADLINE)]
    table.insert(0, "min_firing_rate_hz", [GATES[g] for g in table.index])
    table = table[["min_firing_rate_hz", *ordered]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    wide_path = args.out_dir / "part4_rate_threshold_sensitivity.csv"
    long_path = args.out_dir / "part4_rate_threshold_sensitivity_long.csv"
    table.to_csv(wide_path)
    pd.DataFrame(long_rows).to_csv(long_path, index=False)
    print(table[["min_firing_rate_hz", *HEADLINE]].T.to_string())
    print(json.dumps({"wide": str(wide_path), "long": str(long_path)}))


if __name__ == "__main__":
    main()
