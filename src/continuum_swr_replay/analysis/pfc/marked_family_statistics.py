"""Population statistics of the mPFC marked-model family bundles.

Units aggregate unit -> session -> animal, never pooled across sessions as if
they were independent: a fraction or a curve is taken per session, averaged
over each animal's sessions, then over animals.
``scripts/pfc/summarize_pfc_marked_family.py`` reads the bundles, calls these
and writes the summary the mpfc summary builder reads.

- :func:`animal_balanced_fraction_positive`: the share of units whose held-out
  deviance improves with a target's kernel, and the deltas' scale.
- The population curve of per-session curves and its standard error over
  animals are :mod:`~continuum_swr_replay.analysis.shared_statistics`'
  :func:`animal_balanced_curve` and :func:`animal_curve_sem`, which Figure 4c
  also uses.
- :func:`population_curve_summary`: its peak, its pre- and post-onset means
  and how many sessions and animals share their signs.
- :func:`shuffled_band`: the same curve over shuffled-mark refits, and its
  per-lag band; :func:`null_band_summary` sets the population curve against it.
- :func:`mark_correlations`: the within-session correlation of every pair of
  marks, from the bundles' sidecars.
- :func:`complete_bundle_records`: the complete bundles of a directory, the
  only ones :func:`mark_correlations` and the summary script read.
- :func:`summary_table`: all of these as the one long table the summary
  builder reads.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.shared_statistics import (
    animal_balanced_curve,
    animal_curve_sem,
    session_mean_curve,
)
from continuum_swr_replay.decoding.identifiers import parse_nwb_filename

#: The max-statistic call: outside the reference when the plus-one p is at
#: most this.
MAX_P_THRESHOLD = 0.05
#: Lags of the pre-onset, post-onset and whole-window means, seconds.
PRE_ONSET_MAX_S = -0.1
POST_ONSET_WINDOW_S = (0.0, 0.15)
#: Absolute held-out delta above which a unit's comparison is a blow-up (a
#: held-out prediction that exploded on one fold) and left out of
#: :func:`animal_balanced_fraction_positive`'s denominator.
DEFAULT_BLOWUP = 1000.0


def complete_bundle_records(bundle_dir: Path) -> list[tuple[Path, dict]]:
    """The sidecar of every complete bundle in a directory, in name order.

    A bundle is complete when its ``*_marked_family.provenance.json`` sidecar
    records ``bundle_status == "complete"``; a session the runner found
    ineligible or without units leaves a sidecar with another status and no
    bundle.

    Parameters
    ----------
    bundle_dir : Path
        Directory holding ``*_marked_family.pkl`` bundles and sidecars.

    Returns
    -------
    list of (Path, dict)
        Each complete bundle's sidecar path and the record's ``extra`` (the
        record itself when it has none), sorted by sidecar path.
    """
    records = []
    for sidecar in sorted(bundle_dir.glob("*_marked_family.provenance.json")):
        record = json.load(open(sidecar))
        extra = record.get("extra", record)
        if extra.get("bundle_status") == "complete":
            records.append((sidecar, extra))
    return records


def mark_correlations(bundle_dir: Path) -> pd.DataFrame:
    """Correlation of every pair of marks in each model and class.

    Read from each complete bundle's sidecar (:func:`complete_bundle_records`;
    ``marked_family_diagnostics.event_mark_geometry_by_class``: the
    within-session correlation of the standardized event marks), then
    averaged unit -> session -> animal like every other statistic here: the
    mean over animals of each animal's mean over sessions. A slope on a mark
    strongly correlated with another can carry part of that mark's effect
    under the shared ridge prior (a same-sign bias in every animal), so these
    are reported beside the slopes.

    Parameters
    ----------
    bundle_dir : Path

    Returns
    -------
    pandas.DataFrame
        ``model``, ``event_class``, ``target`` (``"<mark a> & <mark b>"``),
        ``statistic`` (``"mark_correlation"``), ``value``, ``n_sessions``.
    """
    per_session: dict = defaultdict(lambda: defaultdict(list))
    for _, extra in complete_bundle_records(bundle_dir):
        animal, _ = parse_nwb_filename(extra["nwb_file_name"])
        geometry = extra.get("marked_family_diagnostics", {}).get(
            "event_mark_geometry_by_class", {}
        )
        for event_class, entry in geometry.items():
            marks = entry.get("marks") or []
            matrix = np.asarray(entry.get("correlation") or [], dtype=float)
            if matrix.shape != (len(marks), len(marks)):
                continue
            for i in range(len(marks)):
                for j in range(i + 1, len(marks)):
                    key = (extra["model_name"], event_class, f"{marks[i]} & {marks[j]}")
                    per_session[key][animal].append(matrix[i, j])
    rows = []
    for (model, event_class, pair), by_animal in per_session.items():
        animal_means = [
            float(np.nanmean(values))
            for values in by_animal.values()
            if np.isfinite(values).any()
        ]
        rows.append(
            {
                "model": model,
                "event_class": event_class,
                "target": pair,
                "statistic": "mark_correlation",
                "value": float(np.mean(animal_means)) if animal_means else np.nan,
                "n_sessions": int(sum(len(v) for v in by_animal.values())),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["model", "event_class", "target", "statistic", "value", "n_sessions"],
    )


def usable_delta(delta: pd.Series, blowup: float) -> pd.Series:
    """Return ``delta`` with non-finite and exploding values set to NaN."""
    values = pd.to_numeric(delta, errors="coerce")
    return values.where(np.isfinite(values) & (values.abs() <= blowup))


def animal_balanced_fraction_positive(
    units: pd.DataFrame, blowup: float
) -> pd.DataFrame:
    """Unit -> session -> animal aggregation of the positive-gain fraction.

    Parameters
    ----------
    units : pandas.DataFrame
        The unit-level table, one row per unit x model x class x target, as
        ``scripts/pfc/summarize_pfc_marked_family.py`` reads it from the
        marked-family bundles.
    blowup : float
        Absolute delta above which a unit is excluded as a blow-up.

    Returns
    -------
    pandas.DataFrame
        One row per (model, event_class, target) with ``n_units``,
        ``n_usable``, ``n_excluded_blowup``, ``n_sessions``, ``n_animals``,
        ``frac_pos`` (the mean over animals of each animal's mean over its
        sessions of each session's share of usable units whose delta is
        positive: the reduction of
        :func:`~continuum_swr_replay.analysis.shared_statistics.animal_balanced_curve`,
        which the population curves use), ``frac_pos_min``/``frac_pos_max``
        across those animal means, ``delta_median`` and ``delta_p95`` over
        usable units, pooled for the deltas' scale.
    """
    table = units.copy()
    table["usable"] = usable_delta(table["delta"], blowup)
    table["blowup"] = np.isfinite(table["delta"]) & (table["delta"].abs() > blowup)
    keys = ["model", "event_class", "target"]
    out = []
    for key, group in table.groupby(keys, sort=False):
        usable = group.dropna(subset=["usable"])
        positive = (usable["usable"] > 0).to_numpy(dtype=float)
        sessions = usable["session"].to_numpy()
        by_session = {
            session: session_mean_curve(positive[None, sessions == session])
            for session in np.unique(sessions)
        }
        animal_of = dict(zip(sessions, usable["animal"], strict=True))
        frac_pos, per_animal = animal_balanced_curve(by_session, animal_of)
        animal_means = [float(value[0]) for value in per_animal.values()]
        out.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_units": int(len(group)),
                "n_usable": int(len(usable)),
                "n_excluded_blowup": int(group["blowup"].sum()),
                "n_sessions": len(by_session),
                "n_animals": len(per_animal),
                "frac_pos": float(frac_pos[0]),
                "frac_pos_min": min(animal_means),
                "frac_pos_max": max(animal_means),
                "delta_median": float(usable["usable"].median()),
                "delta_p95": float(usable["usable"].quantile(0.95)),
            }
        )
    return pd.DataFrame(out)


def population_curve_summary(
    curves: dict, lag_time: np.ndarray, animal_of: dict[str, str]
) -> pd.DataFrame:
    """Session-mean curves -> animal-balanced population curve, peak, signs.

    The population curve is unit -> session -> animal
    (:func:`animal_balanced_curve`), the weighting every population quantity
    here uses, the shuffled-mark reference included.

    Parameters
    ----------
    curves : dict
        ``(model, event_class, target) -> {session: curve}``, the per-session
        mean curves ``scripts/pfc/summarize_pfc_marked_family.py`` reads from
        the marked-family bundles.
    lag_time : NDArray, shape (n_lags,)
        Lag axis in seconds.
    animal_of : dict
        ``session -> animal``.

    Returns
    -------
    pandas.DataFrame
        One row per key with ``peak_value`` (log rate per SD at the lag where
        the population curve is largest in magnitude), ``peak_lag_ms``,
        ``sem_at_peak`` (:func:`animal_curve_sem` at that lag; NaN with one
        animal),
        ``sessions_same_sign`` (fraction of
        session-mean curves sharing the peak's sign at that lag) and
        ``animals_same_sign`` (count of animals whose mean curve does, of
        ``curve_n_animals``), ``pre_onset_mean`` (mean of the population
        curve over lags <= -100 ms), ``post_onset_mean`` (0 to +150 ms),
        ``post_onset_animals_same_sign`` (count of animals whose own 0 to
        +150 ms mean has the sign of ``post_onset_mean``) and
        ``post_minus_pre``; a curve with |post - pre| small relative to
        |pre| is a context offset, not an event-locked modulation.

        ``animals_same_sign`` is read at the lag the population curve picked,
        so the animals agree there more often than chance alone would make
        them; ``post_onset_animals_same_sign`` is read on a window fixed in
        advance and does not carry that selection.
    """
    out = []
    for (model, event_class, target), by_session in curves.items():
        stack = np.stack(list(by_session.values()))  # (n_sessions, n_lags)
        mean, per_animal = animal_balanced_curve(by_session, animal_of)
        peak = int(np.argmax(np.abs(mean)))
        # A kernel that is as large before onset as after it is a stationary
        # rate difference between the contexts the events occur in, not an
        # event-locked response; report both halves so they can be told apart.
        pre = lag_time <= PRE_ONSET_MAX_S
        post = (lag_time >= POST_ONSET_WINDOW_S[0]) & (
            lag_time <= POST_ONSET_WINDOW_S[1]
        )
        pre_mean = float(mean[pre].mean())
        post_mean = float(mean[post].mean())
        sign = np.sign(mean[peak])
        animal_means = {a: float(curve[peak]) for a, curve in per_animal.items()}
        animal_post = [float(curve[post].mean()) for curve in per_animal.values()]
        out.append(
            {
                "model": model,
                "event_class": event_class,
                "target": target,
                "peak_value": float(mean[peak]),
                "peak_lag_ms": float(lag_time[peak] * 1000),
                "pre_onset_mean": pre_mean,
                "post_onset_mean": post_mean,
                "post_minus_pre": post_mean - pre_mean,
                "sem_at_peak": float(animal_curve_sem(per_animal)[peak]),
                "sessions_same_sign": float(np.mean(np.sign(stack[:, peak]) == sign)),
                "animals_same_sign": int(
                    sum(np.sign(v) == sign for v in animal_means.values())
                ),
                "curve_n_animals": int(len(animal_means)),
                "post_onset_animals_same_sign": int(
                    sum(np.sign(v) == np.sign(post_mean) for v in animal_post)
                ),
            }
        )
    return pd.DataFrame(out)


class ShuffledBand(NamedTuple):
    """A population curve beside the same curve over shuffled-mark refits.

    Attributes
    ----------
    sessions : list of str
        Sessions with both an observed curve and null draws, sorted.
    n_draws : int
        Draws used: the fewest any of those sessions holds, so that draw ``r``
        of every session forms population draw ``r``.
    observed : NDArray, shape (n_lags,)
        The observed population curve over those sessions.
    observed_animals : dict
        ``animal -> its mean observed curve``.
    draws : NDArray, shape (n_draws, n_lags)
        Each draw's population curve.
    draw_animals : dict
        ``animal -> its mean over sessions of each draw``, shape
        (n_draws, n_lags).
    low, high : NDArray, shape (n_lags,)
        The 2.5th and 97.5th percentiles of ``draws`` at each lag.
    """

    sessions: list[str]
    n_draws: int
    observed: np.ndarray
    observed_animals: dict[str, np.ndarray]
    draws: np.ndarray
    draw_animals: dict[str, np.ndarray]
    low: np.ndarray
    high: np.ndarray


def shuffled_band(
    observed_by_session: dict[str, np.ndarray],
    null_by_session: dict[str, np.ndarray],
    animal_of: dict[str, str],
) -> ShuffledBand | None:
    """The population curve and its shuffled-mark band, over the same sessions.

    Both are aggregated unit -> session -> animal
    (:func:`animal_balanced_curve`) over the sessions that carry both an
    observed curve and null draws, each session's draws cut to the fewest any
    of them holds.

    Parameters
    ----------
    observed_by_session : dict
        ``session -> observed curve``, shape (n_lags,).
    null_by_session : dict
        ``session -> null draws``, shape (n_draws, n_lags).
    animal_of : dict
        ``session -> animal``.

    Returns
    -------
    ShuffledBand or None
        None when no session carries both.
    """
    sessions = sorted(set(null_by_session) & set(observed_by_session))
    if not sessions:
        return None
    n_draws = min(null_by_session[s].shape[0] for s in sessions)
    observed, observed_animals = animal_balanced_curve(
        {s: observed_by_session[s] for s in sessions}, animal_of
    )
    draws, draw_animals = animal_balanced_curve(
        {s: null_by_session[s][:n_draws] for s in sessions}, animal_of
    )
    low, high = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    return ShuffledBand(
        sessions, n_draws, observed, observed_animals, draws, draw_animals, low, high
    )


def null_band_summary(
    curves: dict,
    nulls: dict,
    animal_of: dict[str, str],
) -> pd.DataFrame:
    """Observed population slope kernels against their shuffled-mark bands.

    For each (model, class, target), the observed population curve and each
    shuffled draw's population curve are aggregated the same way, unit ->
    session -> animal, over the sessions that carry a null (draw ``r`` of
    every session forms population draw ``r``; :func:`shuffled_band`). The
    band is the per-lag 2.5th and 97.5th percentile over draws.

    The observed peak is the lag of the largest ``|curve|``, so comparing it
    with the band at that lag selects the most extreme of many lags. The
    max-statistic call accounts for that: each draw's largest ``|curve|``
    over the window forms the reference for the observed largest ``|curve|``,
    with the plus-one empirical p ``(1 + #draws >= observed) / (R + 1)``;
    the call is ``p <= MAX_P_THRESHOLD``. That call is the one to quote for
    "outside the reference"; the per-lag band is descriptive. The call is not "observed
    above the 95th percentile of the draws": with ``R = 30`` the interpolated
    percentile sits between the 29th and 30th largest draw, so an observed
    value exchangeable with the draws exceeds it about 8 % of the time, while
    the call (at a threshold of 0.05 and R = 30, no draw at or above it) holds
    about 3 % of the time.

    Parameters
    ----------
    curves, nulls : dict
        The per-session observed curves and per-draw shuffled-mark curves,
        keyed as in :func:`population_curve_summary`, as
        ``scripts/pfc/summarize_pfc_marked_family.py`` reads them from the
        marked-family bundles.
    animal_of : dict
        ``session -> animal``.

    Returns
    -------
    pandas.DataFrame
        One row per key with a null: ``band_low_at_peak`` /
        ``band_high_at_peak`` and ``peak_outside_band`` (per-lag band at the
        observed peak lag), ``max_abs_value`` (observed largest ``|curve|``),
        ``max_abs_null_p95`` (95th percentile of the draws' largest
        ``|curve|``, descriptive), ``max_p_value`` and
        ``max_outside_reference`` (``max_p_value <= MAX_P_THRESHOLD``),
        ``window_mean`` (over every lag) and its band
        (``window_band_low``/``window_band_high``), ``window_outside_band``,
        ``n_draws``, ``n_null_sessions``, and ``animals_peak_outside_band``
        (animals whose own curve's value at the population peak lag lies
        outside that animal's band) of ``n_null_animals``.
    """
    out = []
    for key, by_session in nulls.items():
        band = shuffled_band(curves.get(key, {}), by_session, animal_of)
        if band is None:
            continue
        observed, null, low, high = band.observed, band.draws, band.low, band.high
        peak = int(np.argmax(np.abs(observed)))
        observed_max = float(np.abs(observed).max())
        null_max = np.nanmax(np.abs(null), axis=1)
        max_p = (1 + np.sum(null_max >= observed_max)) / (len(null_max) + 1)
        window = observed.mean()
        window_low, window_high = np.nanpercentile(null.mean(axis=1), [2.5, 97.5])
        animals_outside = 0
        for animal, curve in band.observed_animals.items():
            animal_low, animal_high = np.nanpercentile(
                band.draw_animals[animal][:, peak], [2.5, 97.5]
            )
            animals_outside += int(not animal_low <= curve[peak] <= animal_high)
        model, event_class, target = key
        out.append(
            {
                "model": model,
                "event_class": event_class,
                "target": target,
                "band_low_at_peak": float(low[peak]),
                "band_high_at_peak": float(high[peak]),
                "peak_outside_band": bool(
                    not low[peak] <= observed[peak] <= high[peak]
                ),
                "max_abs_value": observed_max,
                "max_abs_null_p95": float(np.percentile(null_max, 95)),
                "max_outside_reference": bool(max_p <= MAX_P_THRESHOLD),
                "max_p_value": float(max_p),
                "window_mean": float(window),
                "window_band_low": float(window_low),
                "window_band_high": float(window_high),
                "window_outside_band": bool(not window_low <= window <= window_high),
                "n_draws": int(band.n_draws),
                "n_null_sessions": len(band.sessions),
                "animals_peak_outside_band": animals_outside,
                "n_null_animals": len(band.observed_animals),
            }
        )
    return pd.DataFrame(out)


def summary_table(
    fractions: pd.DataFrame,
    peaks: pd.DataFrame,
    bands: pd.DataFrame,
    tiers: pd.DataFrame,
    correlations: pd.DataFrame,
) -> pd.DataFrame:
    """Long table: one row per model, tier, class, target and statistic.

    Parameters
    ----------
    fractions, peaks, bands : pandas.DataFrame
        From :func:`animal_balanced_fraction_positive`,
        :func:`population_curve_summary` and :func:`null_band_summary`
        (``bands`` may be empty).
    tiers : pandas.DataFrame
        ``model`` and ``tier`` of every model.
    correlations : pandas.DataFrame
        From :func:`mark_correlations`; appended after the other statistics,
        each row with its model's tier and without ``n_sessions``.

    Returns
    -------
    pandas.DataFrame
        Columns ``model``, ``tier``, ``event_class``, ``target``,
        ``statistic`` and ``value`` (float; booleans as 0/1).
    """
    keys = ["model", "event_class", "target"]
    wide = fractions.merge(peaks, on=keys, how="outer")
    if len(bands):
        wide = wide.merge(bands, on=keys, how="left")
    wide = wide.merge(tiers.drop_duplicates("model"), on="model", how="left")
    long = wide.melt(
        id_vars=["model", "tier", "event_class", "target"],
        var_name="statistic",
        value_name="value",
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce").astype(float)
    long = long.sort_values(
        ["model", "event_class", "target", "statistic"]
    ).reset_index(drop=True)
    tiers_of = tiers.drop_duplicates("model").set_index("model")["tier"]
    return pd.concat(
        [
            long,
            correlations.drop(columns="n_sessions").assign(
                tier=correlations["model"].map(tiers_of)
            ),
        ],
        ignore_index=True,
    )
