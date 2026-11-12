"""Population statistics of the mPFC marked-model family bundles.

Units aggregate unit -> session -> animal, never pooled across sessions as if
they were independent: a fraction or a curve is taken per session, averaged
(or, for the share of units whose held-out deviance improves, the median taken)
over each animal's sessions, then over animals.
``scripts/pfc/summarize_pfc_marked_family.py`` reads the bundles, calls these
and writes the summary the mpfc summary builder reads.

- :func:`animal_balanced_fraction_positive`: the share of units whose held-out
  deviance improves with a target's kernel, and the deltas' scale.
- :func:`animal_balanced_curve`: the population curve of per-session curves.
- :func:`population_curve_summary`: its peak, its pre- and post-onset means
  and how many sessions and animals share their signs.
- :func:`null_band_summary`: the population curve against the same curve over
  shuffled-mark refits.
- :func:`mark_correlations`: the within-session correlation of every pair of
  marks, from the bundles' sidecars.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.shared_statistics import session_median_statistics
from continuum_swr_replay.decoding.identifiers import parse_nwb_filename

#: The max-statistic call: outside the reference when the plus-one p is at
#: most this.
MAX_P_THRESHOLD = 0.05
#: Lags of the pre-onset, post-onset and whole-window means, seconds.
PRE_ONSET_MAX_S = -0.1
POST_ONSET_WINDOW_S = (0.0, 0.15)


def mark_correlations(bundle_dir: Path) -> pd.DataFrame:
    """Correlation of every pair of marks in each model and class.

    Read from each complete bundle's sidecar
    (``marked_family_diagnostics.event_mark_geometry_by_class``: the
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
    for sidecar in sorted(bundle_dir.glob("*_marked_family.provenance.json")):
        record = json.load(open(sidecar))
        extra = record.get("extra", record)
        if extra.get("bundle_status") != "complete":
            continue
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
        The unit-level table from :func:`load_bundles`.
    blowup : float
        Absolute delta above which a unit is excluded as a blow-up.

    Returns
    -------
    pandas.DataFrame
        One row per (model, event_class, target) with ``n_units``,
        ``n_usable``, ``n_excluded_blowup``, ``n_sessions``, ``n_animals``,
        ``frac_pos`` (median over animals of the within-animal median over
        sessions of each session's share of usable units whose delta is
        positive: the ``animal_median`` of
        :func:`~continuum_swr_replay.analysis.shared_statistics.session_median_statistics`),
        ``frac_pos_min``/``frac_pos_max`` across animals,
        ``delta_median`` and ``delta_p95`` over usable units.
    """
    table = units.copy()
    table["usable"] = usable_delta(table["delta"], blowup)
    table["blowup"] = np.isfinite(table["delta"]) & (table["delta"].abs() > blowup)
    keys = ["model", "event_class", "target"]
    out = []
    for key, group in table.groupby(keys, sort=False):
        usable = group.dropna(subset=["usable"])
        per_session = (
            usable.assign(pos=usable["usable"] > 0)
            .groupby(["animal", "session"])["pos"]
            .mean()
        )
        balanced = session_median_statistics(
            per_session,
            pd.Series(
                per_session.index.get_level_values("animal"), index=per_session.index
            ),
        )
        # The range across animals of the animal medians balanced is the median of.
        per_animal = per_session.groupby(level="animal").median()
        out.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_units": int(len(group)),
                "n_usable": int(len(usable)),
                "n_excluded_blowup": int(group["blowup"].sum()),
                "n_sessions": int(balanced["n_sessions"]),
                "n_animals": int(balanced["n_animals"]),
                "frac_pos": float(balanced["animal_median"]),
                "frac_pos_min": float(per_animal.min()),
                "frac_pos_max": float(per_animal.max()),
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
        ``(model, event_class, target) -> {session: curve}`` from
        :func:`load_bundles`.
    lag_time : NDArray, shape (n_lags,)
        Lag axis in seconds.
    animal_of : dict
        ``session -> animal``.

    Returns
    -------
    pandas.DataFrame
        One row per key with ``peak_value`` (log rate per SD at the lag where
        the population curve is largest in magnitude), ``peak_lag_ms``,
        ``sem_at_peak`` over animal means (NaN with one animal),
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
        animal_values = np.array(list(animal_means.values()))
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
                "sem_at_peak": (
                    float(animal_values.std(ddof=1) / np.sqrt(len(animal_values)))
                    if len(animal_values) > 1
                    else float("nan")
                ),
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


def animal_balanced_curve(
    by_session: dict[str, np.ndarray], animal_of: dict[str, str]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Unit-mean session curves -> per-animal means -> mean over animals.

    Parameters
    ----------
    by_session : dict
        ``session -> array``; the last axis is lag (a curve, or draws x lags).
    animal_of : dict
        ``session -> animal``.

    Returns
    -------
    population : NDArray
        Mean over animals of each animal's mean over its sessions.
    per_animal : dict
        ``animal -> that animal's mean over its sessions``.
    """
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for session, curve in by_session.items():
        grouped[animal_of[session]].append(np.asarray(curve, dtype=float))
    per_animal = {
        animal: np.mean(np.stack(group), axis=0)
        for animal, group in sorted(grouped.items())
    }
    return np.mean(np.stack(list(per_animal.values())), axis=0), per_animal


def null_band_summary(
    curves: dict,
    nulls: dict,
    lag_time: np.ndarray,
    animal_of: dict[str, str],
) -> pd.DataFrame:
    """Observed population slope kernels against their shuffled-mark bands.

    For each (model, class, target), the observed population curve and each
    shuffled draw's population curve are aggregated the same way, unit ->
    session -> animal, over the sessions that carry a null (draw ``r`` of
    every session forms population draw ``r``). The band is the per-lag
    2.5th and 97.5th percentile over draws.

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
        From :func:`load_bundles`.
    lag_time : NDArray, shape (n_lags,)
        Lag axis in seconds.
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
        sessions = sorted(set(by_session) & set(curves.get(key, {})))
        if not sessions:
            continue
        n_draws = min(by_session[s].shape[0] for s in sessions)
        observed, observed_animals = animal_balanced_curve(
            {s: curves[key][s] for s in sessions}, animal_of
        )
        null, null_animals = animal_balanced_curve(
            {s: by_session[s][:n_draws] for s in sessions}, animal_of
        )
        low, high = np.nanpercentile(null, [2.5, 97.5], axis=0)
        peak = int(np.argmax(np.abs(observed)))
        observed_max = float(np.abs(observed).max())
        null_max = np.nanmax(np.abs(null), axis=1)
        max_p = (1 + np.sum(null_max >= observed_max)) / (len(null_max) + 1)
        window = observed.mean()
        window_low, window_high = np.nanpercentile(null.mean(axis=1), [2.5, 97.5])
        animals_outside = 0
        for animal, curve in observed_animals.items():
            animal_low, animal_high = np.nanpercentile(
                null_animals[animal][:, peak], [2.5, 97.5]
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
                "n_draws": int(n_draws),
                "n_null_sessions": len(sessions),
                "animals_peak_outside_band": animals_outside,
                "n_null_animals": len(observed_animals),
            }
        )
    return pd.DataFrame(out)
