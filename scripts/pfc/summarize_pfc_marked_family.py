"""Population summary of the mPFC marked-model family bundles.

Reads every complete bundle written by ``run_pfc_marked_model_family.py`` and
aggregates unit -> session -> animal, never pooling units across sessions as
if they were independent. Everything here is descriptive: the design doc
attaches no p-value to a held-out delta, and none is computed.

Per model, replay class and target mark the summary reports

- the fraction of units whose held-out deviance improved when the target's
  kernel was in the model (``frac_pos``), first per session over that
  session's finite deltas, then the median over sessions within animal, then
  the median over animals (animal-balanced) with the range across animals;
- the median and 95th percentile of finite unit deltas, pooled only for
  scale;
- the population mean one-SD slope kernel (mean over session means, SEM over
  sessions), its peak value and lag, and how many sessions and animals share
  the sign of that peak;
- when the bundles carry shuffled-mark nulls (``*.slope_null.npy``), the 95 %
  band of the same population curve over the shuffled refits, and whether
  the observed peak (and window mean) lies outside it, overall and per
  animal;
- beside each joint-model target, the same target's marginal-tier result.

Bundles of one directory must share one fit contract version and one event
set; a directory that mixes them is refused. Besides the markdown report the
summary writes ``summary.parquet`` (one row per model, tier, class, target and
statistic) into the bundle directory, or to ``--summary-parquet``. Units
aggregate unit -> session -> animal there: the population curve is the mean
over animals of each animal's mean over its sessions' unit-mean curves.

Units whose delta is non-finite, or whose |delta| exceeds ``--blowup`` (a
held-out prediction that exploded on one fold), are excluded from that
comparison's denominator and counted in the table.

Usage
-----
``uv run python scripts/pfc/summarize_pfc_marked_family.py \
    --out .claude/docs/reviews/<date>-pfc-marked-family-summary.md \
    --figure-dir paper/figures/pfc_marked_family``

The follow-up corpus::

    uv run python scripts/pfc/summarize_pfc_marked_family.py \
        --bundle-dir Processed-Data/pfc_response_followups/marked_family \
        --out .claude/docs/reviews/<date>-pfc-marked-followups-summary.md \
        --figure-dir paper/figures/pfc_marked_family/followups
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.pfc.marked_glm import (
    EXTENDED_MARKED_EVENT_CLASSES,
    MARKED_EVENT_CLASSES,
)
from continuum_swr_replay.analysis.pfc.marked_model_family import (
    POOLED_CLASS,
    model_set_of,
)
from continuum_swr_replay.decoding.identifiers import parse_nwb_filename
from continuum_swr_replay.paths import processed_data_dir

BUNDLE_SUBDIR = "pfc_marked_model_family"
DEFAULT_BLOWUP = 1000.0
CLASS_LABEL = {
    "non_local_no_SWR": "non-SWR replay",
    "non_local_SWR": "SWR-coupled replay",
    "SWR_no_non_local": "SWRs without replay",
    "pooled": "all classes (shared slope)",
}
#: Order of the target classes in tables and figure columns.
CLASS_ORDER = (*EXTENDED_MARKED_EVENT_CLASSES, "pooled")
#: The max-statistic call: outside the reference when the plus-one p is at
#: most this.
MAX_P_THRESHOLD = 0.05
#: Lags of the pre-onset, post-onset and whole-window means, seconds.
PRE_ONSET_MAX_S = -0.1
POST_ONSET_WINDOW_S = (0.0, 0.15)
#: Okabe-Ito, one colour per animal, in the corpus's animal order.
ANIMAL_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]


def _model_set(model: str) -> str:
    """The model set a bundle without a ``model_set`` key belongs to."""
    try:
        return model_set_of(model)
    except KeyError:
        return "unknown"


class MixedContractError(ValueError):
    """A bundle directory holds bundles of more than one contract or event set."""


def load_bundles(
    bundle_dir: Path,
) -> tuple[pd.DataFrame, dict, dict, np.ndarray]:
    """Read every complete bundle into a unit-level table and per-session curves.

    Parameters
    ----------
    bundle_dir : Path
        Directory holding ``*_marked_family.pkl`` bundles and sidecars.

    Returns
    -------
    units : pandas.DataFrame
        One row per unit x model x class x target: ``animal``, ``session``,
        ``model``, ``tier`` (``joint`` or ``marginal``), ``event_class``,
        ``target``, ``delta`` (raw held-out delta), ``curve_finite``.
    curves : dict
        ``(model, event_class, target) -> {session: mean slope curve over
        units with finite curves, shape (n_lags,)}``.
    nulls : dict
        ``(model, event_class, target) -> {session: per-draw mean null curve
        over the same units, shape (n_draws, n_lags)}`` for bundles with a
        ``.slope_null.npy`` companion.
    lag_time : NDArray, shape (n_lags,)
        The reported lag axis in seconds.

    Raises
    ------
    MixedContractError
        If the complete bundles record more than one
        ``bundle_contract_version``, ``event_set`` or model set
        (``recommended`` versus ``followup``).
    FileNotFoundError
        If there is no complete bundle.
    """
    rows: list[dict] = []
    curves: dict = defaultdict(dict)
    nulls: dict = defaultdict(dict)
    lag_time = None
    contracts: set = set()
    event_sets: set = set()
    model_sets: set = set()
    for sidecar in sorted(bundle_dir.glob("*_marked_family.provenance.json")):
        record = json.load(open(sidecar))
        extra = record.get("extra", record)
        if extra.get("bundle_status") != "complete":
            continue
        contracts.add(extra.get("bundle_contract_version"))
        event_sets.add(extra.get("event_set", "canonical"))
        model = extra["model_name"]
        model_sets.add(extra.get("model_set") or _model_set(model))
        tier = extra.get("tier", "joint")
        animal, _ = parse_nwb_filename(extra["nwb_file_name"])
        session = f"{extra['nwb_file_name'].replace('_.nwb', '')}_{extra['epoch_name']}"
        base = sidecar.with_name(sidecar.name.replace(".provenance.json", ""))
        stats = pd.read_pickle(base.with_suffix(".pkl"))
        if lag_time is None:
            lag_time = np.load(base.with_suffix(".lag_time.npy"))
        targets = extra["marked_family_diagnostics"]["scientific_marks"]
        target_classes = extra.get("target_classes", list(MARKED_EVENT_CLASSES))
        # A class too short for the model in this session was dropped from it
        # and has no columns; it is absent, not a failed read.
        # (A pooled model's single target class is not an event class and is
        # always kept.)
        fitted = extra.get("fitted_event_classes")
        if fitted is not None:
            target_classes = [
                c for c in target_classes if c in set(fitted) or c == POOLED_CLASS
            ]
        for event_class in target_classes:
            for target in targets:
                delta = stats[
                    f"{event_class}:{target}: heldout_delta_deviance"
                ].to_numpy(dtype=float)
                slope = np.load(base.with_suffix(f".{event_class}.{target}.slope.npy"))
                finite_curve = np.isfinite(slope).all(axis=0)
                key = (model, event_class, target)
                if finite_curve.any():
                    curves[key][session] = slope[:, finite_curve].mean(axis=1)
                    null_path = base.with_suffix(
                        f".{event_class}.{target}.slope_null.npy"
                    )
                    if null_path.is_file():
                        null = np.load(null_path)[:, :, finite_curve]
                        nulls[key][session] = np.nanmean(null, axis=2)
                for unit_index in range(len(stats)):
                    rows.append(
                        {
                            "animal": animal,
                            "session": session,
                            "model": model,
                            "tier": tier,
                            "event_class": event_class,
                            "target": target,
                            "delta": delta[unit_index],
                            "curve_finite": bool(finite_curve[unit_index]),
                        }
                    )
    if lag_time is None:
        raise FileNotFoundError(f"no complete bundle under {bundle_dir}")
    if len(contracts) > 1 or len(event_sets) > 1 or len(model_sets) > 1:
        raise MixedContractError(
            f"{bundle_dir} mixes bundle contracts {sorted(map(str, contracts))}, "
            f"event sets {sorted(event_sets)} or model sets {sorted(model_sets)}; "
            "summarize one corpus at a time"
        )
    return pd.DataFrame(rows), curves, nulls, lag_time


#: Absolute mark-pair correlation from which the report flags possible ridge
#: leakage between the two marks' slopes.
STRONG_MARK_CORRELATION = 0.5


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
        sessions), ``frac_pos_min``/``frac_pos_max`` across animals,
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
        per_animal = per_session.groupby(level="animal").median()
        out.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_units": int(len(group)),
                "n_usable": int(len(usable)),
                "n_excluded_blowup": int(group["blowup"].sum()),
                "n_sessions": int(per_session.size),
                "n_animals": int(per_animal.size),
                "frac_pos": float(per_animal.median()),
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
        curve over lags <= -100 ms), ``post_onset_mean`` (0 to +150 ms) and
        ``post_minus_pre``; a curve with |post - pre| small relative to
        |pre| is a context offset, not an event-locked modulation.
    """
    out = []
    for (model, event_class, target), by_session in curves.items():
        stack = np.stack(list(by_session.values()))  # (n_sessions, n_lags)
        mean, per_animal = animal_balanced_curve(by_session, animal_of)
        peak = int(np.argmax(np.abs(mean)))
        # A kernel that is as large before onset as after it is a stationary
        # rate difference between the contexts the events occur in, not an
        # event-locked response; report both halves so they can be told apart.
        pre = lag_time <= -0.1
        post = (lag_time >= 0.0) & (lag_time <= 0.15)
        pre_mean = float(mean[pre].mean())
        post_mean = float(mean[post].mean())
        sign = np.sign(mean[peak])
        animal_means = {a: float(curve[peak]) for a, curve in per_animal.items()}
        animal_values = np.array(list(animal_means.values()))
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
    the call is ``p <= 0.05``. That call is the one to quote for "outside the
    reference"; the per-lag band is descriptive. The call is not "observed
    above the 95th percentile of the draws": with ``R = 30`` the interpolated
    percentile sits between the 29th and 30th largest draw, so an observed
    value exchangeable with the draws exceeds it about 8 % of the time, while
    ``p <= 0.05`` (no draw at or above it, at R = 30) holds about 3 % of the
    time.

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
        ``max_outside_reference`` (``max_p_value <= 0.05``),
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


def summary_table(
    fractions: pd.DataFrame,
    peaks: pd.DataFrame,
    bands: pd.DataFrame,
    tiers: pd.DataFrame,
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
    return long.sort_values(
        ["model", "event_class", "target", "statistic"]
    ).reset_index(drop=True)


def joint_versus_marginal(merged: pd.DataFrame) -> pd.DataFrame:
    """Each joint-model target beside its marginal-tier fit, per class.

    Parameters
    ----------
    merged : pandas.DataFrame
        Fractions merged with peaks, carrying ``tier``.

    Returns
    -------
    pandas.DataFrame
        One row per joint model, class and target that has a marginal fit:
        ``joint_post_onset_mean``, ``marginal_post_onset_mean``,
        ``joint_frac_pos`` and ``marginal_frac_pos``.
    """
    joint = merged.loc[merged["tier"] == "joint"]
    marginal = merged.loc[merged["tier"] == "marginal"]
    paired = joint.merge(
        marginal[["event_class", "target", "post_onset_mean", "frac_pos"]],
        on=["event_class", "target"],
        suffixes=("", "_marginal"),
    )
    return pd.DataFrame(
        {
            "model": paired["model"],
            "event_class": paired["event_class"],
            "target": paired["target"],
            "joint_post_onset_mean": paired["post_onset_mean"],
            "marginal_post_onset_mean": paired["post_onset_mean_marginal"],
            "joint_frac_pos": paired["frac_pos"],
            "marginal_frac_pos": paired["frac_pos_marginal"],
        }
    )


def plot_model(
    model: str,
    curves: dict,
    lag_time: np.ndarray,
    animal_of: dict[str, str],
    figure_dir: Path,
    nulls: dict | None = None,
) -> Path:
    """One figure per model: rows are targets, columns are its classes.

    The shuffled-mark band (grey) sits behind the observed population curve
    when the bundles carry nulls; a target's marginal-tier population curve,
    when fit, is overlaid dashed.
    """
    nulls = nulls or {}
    keys = [k for k in curves if k[0] == model]
    targets = list(dict.fromkeys(k[2] for k in keys))
    classes = [c for c in CLASS_ORDER if any(k[1] == c for k in keys)]
    animals = sorted(set(animal_of.values()))
    colour = dict(zip(animals, ANIMAL_COLOURS, strict=False))
    fig, axes = plt.subplots(
        len(targets),
        len(classes),
        figsize=(3.5 * len(classes), 1.9 * len(targets) + 0.6),
        sharex=True,
        squeeze=False,
    )
    for row, target in enumerate(targets):
        for col, event_class in enumerate(classes):
            ax = axes[row, col]
            by_session = curves.get((model, event_class, target), {})
            if not by_session:
                ax.set_visible(False)
                continue
            per_animal = defaultdict(list)
            for session, curve in by_session.items():
                per_animal[animal_of[session]].append(curve)
            for animal, group in per_animal.items():
                ax.plot(
                    lag_time * 1000,
                    np.mean(group, axis=0),
                    color=colour[animal],
                    lw=0.8,
                    alpha=0.8,
                    label=f"{animal} (n={len(group)})",
                )
            null = nulls.get((model, event_class, target))
            if null:
                draws, _ = animal_balanced_curve(
                    {s: v for s, v in null.items() if s in by_session}, animal_of
                )
                low, high = np.nanpercentile(draws, [2.5, 97.5], axis=0)
                ax.fill_between(
                    lag_time * 1000,
                    low,
                    high,
                    color="0.75",
                    lw=0,
                    label="shuffled 95 %",
                )
            mean, animal_curves = animal_balanced_curve(by_session, animal_of)
            animal_stack = np.stack(list(animal_curves.values()))
            sem = (
                animal_stack.std(0, ddof=1) / np.sqrt(len(animal_stack))
                if len(animal_stack) > 1
                else 0
            )
            ax.fill_between(
                lag_time * 1000, mean - sem, mean + sem, color="k", alpha=0.15, lw=0
            )
            ax.plot(lag_time * 1000, mean, color="k", lw=1.6, label="all animals")
            marginal = curves.get((f"marginal_{target}", event_class, target))
            if marginal and not model.startswith("marginal_"):
                ax.plot(
                    lag_time * 1000,
                    animal_balanced_curve(marginal, animal_of)[0],
                    color="k",
                    lw=1.0,
                    ls="--",
                    label="marginal tier",
                )
            ax.axhline(0, color="0.6", lw=0.6)
            ax.axvline(0, color="0.6", lw=0.6)
            if row == 0:
                ax.set_title(CLASS_LABEL[event_class], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{target}\nlog rate / SD", fontsize=7)
            ax.tick_params(labelsize=7)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    axes[-1, 0].set_xlabel("lag from replay onset (ms)", fontsize=8)
    axes[-1, -1].set_xlabel("lag from replay onset (ms)", fontsize=8)
    axes[0, 0].legend(fontsize=6, frameon=False, loc="upper left")
    fig.suptitle(
        f"{model}: one-SD slope kernels (animal means; black = mean over animals "
        "± SEM over animals; grey = shuffled-mark 95 %)",
        fontsize=9,
    )
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"{model}_slopes.png"
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def _md(frame: pd.DataFrame, columns: list[str], fmt: dict[str, str]) -> str:
    head = "| " + " | ".join(columns) + " |\n|" + "---|" * len(columns) + "\n"
    body = ""
    for _, row in frame.iterrows():
        cells = [fmt.get(c, "{}").format(row[c]) for c in columns]
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def build_report(
    fractions: pd.DataFrame,
    peaks: pd.DataFrame,
    bundle_dir: Path,
    blowup: float,
    figure_paths: dict[str, Path],
    bands: pd.DataFrame | None = None,
    pairs: pd.DataFrame | None = None,
) -> str:
    """Render the markdown report."""
    merged = fractions.merge(peaks, on=["model", "event_class", "target"], how="left")
    merged["class"] = merged["event_class"].map(CLASS_LABEL)
    merged["frac_pos_range"] = merged.apply(
        lambda r: f"{r.frac_pos:.2f} [{r.frac_pos_min:.2f}–{r.frac_pos_max:.2f}]",
        axis=1,
    )
    merged["peak"] = merged.apply(
        lambda r: f"{r.peak_value:+.3f} ± {r.sem_at_peak:.3f} at {r.peak_lag_ms:+.0f} ms",
        axis=1,
    )
    merged["pre_post"] = merged.apply(
        lambda r: f"{r.pre_onset_mean:+.3f} / {r.post_onset_mean:+.3f}", axis=1
    )
    merged["sign"] = merged.apply(
        lambda r: f"{r.sessions_same_sign:.2f} sessions, {int(r.animals_same_sign)}/{int(r.curve_n_animals)} animals",
        axis=1,
    )
    text = (
        "# mPFC marked-model family: population summary\n\n"
        f"Generated by `scripts/pfc/summarize_pfc_marked_family.py` over `{bundle_dir}`. "
        "Units aggregate to sessions, sessions to animals; `frac_pos` is the median over "
        "animals of the within-animal median over sessions of the fraction of units whose "
        "held-out deviance improved with the target's kernel, with the range across animals. "
        f"Units with a non-finite delta or |delta| > {blowup:g} are excluded from that "
        "comparison and counted. `peak` is the population mean one-SD slope kernel "
        "(unit -> session -> animal: mean over animals of each animal's mean over its "
        "sessions, ± SEM over animals) at the lag of its largest magnitude; "
        "`sign` is how many session means and animal means share that sign at that lag. "
        "`pre / post` are the population curve's means over lags <= -100 ms and 0 to "
        "+150 ms: a kernel that is as large before onset as after is a stationary rate "
        "difference between the contexts the events occur in, not an event-locked "
        "response (contract-2 designs carry per-stop well and reward-outcome baseline "
        "columns, so their typical kernel is the response at the reference well after "
        "an unrewarded, known outcome). "
        "No per-unit p-values: the only p is the shuffled-mark reference's for the "
        "population curve, a descriptive check.\n\n"
    )
    columns = [
        "class",
        "target",
        "n_usable",
        "n_excluded_blowup",
        "frac_pos_range",
        "delta_median",
        "delta_p95",
        "peak",
        "pre_post",
        "sign",
    ]
    fmt = {"delta_median": "{:.2f}", "delta_p95": "{:.1f}"}
    if bands is not None and len(bands):
        merged = merged.merge(
            bands[
                [
                    "model",
                    "event_class",
                    "target",
                    "band_low_at_peak",
                    "band_high_at_peak",
                    "peak_outside_band",
                    "max_outside_reference",
                    "max_p_value",
                    "animals_peak_outside_band",
                    "n_null_animals",
                ]
            ],
            on=["model", "event_class", "target"],
            how="left",
        )
        merged["reference"] = merged.apply(
            lambda r: (
                ""
                if pd.isna(r.band_low_at_peak)
                else f"max abs curve {'outside' if r.max_outside_reference else 'inside'} "
                f"(p {r.max_p_value:.3f}); per-lag band at peak "
                f"[{r.band_low_at_peak:+.3f}, {r.band_high_at_peak:+.3f}] "
                f"{'outside' if r.peak_outside_band else 'inside'}; "
                f"{int(r.animals_peak_outside_band)}/{int(r.n_null_animals)} animals outside"
            ),
            axis=1,
        )
        columns = [*columns, "reference"]
        text += (
            "`reference` compares the population curve (unit -> session -> animal) "
            "with the same curve over shuffled-mark refits. The quoted call is the "
            "max statistic: the observed largest |curve| over the window against "
            "each draw's largest |curve|, outside when the plus-one p "
            f"(1 + #draws >= observed) / (R + 1) is at most {MAX_P_THRESHOLD}. "
            "The per-lag band at the observed peak lag is descriptive (the peak "
            "lag is selected as the most extreme).\n\n"
        )
    for model, group in merged.groupby("model", sort=False):
        text += f"## {model}\n\n"
        text += _md(group, columns, fmt) + "\n"
        if model in figure_paths:
            text += f"Figure: `{figure_paths[model]}`\n\n"
    if pairs is not None and len(pairs):
        text += "## Joint versus marginal\n\n"
        pairs = pairs.assign(**{"class": pairs["event_class"].map(CLASS_LABEL)})
        text += _md(
            pairs,
            [
                "model",
                "class",
                "target",
                "joint_post_onset_mean",
                "marginal_post_onset_mean",
                "joint_frac_pos",
                "marginal_frac_pos",
            ],
            {
                "joint_post_onset_mean": "{:+.3f}",
                "marginal_post_onset_mean": "{:+.3f}",
                "joint_frac_pos": "{:.2f}",
                "marginal_frac_pos": "{:.2f}",
            },
        )
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--bundle-dir", type=Path, default=processed_data_dir() / BUNDLE_SUBDIR
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--figure-dir", type=Path, default=Path("paper/figures/pfc_marked_family")
    )
    parser.add_argument("--blowup", type=float, default=DEFAULT_BLOWUP)
    parser.add_argument(
        "--summary-parquet",
        type=Path,
        help="Long statistics table (default: <bundle-dir>/summary.parquet).",
    )
    args = parser.parse_args()
    units, curves, nulls, lag_time = load_bundles(args.bundle_dir)
    animal_of = dict(units.drop_duplicates("session")[["session", "animal"]].to_numpy())
    fractions = animal_balanced_fraction_positive(units, args.blowup)
    peaks = population_curve_summary(curves, lag_time, animal_of)
    bands = null_band_summary(curves, nulls, lag_time, animal_of)
    tiers = units[["model", "tier"]].drop_duplicates()
    merged = fractions.merge(peaks, on=["model", "event_class", "target"]).merge(
        tiers, on="model"
    )
    pairs = joint_versus_marginal(merged)
    figure_paths = {
        model: plot_model(model, curves, lag_time, animal_of, args.figure_dir, nulls)
        for model in fractions["model"].unique()
    }
    report = build_report(
        fractions, peaks, args.bundle_dir, args.blowup, figure_paths, bands, pairs
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    fractions.merge(peaks, on=["model", "event_class", "target"]).to_csv(
        args.out.with_suffix(".csv"), index=False
    )
    correlations = mark_correlations(args.bundle_dir)
    if len(correlations):
        strong = correlations[correlations["value"].abs() >= STRONG_MARK_CORRELATION]
        report += (
            "\n\n## Correlated marks\n\n"
            "Pairs of marks in one model and class whose within-session correlation "
            f"(mean over animals of session means) is at least {STRONG_MARK_CORRELATION} "
            "in absolute value. Under the shared ridge prior a small slope on either "
            "mark of such a pair can carry part of the other's effect, with the same "
            "sign in every animal; read it as possible leakage.\n\n"
            + _md(
                strong.rename(columns={"target": "pair"}),
                ["model", "event_class", "pair", "value", "n_sessions"],
                {"value": "{:+.2f}"},
            )
        )
        args.out.write_text(report)
    summary_path = args.summary_parquet or args.bundle_dir / "summary.parquet"
    tiers_of = tiers.drop_duplicates("model").set_index("model")["tier"]
    pd.concat(
        [
            summary_table(fractions, peaks, bands, tiers),
            correlations.drop(columns="n_sessions").assign(
                tier=correlations["model"].map(tiers_of)
            ),
        ],
        ignore_index=True,
    ).to_parquet(summary_path, index=False)
    print(report)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
