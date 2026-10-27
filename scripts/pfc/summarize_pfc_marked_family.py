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
  the sign of that peak.

Units whose delta is non-finite, or whose |delta| exceeds ``--blowup`` (a
held-out prediction that exploded on one fold), are excluded from that
comparison's denominator and counted in the table.

Usage
-----
``uv run python scripts/pfc/summarize_pfc_marked_family.py \
    --out .claude/docs/reviews/<date>-pfc-marked-family-summary.md \
    --figure-dir paper/figures/pfc_marked_family``
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

from continuum_swr_replay.analysis.pfc.marked_glm import MARKED_EVENT_CLASSES
from continuum_swr_replay.decoding.identifiers import parse_nwb_filename
from continuum_swr_replay.paths import processed_data_dir

BUNDLE_SUBDIR = "pfc_marked_model_family"
DEFAULT_BLOWUP = 1000.0
CLASS_LABEL = {
    "non_local_no_SWR": "non-SWR replay",
    "non_local_SWR": "SWR-coupled replay",
}
#: Okabe-Ito, one colour per animal, in the corpus's animal order.
ANIMAL_COLOURS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]


def load_bundles(bundle_dir: Path) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Read every complete bundle into a unit-level table and per-session curves.

    Parameters
    ----------
    bundle_dir : Path
        Directory holding ``*_marked_family.pkl`` bundles and sidecars.

    Returns
    -------
    units : pandas.DataFrame
        One row per unit x model x class x target: ``animal``, ``session``,
        ``model``, ``event_class``, ``target``, ``delta`` (raw held-out
        delta), ``curve_finite``.
    curves : dict
        ``(model, event_class, target) -> {session: mean slope curve over
        units with finite curves, shape (n_lags,)}``.
    lag_time : NDArray, shape (n_lags,)
        The reported lag axis in seconds.
    """
    rows: list[dict] = []
    curves: dict = defaultdict(dict)
    lag_time = None
    for sidecar in sorted(bundle_dir.glob("*_marked_family.provenance.json")):
        record = json.load(open(sidecar))
        extra = record.get("extra", record)
        if extra.get("bundle_status") != "complete":
            continue
        model = extra["model_name"]
        animal, _ = parse_nwb_filename(extra["nwb_file_name"])
        session = f"{extra['nwb_file_name'].replace('_.nwb', '')}_{extra['epoch_name']}"
        base = sidecar.with_name(sidecar.name.replace(".provenance.json", ""))
        stats = pd.read_pickle(base.with_suffix(".pkl"))
        if lag_time is None:
            lag_time = np.load(base.with_suffix(".lag_time.npy"))
        targets = extra["marked_family_diagnostics"]["scientific_marks"]
        for event_class in MARKED_EVENT_CLASSES:
            for target in targets:
                delta = stats[
                    f"{event_class}:{target}: heldout_delta_deviance"
                ].to_numpy(dtype=float)
                slope = np.load(base.with_suffix(f".{event_class}.{target}.slope.npy"))
                finite_curve = np.isfinite(slope).all(axis=0)
                if finite_curve.any():
                    curves[(model, event_class, target)][session] = slope[
                        :, finite_curve
                    ].mean(axis=1)
                for unit_index in range(len(stats)):
                    rows.append(
                        {
                            "animal": animal,
                            "session": session,
                            "model": model,
                            "event_class": event_class,
                            "target": target,
                            "delta": delta[unit_index],
                            "curve_finite": bool(finite_curve[unit_index]),
                        }
                    )
    if lag_time is None:
        raise FileNotFoundError(f"no complete bundle under {bundle_dir}")
    return pd.DataFrame(rows), curves, lag_time


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
    """Session-mean curves -> population mean, peak, and sign consistency.

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
        the population mean curve is largest in magnitude), ``peak_lag_ms``,
        ``sem_at_peak`` over sessions, ``sessions_same_sign`` (fraction of
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
        mean = stack.mean(axis=0)
        peak = int(np.argmax(np.abs(mean)))
        # A kernel that is as large before onset as after it is a stationary
        # rate difference between the contexts the events occur in, not an
        # event-locked response; report both halves so they can be told apart.
        pre = lag_time <= -0.1
        post = (lag_time >= 0.0) & (lag_time <= 0.15)
        pre_mean = float(mean[pre].mean())
        post_mean = float(mean[post].mean())
        sign = np.sign(mean[peak])
        animals = defaultdict(list)
        for session, curve in by_session.items():
            animals[animal_of[session]].append(curve[peak])
        animal_means = {a: float(np.mean(v)) for a, v in animals.items()}
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
                "sem_at_peak": float(stack[:, peak].std(ddof=1) / np.sqrt(len(stack))),
                "sessions_same_sign": float(np.mean(np.sign(stack[:, peak]) == sign)),
                "animals_same_sign": int(
                    sum(np.sign(v) == sign for v in animal_means.values())
                ),
                "curve_n_animals": int(len(animal_means)),
            }
        )
    return pd.DataFrame(out)


def plot_model(
    model: str,
    curves: dict,
    lag_time: np.ndarray,
    animal_of: dict[str, str],
    figure_dir: Path,
) -> Path:
    """One figure per model: rows are targets, columns are replay classes."""
    keys = [k for k in curves if k[0] == model]
    targets = list(dict.fromkeys(k[2] for k in keys))
    animals = sorted(set(animal_of.values()))
    colour = dict(zip(animals, ANIMAL_COLOURS, strict=False))
    fig, axes = plt.subplots(
        len(targets),
        len(MARKED_EVENT_CLASSES),
        figsize=(7.0, 1.9 * len(targets) + 0.6),
        sharex=True,
        squeeze=False,
    )
    for row, target in enumerate(targets):
        for col, event_class in enumerate(MARKED_EVENT_CLASSES):
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
            stack = np.stack(list(by_session.values()))
            mean, sem = stack.mean(0), stack.std(0, ddof=1) / np.sqrt(len(stack))
            ax.fill_between(
                lag_time * 1000, mean - sem, mean + sem, color="k", alpha=0.15, lw=0
            )
            ax.plot(lag_time * 1000, mean, color="k", lw=1.6, label="all sessions")
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
        f"{model}: one-SD slope kernels (session means; black = mean ± SEM)", fontsize=9
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
        "(mean over session means ± SEM over sessions) at the lag of its largest magnitude; "
        "`sign` is how many session means and animal means share that sign at that lag. "
        "`pre / post` are the population curve's means over lags <= -100 ms and 0 to "
        "+150 ms: a kernel that is as large before onset as after is a stationary rate "
        "difference between the contexts the events occur in (the model has one intercept "
        "per session and no per-stop baseline terms), not an event-locked response. "
        "No p-values: these are descriptive.\n\n"
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
    for model, group in merged.groupby("model", sort=False):
        text += f"## {model}\n\n"
        text += _md(group, columns, fmt) + "\n"
        if model in figure_paths:
            text += f"Figure: `{figure_paths[model]}`\n\n"
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
    args = parser.parse_args()
    units, curves, lag_time = load_bundles(args.bundle_dir)
    animal_of = dict(units.drop_duplicates("session")[["session", "animal"]].to_numpy())
    fractions = animal_balanced_fraction_positive(units, args.blowup)
    peaks = population_curve_summary(curves, lag_time, animal_of)
    figure_paths = {
        model: plot_model(model, curves, lag_time, animal_of, args.figure_dir)
        for model in fractions["model"].unique()
    }
    report = build_report(fractions, peaks, args.bundle_dir, args.blowup, figure_paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    fractions.merge(peaks, on=["model", "event_class", "target"]).to_csv(
        args.out.with_suffix(".csv"), index=False
    )
    print(report)


if __name__ == "__main__":
    main()
