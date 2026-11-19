#!/usr/bin/env python
"""Fit three replay-CONTENT estimators on one at-well cohort and compare them.

The content question -- which of the nine arms an immobility non-local event
represents -- has been asked with three different likelihoods, and the three do
not agree about how much an event is worth:

``conditional_poisson_count``
    The retired count model. Its conditioning group is the event, but its
    observation is the timebin, so an event that spent 80 bins carries 80 times
    the weight of an event that spent one, and its standard errors assume those
    bins are independent draws.
``conditional_poisson_fraction``
    The shipped model. The response is the event's per-arm allocation
    *fraction*, so every event weighs one, and the reported uncertainty is the
    event-clustered sandwich (``bse_cluster_<term>``) rather than the model SE.
    Its likelihood is a quasi-likelihood on the event scale: the AIC it implies
    is not a count-scale AIC and is labeled as such here.
``dirichlet_multinomial``
    The overdispersion-aware model. It keeps the counts, keeps the softmax mean
    structure, and adds an explicit within-event concentration ``log phi_e = g0
    + g1 * is_swr_e``. Its standard errors are event-level by construction, and
    ``coef_log_phi_is_swr`` is a direct test of whether SWR events spread their
    mass over more arms than non-SWR events.

All three are fit **on exactly the same events**: the cohort rules of the
shipped contrast runner are applied once per session for *both* response
columns and intersected, so a difference between estimators is a difference of
likelihood and never of cohort. The per-(session, rung) event counts are
asserted equal across the three, and the events the intersection dropped are
recorded.

Cross-estimator AIC is only meaningful once the likelihood constants match.
The statsmodels conditional likelihood omits the multinomial coefficient and
the Dirichlet-multinomial keeps it, so ``multinomial_log_constant`` -- computed
from the COUNT matrix in both cases -- is added to the two conditional fits
before ``aic_equalized`` is formed. ``pseudo_r2`` has a different null under
each estimator and is deliberately not tabulated.

Inference is the project's standard: five animals, sign consistency, and a
95 % interval from a nested bootstrap (resample the animals present with
replacement, then each drawn animal's sessions; statistic = median over the
drawn animals of each one's median over its drawn sessions): the median plus or
minus a small-sample t correction over the animals times the draws' standard
deviation.

Usage:
    uv run python scripts/replay_models/content/compare_content_estimators.py
    uv run python scripts/replay_models/content/compare_content_estimators.py --max-per-animal 1 --n-jobs 4
    uv run python scripts/replay_models/content/compare_content_estimators.py --rungs C2_base
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
import warnings
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from continuum_swr_replay.analysis.replay_models.content.dirichlet_multinomial import (  # noqa: E402
    DEFAULT_MIN_EVENTS_SESSION_PHI,
    DEFAULT_N_JOBS,
    LOG_PHI_INTERCEPT,
)
from continuum_swr_replay.analysis.replay_models.content.dirichlet_multinomial import (  # noqa: E402
    ESTIMATOR as DIRICHLET_MULTINOMIAL,
)
from continuum_swr_replay.analysis.replay_models.content.estimator_comparison import (  # noqa: E402
    BOOTSTRAP_BASENAME,
    BOOTSTRAP_SEED,
    COEFFICIENTS_BASENAME,
    COUNT_RESPONSE,
    ESTIMATOR_LABELS,
    ESTIMATORS,
    FRACTION_RESPONSE,
    LOG_PHI_IS_SWR,
    N_BOOTSTRAP,
    QUASI_SCALE,
    SKIPS_BASENAME,
    SUMMARY_BASENAME,
    identified_rows,
    n_fits_delta_aic_row_mismatch,
    n_unpaired_triples,
    run_comparison,
    wide_by_estimator,
)

# The cohort this comparison runs on is the shipped contrast runner's own at-well
# cohort, assembled by the same helpers rather than re-derived here, so the two
# cannot drift.
from continuum_swr_replay.analysis.replay_models.contrast_aggregation import (  # noqa: E402
    assemble_arm,
    contrast_fit_inputs,
    select_outcome_stratum,
)
from continuum_swr_replay.analysis.replay_models.fitting import (  # noqa: E402
    CONDITIONAL_POISSON_FRACTION,
)
from continuum_swr_replay.analysis.replay_models.model_specs import (  # noqa: E402
    CONTENT_ESTIMATOR_COMPARISON_RUNGS,
    ComparisonRung,
    resolve_comparison_rungs,
)
from continuum_swr_replay.paths import (
    CONTENT_ESTIMATOR_COMPARISON_DIR,
    figure_dir,
    replay_content_dir,
)
from continuum_swr_replay.runs.fit_provenance import (  # noqa: E402
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.tables.io import write_table  # noqa: E402
from continuum_swr_replay.tables.session_keys import (  # noqa: E402
    SESSION_COLUMNS,
    subset_max_per_animal,
)

# Quasi-separated coefficients legitimately overflow exp() outside the fit. Fit
# quality itself is carried per fit as converged/separated flags, so only the
# benign RuntimeWarning noise is silenced -- not a blanket ignore.
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SC = list(SESSION_COLUMNS)

PROVENANCE_ESTIMATOR = "three_estimator_comparison"

#: Rung whose Dirichlet-multinomial concentration the log-phi figure shows.
DEFAULT_LOG_PHI_RUNG = "C2_base"


def log_phi_figure_rung(rungs: Sequence[ComparisonRung]) -> str | None:
    """The rung the concentration figure is drawn for, or ``None`` if unselected.

    ``--rungs`` can legitimately select a set that excludes
    :data:`DEFAULT_LOG_PHI_RUNG`. That is a partial run, not a failure: the
    figure is pinned to one rung because its caption names that rung, and
    silently substituting another would publish a figure whose caption describes
    a different model. Returning ``None`` lets the caller skip the panel and
    still finish writing its tables and provenance.
    """
    return (
        DEFAULT_LOG_PHI_RUNG
        if DEFAULT_LOG_PHI_RUNG in {rung.name for rung in rungs}
        else None
    )


# ── figures ─────────────────────────────────────────────────────────────────


def _animal_colors(animals: Sequence[str]) -> dict:
    """A stable color per animal, in sorted order."""
    palette = plt.get_cmap("tab10")
    return {animal: palette(i % 10) for i, animal in enumerate(sorted(animals))}


def plot_coefficient_agreement(
    coefficients: pd.DataFrame, rungs: Sequence[ComparisonRung], path: Path
) -> Path:
    """Per key term, per-session count-vs-fraction and count-vs-DM scatters."""
    identified = identified_rows(coefficients)
    wide = wide_by_estimator(identified) if not identified.empty else pd.DataFrame()
    pairs = [
        (rung.name, rung.key_term)
        for rung in rungs
        if rung.key_term is not None
        and not wide.empty
        and not wide[
            (wide["model"] == rung.name) & (wide["term"] == rung.key_term)
        ].empty
    ]
    n_rows = max(len(pairs), 1)
    fig, axes = plt.subplots(
        n_rows, 2, figsize=(7.0, 3.0 * n_rows), squeeze=False, constrained_layout=True
    )
    colors = _animal_colors(coefficients["animal"].unique()) if not wide.empty else {}
    others = [
        (CONDITIONAL_POISSON_FRACTION, "conditional Poisson, fractions"),
        (DIRICHLET_MULTINOMIAL, "Dirichlet-multinomial"),
    ]
    for row, (model, term) in enumerate(pairs):
        subset = wide[(wide["model"] == model) & (wide["term"] == term)]
        for column, (estimator, label) in enumerate(others):
            ax = axes[row][column]
            other = ESTIMATOR_LABELS[estimator]
            for animal, group in subset.groupby("animal", sort=True):
                ax.scatter(
                    group["coef_count"],
                    group[f"coef_{other}"],
                    s=14,
                    alpha=0.75,
                    color=colors.get(animal),
                    label=animal if row == 0 and column == 0 else None,
                )
            finite = np.concatenate(
                [
                    subset["coef_count"].to_numpy(dtype=float),
                    subset[f"coef_{other}"].to_numpy(dtype=float),
                ]
            )
            finite = finite[np.isfinite(finite)]
            if finite.size:
                lo, hi = float(finite.min()), float(finite.max())
                pad = 0.05 * (hi - lo) if hi > lo else 0.5
                limits = (lo - pad, hi + pad)
                ax.plot(limits, limits, color="0.4", lw=0.8, ls="--", zorder=0)
                # Shared limits, so the dashed line reads as the 45-degree
                # identity rather than an arbitrary diagonal.
                ax.set_xlim(limits)
                ax.set_ylim(limits)
                ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("conditional Poisson, counts")
            ax.set_ylabel(label)
            ax.set_title(
                f"{model}\n{textwrap.fill(term, 34)}", fontsize=7, linespacing=1.2
            )
            ax.tick_params(labelsize=7)
    if pairs:
        axes[0][0].legend(fontsize=6, frameon=False, title="animal", title_fontsize=6)
    else:
        for ax in axes.ravel():
            ax.set_axis_off()
        axes[0][0].text(0.5, 0.5, "no identified key-term fits", ha="center")
    fig.suptitle("Per-session content coefficients by estimator", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_log_phi_by_session(
    coefficients: pd.DataFrame,
    bootstrap: pd.DataFrame,
    path: Path,
    *,
    rung: str = DEFAULT_LOG_PHI_RUNG,
) -> Path:
    """Concentration intercept and its is_swr modifier, session by session.

    Raises
    ------
    ValueError
        If ``rung`` has no Dirichlet-multinomial rows. The caption names the
        rung the panels show, so quietly substituting another one would publish
        a figure whose caption describes a different model.
    """
    identified = identified_rows(coefficients)
    dm = identified[
        (identified["estimator"] == DIRICHLET_MULTINOMIAL)
        & (identified["model"] == rung)
    ]
    if dm.empty:
        available = sorted(
            identified.loc[
                identified["estimator"] == DIRICHLET_MULTINOMIAL, "model"
            ].unique()
        )
        raise ValueError(
            f"no identified Dirichlet-multinomial fits for rung {rung!r}; the "
            f"concentration figure cannot be drawn (rungs with DM fits: "
            f"{available})"
        )

    terms = [
        (LOG_PHI_INTERCEPT, "log φ intercept (non-SWR)"),
        (LOG_PHI_IS_SWR, "log φ shift on SWR events"),
    ]
    fig, axes = plt.subplots(
        2, 1, figsize=(7.0, 5.0), sharex=True, constrained_layout=True
    )
    colors = _animal_colors(coefficients["animal"].unique()) if not dm.empty else {}
    for ax, (term, label) in zip(axes, terms, strict=True):
        subset = dm[dm["term"] == term].sort_values([*SC])
        for animal, group in subset.groupby("animal", sort=True):
            order = np.arange(len(group))
            ax.plot(
                order,
                group["coef"].to_numpy(dtype=float),
                marker="o",
                ms=3.5,
                lw=0.8,
                color=colors.get(animal),
                label=animal,
            )
        band = bootstrap[
            (bootstrap["model"] == rung)
            & (bootstrap["term"] == term)
            & (bootstrap["estimator"] == DIRICHLET_MULTINOMIAL)
        ]
        if not band.empty:
            row = band.iloc[0]
            ax.axhspan(
                float(row["low"]),
                float(row["high"]),
                color="0.75",
                alpha=0.4,
                zorder=0,
            )
            ax.axhline(float(row["statistic"]), color="0.3", lw=0.9, zorder=1)
        if term == LOG_PHI_IS_SWR:
            ax.axhline(0.0, color="k", lw=0.6, ls=":", zorder=1)
        ax.set_ylabel(label, fontsize=8)
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel("session order within animal")
    axes[0].legend(fontsize=6, frameon=False, ncol=5, title="animal", title_fontsize=6)
    fig.suptitle(
        f"Dirichlet-multinomial concentration, {rung} "
        "(band = 95 % nested-bootstrap interval)",
        fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


CAPTION = """# Content estimators compared

**Panels.** `coef_agreement.png` — one row per model rung, showing that rung's
key coefficient for every session that all three estimators fit. The x axis is
the retired conditional Poisson on per-arm timebin counts; the left column's y
axis is the shipped conditional Poisson on per-event allocation fractions
(event-clustered standard errors), the right column's is the
Dirichlet-multinomial on the same counts with an `is_swr` concentration
modifier. The dashed line is the identity. Color identifies the animal. All
three estimators were fit on one shared per-(session, rung) event set, so a
departure from the identity is a difference of likelihood, not of cohort.
Unidentified fits are excluded, by the same rule the shipped aggregation uses:
the fitter flagged that term as separated, or the effect-scale magnitude
|coefficient| × SD(predictor) reaches 10. That effect-scale backstop applies to
the concentration modifier `log_phi_is_swr` as well, on the SD of the event-level
`is_swr` flag; `log_phi_intercept` is the one term it cannot apply to, because
its design column is a constant with no SD, so there the separation flag is the
whole criterion. The drop is applied to the whole (session, rung, term) triple,
so the three estimators are always compared over the same sessions.

`log_phi_by_session.png` — the Dirichlet-multinomial concentration for the
`C2_base` rung. **Top,** the log concentration intercept, i.e. how tightly a
non-SWR event's timebins cluster on a few arms; **bottom,** the additive shift
of that log concentration on SWR events, where a value below zero means SWR
events are *more* concentrated (more redundant bins) and above zero means their
mass is spread over more arms. Each line is one animal, plotted in session order
within that animal. The gray line is the median over animals of each animal's
median, and the gray band its 95 % interval from a nested bootstrap: the animals
present are resampled with replacement, then each drawn animal's sessions, for
2000 draws at **seed 0**, and each draw's statistic is the median over the drawn
animals of each one's median over its drawn sessions. The band is the median plus
or minus t(0.975, A - 1) sqrt(A / (A - 1)) times the draws' standard deviation,
over the A animals.
"""


def write_caption(path: Path) -> Path:
    """Write the tracked caption markdown beside the figures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CAPTION)
    return path


# ── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--rungs",
        default=None,
        help="comma-separated rung names (default: every comparison rung); "
        f"available: {','.join(rung.name for rung in CONTENT_ESTIMATOR_COMPARISON_RUNGS)}",
    )
    parser.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help="smoke subset: keep the first N sessions per animal",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help="worker processes for the Dirichlet-multinomial session fits",
    )
    parser.add_argument(
        "--min-events-session-phi",
        type=int,
        default=DEFAULT_MIN_EVENTS_SESSION_PHI,
        help="sessions below this many events get a pooled concentration intercept",
    )
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=replay_content_dir() / CONTENT_ESTIMATOR_COMPARISON_DIR,
    )
    parser.add_argument(
        "--figures-dir", type=Path, default=figure_dir("content_estimators")
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Assemble the shipped at-well cohort, compare the estimators, write outputs."""
    argv = sys.argv[1:] if argv is None else list(argv)
    args = _parse_args(argv)
    rungs = resolve_comparison_rungs(
        [name.strip() for name in args.rungs.split(",") if name.strip()]
        if args.rungs
        else None
    )

    logger.info("loading the at-well arm cohort")
    contrast_inputs = contrast_fit_inputs()
    ev, _z_reference = assemble_arm(contrast_inputs)
    inputs = record_inputs(contrast_inputs)
    # Only the arm cohort is compared, so the patch cohort is not assembled:
    # this is prepare_cohorts' arm half, for the "all" stratum.
    ev, stratum_audit = select_outcome_stratum(ev, "all")
    if args.max_per_animal is not None:
        ev = subset_max_per_animal(ev, args.max_per_animal)
    logger.info(
        "cohort: %d events over %d sessions",
        ev["event_id"].nunique(),
        len(ev[SC].drop_duplicates()),
    )

    result = run_comparison(
        ev,
        rungs,
        n_jobs=args.n_jobs,
        min_events_session_phi=args.min_events_session_phi,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficients_path = write_table(
        result.coefficients, output_dir / COEFFICIENTS_BASENAME
    )
    result.summary.to_csv(output_dir / f"{SUMMARY_BASENAME}.csv", index=False)
    result.bootstrap.to_csv(output_dir / f"{BOOTSTRAP_BASENAME}.csv", index=False)
    result.skips.to_csv(output_dir / f"{SKIPS_BASENAME}.csv", index=False)

    figures_dir = Path(args.figures_dir)
    plot_coefficient_agreement(
        result.coefficients, rungs, figures_dir / "coef_agreement.png"
    )
    # The concentration figure is pinned to one rung, and --rungs can select a
    # set that does not include it. That is a legitimate partial run, not a
    # failure: raising here would abort AFTER the tables were written and BEFORE
    # the provenance sidecar, leaving artifacts no reader could date or trace.
    if log_phi_figure_rung(rungs) is not None:
        plot_log_phi_by_session(
            result.coefficients,
            result.bootstrap,
            figures_dir / "log_phi_by_session.png",
        )
    else:
        logger.info(
            "skipping the concentration figure: it is drawn for %s, which is "
            "not among the selected rungs (%s)",
            DEFAULT_LOG_PHI_RUNG,
            ", ".join(rung.name for rung in rungs),
        )
    write_caption(figures_dir / "content_estimators_caption.md")

    write_fit_provenance(
        coefficients_path,
        inputs,
        PROVENANCE_ESTIMATOR,
        [sys.argv[0], *argv],
        extra={
            "rungs": [rung.name for rung in rungs],
            "estimators": list(ESTIMATORS),
            "response_columns": {
                "fraction": FRACTION_RESPONSE,
                "count": COUNT_RESPONSE,
            },
            "outcome_stratum": "all",
            "stratum_audit": stratum_audit,
            "max_per_animal": args.max_per_animal,
            "min_events_session_phi": args.min_events_session_phi,
            "bootstrap": {
                "n_draws": args.n_bootstrap,
                "seed": args.seed,
                "interval": "95 %: the median +/- t(0.975, n_animals - 1) "
                "sqrt(n_animals / (n_animals - 1)) SD(draws)",
                "unit": "animal, then session within animal",
            },
            "aic_equalization": (
                "multinomial_log_constant over the count matrix added to both "
                "conditional-Poisson llf values; the fraction estimator's AIC is "
                f"{QUASI_SCALE}"
            ),
            # (session, rung, term) triples the identifiability rule removed
            # from the medians, the bootstrap and the figures. The drop is
            # paired: if one estimator's row is unidentified, all three go.
            "n_unidentified_triples_dropped": n_unpaired_triples(result.coefficients),
            "n_fraction_rows_model_se_fallback": int(
                (result.coefficients["se_source"] == "model_fallback").sum()
            ),
            # (session, rung, estimator) fits whose parent landed on a
            # different number of events, so their AIC was not differenced.
            "n_fits_delta_aic_row_mismatch": n_fits_delta_aic_row_mismatch(
                result.coefficients
            ),
        },
    )
    logger.info("wrote %s and its companions", coefficients_path)
    print(result.summary.to_string(index=False))


if __name__ == "__main__":
    main()
