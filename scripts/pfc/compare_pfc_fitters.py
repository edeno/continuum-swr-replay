#!/usr/bin/env python
"""Write the equivalence audit of the batched JAX fitter against statsmodels.

**Retired-generation bundles only.** This audit compares two runs of the
unpenalized maximum-likelihood ladder, whose statistics tables carry the
``: lr`` / ``: pval`` / ``: aic`` columns and ``min_aic_model``. Current
bundles are ridge fits scored out of fold (sidecar ``estimator =
"ridge_poisson_glm"``): ``run_pfc_analysis_bandit.py`` drops those columns
(``drop_retired_statistics``) and no longer offers ``--fitter statsmodels``,
which has no penalized path, so there is no statsmodels-side bundle of the
current estimator to compare. Renaming the columns to ``: p_heldout`` /
``min_heldout_model`` would not make this audit apply; ``main`` refuses a
ridge bundle. The batched fitter's equivalence to statsmodels on the
unpenalized fit is pinned by ``tests/analysis/pfc/test_batched_glm.py`` and
``test_glm_pipeline.py::TestFitterEquivalence``.

The mPFC unit GLMs are now fit by a batched IRLS in JAX rather than one
statsmodels fit per neuron. The two are meant to be the same estimator -- same
log link, same starting mean ``(y + mean(y)) / 2``, same IRLS step, same
``|deviance change| < 1e-8`` stopping rule -- so the only question a reader
needs answered is how close they come on real sessions, and this script answers
it from the artifacts of two runs of the same sessions:

- every numeric column of the shipped per-neuron statistics table (the
  likelihood-ratio statistics and p-values, the fractional deviances, the peak
  modulations, the theta statistics and the firing rates);
- the five modulation time courses, which are the fitted coefficients read
  through the spline basis;
- the categorical outputs that must be *identical*, not merely close: the
  best-AIC model in each decomposition, and the NaN pattern that says which
  comparisons were unavailable;
- where the runs wrote fit diagnostics (``--fit-diagnostics``), the parameters,
  deviance, log-likelihood, AIC, rank and convergence flags per neuron and
  model -- the fits themselves, before anything is derived from them.

Usage::

    uv run python scripts/pfc/compare_pfc_fitters.py \\
        --statsmodels-dir /scratch/fitter/statsmodels \\
        --jax-dir /scratch/fitter/jax \\
        --time-file statsmodels:wilbur20210329_02_r1_mPFC:/scratch/.../sm.time \\
        --time-file jax:wilbur20210329_02_r1_mPFC:/scratch/.../jax.time \\
        --out .claude/docs/reviews/2026-09-14-pfc-jax-fitter-equivalence.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.pfc import bundle_files
from continuum_swr_replay.runs.fit_provenance import fit_provenance_path

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.checks._report import _invocation, _md  # noqa: E402

#: Gate tolerances. Parameters and anything derived from them linearly (the
#: modulation curves, the peak modulations, the theta statistics) are compared
#: at ``PARAM_TOL``; deviance-scale quantities (the deviance itself, the LR
#: statistics, the fractional deviances) at ``DEVIANCE_TOL``; p-values at
#: ``PVAL_TOL``. ``AIC_TOL`` is twice ``DEVIANCE_TOL`` because
#: ``aic = -2 llf + 2 rank``.
PARAM_TOL = 1e-5
DEVIANCE_TOL = 1e-6
PVAL_TOL = 1e-6
AIC_TOL = 2 * DEVIANCE_TOL

#: The modulation arrays each bundle carries, by file suffix.
MODULATION_ARRAYS = (
    "SWR_modulation",
    "no_SWR_modulation",
    "non_local_SWR_modulation",
    "non_local_no_SWR_modulation",
    "matched_control_modulation",
)

#: Columns that must match exactly rather than to a tolerance.
EXACT_COLUMNS = (
    "min_aic_model",
    "min_aic_model_replay_split",
    "nwb_file_name",
    "epoch_name",
    "brain_area",
    "stop_cohort",
)


def _column_tolerance(column: str) -> float:
    """Which gate tolerance a statistics column is read against."""
    if column.endswith(": pval"):
        return PVAL_TOL
    if column.endswith((": lr", ": frac_null_dev", ": frac_explained_dev")):
        return DEVIANCE_TOL
    return PARAM_TOL


def bundle_stems(directory: Path, suffix: str = "") -> set[str]:
    """Epoch identifiers of the modulation bundles in ``directory``."""
    return set(bundle_files.bundle_identifiers(directory, suffix))


def load_bundle(directory: Path, stem: str, suffix: str = "") -> dict[str, Any]:
    """One session's shipped artifacts, plus its sidecar and any diagnostics.

    Parameters
    ----------
    directory : Path
        Directory holding the bundle.
    stem : str
        Epoch identifier, e.g. ``"wilbur20210329_02_r1_mPFC"``.
    suffix : str
        Cohort suffix on the artifact names (``""`` is the canonical cohort).

    Returns
    -------
    dict
        ``stats`` (DataFrame), ``modulations`` (dict of array), ``sidecar``
        (dict) and ``diagnostics`` (dict of arrays, or ``None``).
    """
    artifact = bundle_files.bundle_path(directory, stem, suffix)
    base = artifact.with_suffix("")
    sidecar_path = fit_provenance_path(artifact)
    diagnostics_path = base.with_suffix(".fit_diagnostics.npz")
    return {
        "stats": pd.read_pickle(artifact),
        "modulations": {
            name: np.load(base.with_suffix(f".{name}.npy"))
            for name in MODULATION_ARRAYS
        },
        "sidecar": (
            json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
        ),
        "diagnostics": (
            dict(np.load(diagnostics_path, allow_pickle=False))
            if diagnostics_path.exists()
            else None
        ),
    }


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    """Largest absolute difference over the entries finite in both."""
    both_finite = np.isfinite(left) & np.isfinite(right)
    if not both_finite.any():
        return 0.0
    return float(np.max(np.abs(left[both_finite] - right[both_finite])))


def compare_statistics(
    statsmodels_stats: pd.DataFrame, jax_stats: pd.DataFrame
) -> tuple[list[dict[str, Any]], list[str]]:
    """Column-by-column comparison of the shipped per-neuron statistics.

    Returns
    -------
    rows : list of dict
        One row per compared column: its tolerance, the largest difference, and
        whether the NaN pattern agrees.
    failures : list of str
        Human-readable reasons the gate failed, empty when it passed.
    """
    rows: list[dict[str, Any]] = []
    failures: list[str] = []

    if list(statsmodels_stats.columns) != list(jax_stats.columns):
        only_sm = sorted(set(statsmodels_stats.columns) - set(jax_stats.columns))
        only_jax = sorted(set(jax_stats.columns) - set(statsmodels_stats.columns))
        failures.append(
            f"the two runs' statistics tables have different columns "
            f"(statsmodels only: {only_sm}; jax only: {only_jax})"
        )
    if len(statsmodels_stats) != len(jax_stats):
        failures.append(
            f"different neuron counts: {len(statsmodels_stats)} vs {len(jax_stats)}"
        )
        return rows, failures

    for column in statsmodels_stats.columns:
        if column not in jax_stats.columns:
            continue
        left = statsmodels_stats[column]
        right = jax_stats[column]
        if column in EXACT_COLUMNS or left.dtype == object:
            mismatches = int((left.fillna("<na>") != right.fillna("<na>")).sum())
            rows.append(
                {
                    "column": column,
                    "tolerance": "identical",
                    "max_abs_diff": np.nan,
                    "n_mismatched": mismatches,
                    "nan_pattern_matches": True,
                }
            )
            if mismatches:
                failures.append(f"{column}: {mismatches} neuron(s) differ")
            continue

        left_values = left.to_numpy(dtype=float)
        right_values = right.to_numpy(dtype=float)
        # ``isfinite``, not ``isnan``: a value that is NaN on one side and
        # +/-inf on the other has a matching NaN pattern but is not a matching
        # result, and the difference over the finite entries cannot see it.
        nan_matches = bool(
            np.array_equal(np.isfinite(left_values), np.isfinite(right_values))
        )
        difference = _max_abs_difference(left_values, right_values)
        tolerance = _column_tolerance(column)
        rows.append(
            {
                "column": column,
                "tolerance": tolerance,
                "max_abs_diff": difference,
                "n_mismatched": int(
                    np.sum(
                        np.abs(
                            np.where(
                                np.isfinite(left_values) & np.isfinite(right_values),
                                left_values - right_values,
                                0.0,
                            )
                        )
                        > tolerance
                    )
                ),
                "nan_pattern_matches": nan_matches,
            }
        )
        if not nan_matches:
            failures.append(
                f"{column}: finite/non-finite pattern differs between fitters"
            )
        if difference > tolerance:
            failures.append(
                f"{column}: max|diff| {difference:.3e} exceeds {tolerance:.0e}"
            )
    return rows, failures


def compare_modulations(
    statsmodels_modulations: Mapping[str, np.ndarray],
    jax_modulations: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Compare the five modulation time courses, the fitted curves themselves."""
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for name in MODULATION_ARRAYS:
        left = statsmodels_modulations[name]
        right = jax_modulations[name]
        if left.shape != right.shape:
            failures.append(f"{name}: shapes differ {left.shape} vs {right.shape}")
            continue
        nan_matches = bool(np.array_equal(np.isfinite(left), np.isfinite(right)))
        difference = _max_abs_difference(left, right)
        rows.append(
            {
                "array": name,
                "shape": str(left.shape),
                "max_abs_diff": difference,
                "nan_pattern_matches": nan_matches,
            }
        )
        if not nan_matches:
            failures.append(
                f"{name}: finite/non-finite pattern differs between fitters"
            )
        if difference > PARAM_TOL:
            failures.append(
                f"{name}: max|diff| {difference:.3e} exceeds {PARAM_TOL:.0e}"
            )
    return rows, failures


def compare_diagnostics(
    statsmodels_diagnostics: Mapping[str, np.ndarray],
    jax_diagnostics: Mapping[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Compare the fits themselves, per neuron and model.

    The parameters of a rank-deficient design are reported but not gated: both
    fitters return a minimum-norm solution there, but statsmodels reaches it
    through a pseudo-inverse of the weighted design with a singular-value
    cutoff of ``1e-15``, while the batched fitter cuts eigenvalues of the
    information matrix at ``1e-12``, and a direction that sits between those
    two cutoffs is kept by one and dropped by the other. Nothing downstream
    reads those parameters -- ``process_neuron`` routes every statistic of a
    rank-deficient model to NaN -- so the gate is on the estimable models.
    """
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    models = sorted(
        {key.split("|", 1)[0] for key in statsmodels_diagnostics if "|" in key}
    )

    def side(source: Mapping[str, np.ndarray], model: str, quantity: str) -> np.ndarray:
        return np.asarray(source[f"{model}|{quantity}"])

    for model in models:
        params_difference = _max_abs_difference(
            side(statsmodels_diagnostics, model, "params"),
            side(jax_diagnostics, model, "params"),
        )
        deviance_difference = _max_abs_difference(
            side(statsmodels_diagnostics, model, "deviance"),
            side(jax_diagnostics, model, "deviance"),
        )
        llf_difference = _max_abs_difference(
            side(statsmodels_diagnostics, model, "llf"),
            side(jax_diagnostics, model, "llf"),
        )
        aic_difference = _max_abs_difference(
            side(statsmodels_diagnostics, model, "aic"),
            side(jax_diagnostics, model, "aic"),
        )
        statsmodels_rank = side(statsmodels_diagnostics, model, "df_model") + 1
        jax_rank = side(jax_diagnostics, model, "df_model") + 1
        # A neuron with no fit at all -- statsmodels refuses a response with no
        # spikes, and both runs then take the firing-rates-only fallback --
        # has no rank to compare. The two sides must agree on *which* neurons
        # those are, and on the rank of every neuron that does have a fit.
        statsmodels_fitted = np.isfinite(statsmodels_rank) & ~np.all(
            np.isnan(side(statsmodels_diagnostics, model, "params")), axis=1
        )
        jax_fitted = np.isfinite(jax_rank) & ~np.all(
            np.isnan(side(jax_diagnostics, model, "params")), axis=1
        )
        fitted = statsmodels_fitted & jax_fitted
        no_fit_matches = bool(np.array_equal(statsmodels_fitted, jax_fitted))
        rank_matches = no_fit_matches and bool(
            np.array_equal(statsmodels_rank[fitted], jax_rank[fitted])
        )
        n_features = side(statsmodels_diagnostics, model, "params").shape[1]
        estimable = bool(np.all(statsmodels_rank[fitted] == n_features))
        converged_matches = bool(
            np.array_equal(
                side(statsmodels_diagnostics, model, "converged"),
                side(jax_diagnostics, model, "converged"),
            )
        )
        iteration_difference = int(
            np.max(
                np.abs(
                    side(statsmodels_diagnostics, model, "n_iter")
                    - side(jax_diagnostics, model, "n_iter")
                )
            )
        )
        rows.append(
            {
                "model": model,
                "n_without_a_fit": int((~fitted).sum()),
                "estimable": estimable,
                "max_abs_dparams": params_difference,
                "max_abs_ddeviance": deviance_difference,
                "max_abs_dllf": llf_difference,
                "max_abs_daic": aic_difference,
                "rank_matches": rank_matches,
                "converged_matches": converged_matches,
                "max_diter": iteration_difference,
            }
        )
        if not no_fit_matches:
            failures.append(
                f"{model}: the two runs disagree on which neurons have no fit"
            )
        elif not rank_matches:
            failures.append(f"{model}: the two fitters disagree on the design's rank")
        if not converged_matches:
            failures.append(f"{model}: convergence flags differ")
        if deviance_difference > DEVIANCE_TOL:
            failures.append(
                f"{model}: max|d deviance| {deviance_difference:.3e} exceeds "
                f"{DEVIANCE_TOL:.0e}"
            )
        if llf_difference > DEVIANCE_TOL:
            failures.append(
                f"{model}: max|d llf| {llf_difference:.3e} exceeds {DEVIANCE_TOL:.0e}"
            )
        if aic_difference > AIC_TOL:
            failures.append(
                f"{model}: max|d aic| {aic_difference:.3e} exceeds {AIC_TOL:.0e}"
            )
        if estimable and params_difference > PARAM_TOL:
            failures.append(
                f"{model}: max|d params| {params_difference:.3e} exceeds "
                f"{PARAM_TOL:.0e}"
            )
    return rows, failures


# The label itself carries colons ("(h:mm:ss or m:ss)"), so the value is the
# text after the last colon on the line, not the first.
_WALL_CLOCK = re.compile(r"Elapsed \(wall clock\) time[^)]*\):\s*(\S+)")
_MAX_RSS = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def parse_time_file(path: Path) -> dict[str, Any]:
    """Wall clock and peak memory from a ``/usr/bin/time -v`` log.

    Parameters
    ----------
    path : Path

    Returns
    -------
    dict
        ``wall_clock`` (formatted as written), ``wall_clock_s`` (float) and
        ``max_rss_gb`` (float).
    """
    text = path.read_text(errors="replace")
    wall_match = _WALL_CLOCK.search(text)
    rss_match = _MAX_RSS.search(text)
    if wall_match is None:
        raise ValueError(f"{path} has no '/usr/bin/time -v' wall-clock line")
    formatted = wall_match.group(1).strip()
    parts = [float(part) for part in formatted.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return {
        "wall_clock": formatted,
        "wall_clock_s": seconds,
        "max_rss_gb": (
            float(rss_match.group(1)) / (1024**2) if rss_match else float("nan")
        ),
    }


def _timing_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """One row per session: each fitter's wall clock, peak memory, and the ratio.

    Parameters
    ----------
    rows : list of dict
        ``fitter``, ``session`` and the fields :func:`parse_time_file` returns.

    Returns
    -------
    pd.DataFrame
    """
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    table = pd.DataFrame({"session": sorted(frame["session"].unique())})
    for fitter in ("statsmodels", "jax"):
        side = frame[frame["fitter"] == fitter].set_index("session")
        table[f"{fitter}_wall_clock"] = table["session"].map(side["wall_clock"])
        table[f"{fitter}_max_rss_gb"] = table["session"].map(side["max_rss_gb"])
        table[f"{fitter}_seconds"] = table["session"].map(side["wall_clock_s"])
    if {"statsmodels_seconds", "jax_seconds"} <= set(table.columns):
        table["speedup"] = table["statsmodels_seconds"] / table["jax_seconds"]
    return table.drop(columns=["statsmodels_seconds", "jax_seconds"])


#: How the audit's tables print a float, and an empty one.
FLOAT_FORMAT = "{:.3e}"
EMPTY_TABLE = "_(nothing to report)_"


def build_report(
    sessions: list[str],
    comparisons: dict[str, dict[str, Any]],
    timings: pd.DataFrame,
    statsmodels_dir: Path,
    jax_dir: Path,
    command: str,
    notes: Iterable[str] = (),
) -> str:
    """Assemble the audit markdown."""
    lines: list[str] = []
    add = lines.append
    failures = {
        session: comparison["failures"]
        for session, comparison in comparisons.items()
        if comparison["failures"]
    }

    add("# The batched JAX fitter against statsmodels, on real sessions\n")
    add(
        "Generated by `scripts/pfc/compare_pfc_fitters.py`. The mPFC unit GLMs "
        "were fit one neuron at a time with statsmodels' IRLS; they are now "
        "fit for every neuron of a session at once by a batched IRLS in JAX "
        "(`analysis.pfc.batched_glm`). The two are the same estimator by "
        "construction -- Poisson with a log link, starting mean "
        "`(y + mean(y)) / 2`, the IRLS/Newton step, and statsmodels' own "
        "stopping rule `|deviance change| < 1e-8` applied per neuron -- so "
        "this is a measurement of how closely that construction holds on real "
        "designs, not a calibration.\n"
    )
    if command:
        add(f"Command: `{command}`\n")
    for note in notes:
        add(f"{note}\n")

    add(
        f"**Verdict: {'FAIL' if failures else 'PASS'}** "
        f"({len(sessions)} session(s) compared; "
        f"{len(failures)} with a difference over tolerance).\n"
    )

    add("## Gate\n")
    add(
        f"| quantity | tolerance |\n| --- | --- |\n"
        f"| parameters, modulation curves, peak modulations, theta statistics, "
        f"firing rates | {PARAM_TOL:.0e} absolute |\n"
        f"| deviance, log-likelihood, LR statistics, fractional deviances | "
        f"{DEVIANCE_TOL:.0e} absolute |\n"
        f"| LR p-values | {PVAL_TOL:.0e} absolute |\n"
        f"| AIC | {AIC_TOL:.0e} absolute (`aic = -2 llf + 2 rank`) |\n"
        f"| best-AIC model, finite/non-finite pattern, convergence flags, "
        f"design rank | identical |\n"
    )
    add(
        "The parameter gate is read on the estimable models only. A "
        "rank-deficient design has no unique minimum-norm solution the two "
        "fitters must agree on -- statsmodels takes a pseudo-inverse of the "
        "weighted design with a singular-value cutoff of `1e-15` while the "
        "batched fitter cuts eigenvalues of the information matrix at "
        "`1e-12`, and a direction between those is kept by one and dropped by "
        "the other -- and `process_neuron` routes every shipped statistic of "
        "such a model to NaN, so nothing downstream reads those parameters. "
        "The design rank itself is still required to be identical, which is "
        "what says the two fitters agree on *which* models those are.\n"
    )
    add(f"- statsmodels run: `{statsmodels_dir}`\n- JAX run: `{jax_dir}`\n")

    add("## Wall clock\n")
    if timings.empty:
        add("_(no `/usr/bin/time -v` logs were passed)_\n")
    else:
        # Memory in GB and a ratio read better at four significant digits than
        # in the scientific notation the difference tables want.
        add(_md(timings, "{:.4g}", EMPTY_TABLE))

    add("## Per session\n")
    for session in sessions:
        comparison = comparisons[session]
        sidecar = comparison["sidecar"]
        add(f"### {session}\n")
        handed_back = sidecar.get("n_neurons_reference_fit")
        add(
            f"- {comparison['n_neurons']} neurons; JAX backend "
            f"`{sidecar.get('jax_backend')}`, jax `{sidecar.get('jax_version')}`, "
            f"tolerance `{sidecar.get('fitter_tol')}`, cohort "
            f"`{sidecar.get('stop_cohort')}`\n"
        )
        if handed_back is not None:
            add(
                f"- {handed_back} of {comparison['n_neurons']} neurons were "
                "handed back to statsmodels by the batched fitter (a fit that "
                "did not converge, or fitted means spread past what the normal "
                "equations carry); their numbers below are statsmodels' own on "
                "both sides, and the remainder are the batched fitter's.\n"
            )
        if comparison["failures"]:
            add("**Over tolerance:**\n")
            for failure in comparison["failures"]:
                add(f"- {failure}\n")
        else:
            add("Every compared quantity is inside the gate.\n")

        if comparison["diagnostics"]:
            add("Fits, per model (over all neurons):\n")
            add(_md(pd.DataFrame(comparison["diagnostics"]), FLOAT_FORMAT, EMPTY_TABLE))
        else:
            add(
                "_No fit diagnostics in this pair of runs; pass "
                "`--fit-diagnostics` to both to compare the fits directly._\n"
            )

        add("Modulation time courses:\n")
        add(_md(pd.DataFrame(comparison["modulations"]), FLOAT_FORMAT, EMPTY_TABLE))

        statistics = pd.DataFrame(comparison["statistics"])
        worst = statistics.dropna(subset=["max_abs_diff"]).nlargest(12, "max_abs_diff")
        add(
            f"Statistics columns, the 12 with the largest difference of "
            f"{len(statistics)} compared:\n"
        )
        add(_md(worst, FLOAT_FORMAT, EMPTY_TABLE))
        exact = statistics[statistics["tolerance"] == "identical"]
        add(
            "Columns required to be identical: "
            f"{len(exact)} compared, {int(exact['n_mismatched'].sum())} "
            "mismatching neuron(s).\n"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--statsmodels-dir",
        type=Path,
        required=True,
        help="Directory of bundles fit with --fitter statsmodels.",
    )
    parser.add_argument(
        "--jax-dir",
        type=Path,
        required=True,
        help="Directory of bundles fit with --fitter jax.",
    )
    parser.add_argument(
        "--artifact-suffix",
        type=str,
        default="",
        help="Cohort suffix on the bundle names ('' = canonical at-well).",
    )
    parser.add_argument(
        "--time-file",
        action="append",
        default=[],
        metavar="FITTER:SESSION:PATH",
        help=(
            "A '/usr/bin/time -v' log for one run, e.g. "
            "jax:wilbur20210329_02_r1_mPFC:/scratch/jax.time. Repeatable."
        ),
    )
    parser.add_argument(
        "--note",
        action="append",
        default=[],
        help="Extra paragraph to place under the Command line. Repeatable.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write the report here instead of stdout.",
    )
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)

    statsmodels_sessions = bundle_stems(args.statsmodels_dir, args.artifact_suffix)
    jax_sessions = bundle_stems(args.jax_dir, args.artifact_suffix)
    sessions = sorted(statsmodels_sessions & jax_sessions)
    if not sessions:
        raise SystemExit(
            f"No session is present in both {args.statsmodels_dir} and {args.jax_dir}"
        )

    timing_rows = []
    for specification in args.time_file:
        fitter, session, path = specification.split(":", 2)
        timing_rows.append(
            {"fitter": fitter, "session": session, **parse_time_file(Path(path))}
        )
    timings = _timing_table(timing_rows)

    comparisons: dict[str, dict[str, Any]] = {}
    for session in sessions:
        statsmodels_bundle = load_bundle(
            args.statsmodels_dir, session, args.artifact_suffix
        )
        jax_bundle = load_bundle(args.jax_dir, session, args.artifact_suffix)
        for side, bundle in (
            ("statsmodels", statsmodels_bundle),
            ("jax", jax_bundle),
        ):
            if bundle["sidecar"].get("estimator") == "ridge_poisson_glm":
                raise SystemExit(
                    f"{session} on the {side} side is a current ridge bundle "
                    "(estimator='ridge_poisson_glm'). This audit compares the "
                    "retired unpenalized ladder's bundles only; see the module "
                    "docstring."
                )

        statistics, failures = compare_statistics(
            statsmodels_bundle["stats"], jax_bundle["stats"]
        )
        modulations, modulation_failures = compare_modulations(
            statsmodels_bundle["modulations"], jax_bundle["modulations"]
        )
        failures = failures + modulation_failures

        diagnostics: list[dict[str, Any]] = []
        if (
            statsmodels_bundle["diagnostics"] is not None
            and jax_bundle["diagnostics"] is not None
        ):
            diagnostics, diagnostic_failures = compare_diagnostics(
                statsmodels_bundle["diagnostics"], jax_bundle["diagnostics"]
            )
            failures = failures + diagnostic_failures

        sidecar = jax_bundle["sidecar"]
        if sidecar.get("fitter") not in (None, "jax"):
            failures.append(
                f"the JAX-side bundle records fitter={sidecar.get('fitter')!r}"
            )
        statsmodels_fitter = statsmodels_bundle["sidecar"].get("fitter")
        if statsmodels_fitter not in (None, "statsmodels"):
            failures.append(
                f"the statsmodels-side bundle records fitter={statsmodels_fitter!r}"
            )

        comparisons[session] = {
            "statistics": statistics,
            "modulations": modulations,
            "diagnostics": diagnostics,
            "failures": failures,
            "sidecar": sidecar,
            "n_neurons": len(jax_bundle["stats"]),
        }

    report = build_report(
        sessions=sessions,
        comparisons=comparisons,
        timings=timings,
        statsmodels_dir=args.statsmodels_dir,
        jax_dir=args.jax_dir,
        command=_invocation(argv),
        notes=args.note,
    )
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"Wrote {args.out}")

    failed = [session for session in sessions if comparisons[session]["failures"]]
    if failed:
        print(f"FAIL: {len(failed)} session(s) over tolerance: {failed}")
        return 1
    print(f"PASS: {len(sessions)} session(s) inside the gate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
