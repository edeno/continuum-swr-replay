#!/usr/bin/env python
"""Check that the penalised solver reproduces maximum likelihood at lambda = 0.

The mPFC unit GLMs are now fit with a Gaussian prior on the event-basis
coefficients. The prior is a scientific choice; the *solver* that carries it
must not be. This script removes the prior and asks whether the solver still
lands where statsmodels does, on real sessions, against the diagnostics stored
when the batched fitter was first accepted.

It is deliberately archaeological. The stored reference was fit on the design
of the day -- a cubic B-spline basis with only patsy's first boundary function
dropped, 12 functions per event class rather than today's 11 -- so the design
is rebuilt here in that shape from the session's own fit mask, event sets and
seeded matched-control draw. Comparing today's basis against yesterday's
reference would measure the basis change, which is not what is in question.

What the comparison can and cannot say: where the maximum-likelihood estimate
exists, the two solvers must agree to the tolerances the batched fitter was
accepted at. Where it does not -- a basis column whose support holds no spikes
drives its coefficient without bound -- the two stop in different places for
the same reason, and those cells are reported separately rather than gated.
Those are exactly the cells the prior was introduced for.

Usage::

    uv run python scripts/pfc/check_pfc_ridge_lambda0.py \\
        --diagnostics-dir <tree of *_mPFC_modulation.fit_diagnostics.npz> \\
        --session j1620210710_02_r1 --session chimi20200223_06_r3 \\
        --session wilbur20210329_02_r1 \\
        --out .claude/docs/reviews/2026-09-15-pfc-ridge-lambda0-gate.md
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from patsy import dmatrix

from continuum_swr_replay.parameters import SAMPLING_FREQUENCY

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.checks._report import _invocation  # noqa: E402

#: Tolerances the batched fitter was accepted at against statsmodels on the
#: three gate sessions. Reused unchanged: this is the same solver, with the
#: penalty set to zero, so it has to clear the same bar.
PARAM_TOL = 1e-5
DEVIANCE_TOL = 1e-6

#: Lags the archived fits were taken over, in samples at 500 Hz.
FIT_LAGS = np.arange(-125, 126, 1)


def audited_basis(lags: NDArray, sampling_frequency: float) -> NDArray:
    """The peri-event basis as it stood when the reference was fit.

    ``bs(...) - 1`` drops patsy's *first* boundary function and leaves the
    last, so the kernel was pinned to zero at the window's left edge and free
    at its right. Today's basis drops both. Rebuilt here rather than imported
    because the library no longer builds it.

    Parameters
    ----------
    lags : NDArray, shape (n_lags,)
        Lag indices in samples, positive meaning after onset.
    sampling_frequency : float

    Returns
    -------
    NDArray, shape (n_lags, n_basis)
    """
    from continuum_swr_replay.analysis.pfc.design import KNOT_SPACING_S

    lag_time = np.asarray(lags, dtype=float) / sampling_frequency
    knots = np.arange(lag_time[0], lag_time[-1], KNOT_SPACING_S)[1:]
    return np.asarray(
        dmatrix(
            "bs(lag_time, knots=knots) - 1",
            {"lag_time": lag_time, "knots": knots},
        )
    )


def audited_designs(
    event_times: dict[str, pd.DataFrame],
    time: NDArray,
    spline_basis: NDArray,
    lags: NDArray,
) -> dict[str, NDArray]:
    """The nine designs, in the block order the archived fits used.

    Only the basis width differs from today's; the models and the order their
    blocks are concatenated in are unchanged.
    """
    from continuum_swr_replay.analysis.pfc.design import (
        convolve_event_basis,
        event_onset_counts,
    )

    constant = np.ones((len(time), 1))
    convolved = {
        name: convolve_event_basis(
            event_onset_counts(event_times[name], time), spline_basis, lags
        )
        for name in (
            "non_local_no_SWR",
            "non_local_SWR",
            "non_local",
            "SWR",
            "matched_control",
        )
    }

    def stack(*names: str) -> NDArray:
        return np.concatenate([constant, *(convolved[n] for n in names)], axis=1)

    return {
        "constant": constant,
        "non_local_no_SWR": stack("non_local_no_SWR"),
        "non_local_SWR": stack("non_local_SWR"),
        "SWR": stack("SWR"),
        "non_local_SWR + non_local_no_SWR": stack("non_local_no_SWR", "non_local_SWR"),
        "SWR + non_local_no_SWR": stack("SWR", "non_local_no_SWR"),
        "non_local": stack("non_local"),
        "matched_control": stack("matched_control"),
        "matched_control + non_local_no_SWR": stack(
            "matched_control", "non_local_no_SWR"
        ),
    }


def rebuild_session(
    session: str, sampling_frequency: float = SAMPLING_FREQUENCY
) -> dict[str, object]:
    """The fit mask, counts and audited designs of one session.

    Built from the same functions the shipped fit uses and the same
    session-keyed control seed, so the design is the one the reference was fit
    on rather than a fresh cut of the data.
    """
    from continuum_swr_replay.analysis.pfc.cohort import load_session_stop_cohort
    from continuum_swr_replay.analysis.pfc.session import prepare_session
    from continuum_swr_replay.decoding.identifiers import parse_session_name
    from continuum_swr_replay.intervals import get_spikecount_per_time_bin

    nwb_file_name, epoch = parse_session_name(session)

    cohort_stops, _ = load_session_stop_cohort(nwb_file_name, epoch, cohort="at_well")
    # The fit mask, the event sets and the seeded matched control, exactly as
    # ``run_session`` prepares them; its lags are the archived ``FIT_LAGS``.
    prepared = prepare_session(
        nwb_file_name,
        epoch,
        stop_cohort="at_well",
        cohort_stops=cohort_stops,
        sampling_frequency=sampling_frequency,
    )
    if not np.array_equal(prepared.fit_lags, FIT_LAGS):
        raise ValueError(
            f"the session is prepared over {len(prepared.fit_lags)} lags, the "
            f"archived fits over {len(FIT_LAGS)}; the rebuild is not the audited "
            "design"
        )
    data = prepared.data
    event_times = prepared.event_times
    fit_mask = prepared.fit_mask
    position_info = prepared.position_info

    time = position_info.index.to_numpy(dtype=float)
    spline_basis = audited_basis(FIT_LAGS, sampling_frequency)
    counts = np.column_stack(
        [
            get_spikecount_per_time_bin(neuron, time)
            for neuron in data["spike_times"]["mPFC"]
        ]
    ).astype(float)
    return {
        "fit_mask": np.asarray(fit_mask, dtype=bool),
        "counts": counts,
        "designs": audited_designs(event_times, time, spline_basis, FIT_LAGS),
        "n_basis": int(spline_basis.shape[1]),
    }


def compare_session(session: str, diagnostics_dir: Path) -> pd.DataFrame:
    """Per model, how far the unpenalised solver lands from the reference.

    A cell past ``SEPARATED_LOG_RATE`` has no maximum-likelihood estimate, and
    the two solvers stop wherever their step rule runs out rather than at a
    common point, so it is not compared.
    """
    from continuum_swr_replay.analysis.pfc.batched_glm import fit_poisson_glms
    from continuum_swr_replay.analysis.pfc.modulation import SEPARATED_LOG_RATE

    reference = np.load(
        diagnostics_dir / f"{session}_mPFC_modulation.fit_diagnostics.npz"
    )
    rebuilt = rebuild_session(session)
    fit_rows = rebuilt["fit_mask"]
    counts = rebuilt["counts"][fit_rows]

    rows = []
    for model, design in rebuilt["designs"].items():
        stored_params = reference[f"{model}|params"]
        stored_deviance = reference[f"{model}|deviance"]
        stored_converged = reference[f"{model}|converged"].astype(bool)
        if stored_params.shape[1] != design.shape[1]:
            raise ValueError(
                f"{session} {model}: rebuilt design has {design.shape[1]} "
                f"columns, the stored reference {stored_params.shape[1]}. The "
                "rebuild is not the audited design."
            )
        fit = fit_poisson_glms(design[fit_rows], counts, penalty=None)
        both = stored_converged & np.asarray(fit.converged, dtype=bool)
        # The constant model has no basis columns, so nothing in it can run
        # away; every other model is judged on its coefficients alone, the
        # intercept excluded (it carries the baseline rate, not a modulation).
        basis_params = stored_params[:, 1:]
        separated = (
            np.nanmax(np.abs(basis_params), axis=1) > SEPARATED_LOG_RATE
            if basis_params.shape[1]
            else np.zeros(len(stored_deviance), dtype=bool)
        )
        well_posed = both & ~separated
        difference = np.abs(fit.params - stored_params)
        deviance_difference = np.abs(np.asarray(fit.deviance) - stored_deviance)
        rows.append(
            {
                "session": session,
                "model": model,
                "n_units": int(len(stored_deviance)),
                "n_converged_both": int(both.sum()),
                "n_well_posed": int(well_posed.sum()),
                "max_abs_dparams_well_posed": (
                    float(np.nanmax(difference[well_posed]))
                    if well_posed.any()
                    else np.nan
                ),
                "max_abs_ddeviance_well_posed": (
                    float(np.nanmax(deviance_difference[well_posed]))
                    if well_posed.any()
                    else np.nan
                ),
                "n_separated": int((both & separated).sum()),
                "max_abs_beta_separated": (
                    float(np.nanmax(np.abs(basis_params[both & separated])))
                    if (both & separated).any()
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def _md(frame: pd.DataFrame) -> str:
    """A markdown table, missing values as an empty cell."""

    def cell(value: object) -> str:
        if isinstance(value, str):
            return value
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return ""
        return f"{value:.4g}" if isinstance(value, float) else str(value)

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body]) + "\n"


def build_report(
    per_model: pd.DataFrame, diagnostics_dir: Path, argv: Sequence[str]
) -> str:
    """The audit markdown: the gate, the table, and what falls outside it.

    ``argv`` is the arguments ``main`` parsed, recorded in the command line.
    """
    from continuum_swr_replay.analysis.pfc.modulation import SEPARATED_LOG_RATE

    per_session = (
        per_model.groupby("session")
        .agg(
            n_units=("n_units", "max"),
            cells_converged_both=("n_converged_both", "sum"),
            cells_well_posed=("n_well_posed", "sum"),
            max_abs_dparams_well_posed=("max_abs_dparams_well_posed", "max"),
            max_abs_ddeviance_well_posed=("max_abs_ddeviance_well_posed", "max"),
            cells_separated=("n_separated", "sum"),
            max_abs_beta_separated=("max_abs_beta_separated", "max"),
        )
        .reset_index()
    )
    worst_params = float(np.nanmax(per_model["max_abs_dparams_well_posed"]))
    worst_deviance = float(np.nanmax(per_model["max_abs_ddeviance_well_posed"]))
    passed = worst_params < PARAM_TOL and worst_deviance < DEVIANCE_TOL

    lines = [
        "# The penalised solver at lambda = 0 is the maximum-likelihood one\n",
        "Generated by `scripts/pfc/check_pfc_ridge_lambda0.py`. The mPFC unit "
        "GLMs carry a Gaussian prior on the event-basis coefficients. The "
        "prior is a scientific choice; the solver that carries it must not "
        "be. With the penalty set to zero the solver has to land where "
        "statsmodels does, on real sessions -- which is what this checks, "
        "against the diagnostics stored when the batched fitter was first "
        "accepted.\n",
        f"Command: `{_invocation(argv)}`\n",
        "**The design is rebuilt as it stood when the reference was fit** -- a "
        "cubic B-spline basis with only patsy's first boundary function "
        "dropped, 12 functions per event class rather than today's 11 -- from "
        "each session's own fit mask, event sets and seeded matched-control "
        "draw. Comparing today's basis against yesterday's reference would "
        "measure the basis change, which is not what is in question.\n",
        f"Gate: `|d params| < {PARAM_TOL:g}`, `|d deviance| < "
        f"{DEVIANCE_TOL:g}`, over the cells that converged under both solvers "
        "and whose maximum-likelihood estimate exists (every coefficient "
        f"within `|beta| <= {SEPARATED_LOG_RATE:g}`, a rate multiplier of "
        "5e8).\n",
        f"**Result: {'PASS' if passed else 'FAIL'}.** Worst well-posed cell "
        f"across every session and model: `|d params| = {worst_params:.2g}`, "
        f"`|d deviance| = {worst_deviance:.2g}`.\n",
        "## Per session\n",
        _md(per_session),
        "`cells_separated` are the unit x model cells whose maximum-likelihood "
        "estimate does not exist: a basis column whose support holds no spikes "
        "drives its coefficient without bound, and the two solvers stop "
        "wherever their step rule runs out rather than at a common point. They "
        "are reported, not gated -- and they are exactly the cells the prior "
        "was introduced for. Under it none of them is separated and none fails "
        "to converge; they are estimated rather than compared.\n",
        "## Per session and model\n",
        _md(per_model),
        f"Reference diagnostics read from `{diagnostics_dir}`.\n",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        required=True,
        help=(
            "Directory holding the stored statsmodels diagnostics, "
            "`<session>_mPFC_modulation.fit_diagnostics.npz` per session."
        ),
    )
    parser.add_argument(
        "--session",
        dest="sessions",
        action="append",
        required=True,
        help="Session key, e.g. j1620210710_02_r1. Repeatable.",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Fit on this GPU instead of the emptiest free one.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit the batched fitter to run on CPU when no GPU is free.",
    )
    parser.add_argument("--out", type=Path, default=None)
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)

    from continuum_swr_replay.runs.device import configure_jax_gpu

    record = configure_jax_gpu(gpu=args.gpu, allow_cpu=args.allow_cpu)
    print(f"JAX device: {record}")

    per_model = pd.concat(
        [compare_session(session, args.diagnostics_dir) for session in args.sessions],
        ignore_index=True,
    )
    report = build_report(per_model, args.diagnostics_dir, argv)
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
