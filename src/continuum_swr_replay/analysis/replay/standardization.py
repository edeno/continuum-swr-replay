"""Pooled-global standardization of continuous covariates (`<col>_z`).

Standardizing a continuous predictor to unit SD makes a coefficient read as "effect
per 1 SD of predictor" and comparable across predictors. For these Poisson /
conditional-Poisson GLMs the standardized coefficient equals ``raw_coef * SD(x)``
exactly, so the persisted SD reference also documents how to convert a raw fit.

This is the **persisted-table** `_z` (means/SDs over the *full* materialized table),
folded into the owning builders (event_arm side in ``prepare_data``, qvals side in
``qvals_table``) so a clean rebuild carries them. It is deliberately NOT the fit-time
`_z` the is_swr-contrast runners compute over their event/stop cohort — those are a
different scope and must not be repointed here (see the phase-6 plan, F2).

The reference (``{col: {"mean": ..., "std": ...}}``) is persisted next to each table so
the transform is reproducible and auditable; re-applying it reproduces the `_z` exactly.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

Reference = dict[str, dict[str, float]]


def attach_standardized(
    df: pd.DataFrame, cols, reference: Reference | None = None
) -> tuple[pd.DataFrame, Reference]:
    """Append ``<col>_z`` pooled-global standardized columns to ``df`` in place.

    Parameters
    ----------
    df : pd.DataFrame
        Table to standardize (modified in place; also returned).
    cols : iterable of str
        Continuous covariates to standardize. Columns absent from ``df`` are skipped.
    reference : dict, optional
        ``{col: {"mean": m, "std": s}}``. If given, that mean/SD is *applied* (so a
        rebuild reproduces a frozen transform); if ``None``, the mean/SD are computed
        from ``df`` (``np.nanmean`` / ``np.nanstd``) and returned.

    Returns
    -------
    (df, used_reference)
        ``used_reference`` maps every standardized column to the mean/SD actually used.
        A column with ``std == 0`` (or non-finite) yields an all-``NaN`` ``_z`` — the
        coefficient is not identified, matching the original one-off's behavior.
    """
    used: Reference = {}
    for c in cols:
        if c not in df.columns:
            continue
        x = df[c].to_numpy(dtype=float)
        if reference is not None:
            # Apply mode: fail loud if the frozen reference lacks a requested column.
            # Silently recomputing it would mix frozen and freshly-fit standardization
            # (a partial/stale reference file), which is exactly the reproducibility
            # hazard this reference exists to prevent.
            if c not in reference:
                raise KeyError(
                    f"standardization reference is missing column {c!r}; a partial or "
                    f"stale reference cannot be applied (have: {sorted(reference)})"
                )
            mu = float(reference[c]["mean"])
            sd = float(reference[c]["std"])
        else:
            mu = float(np.nanmean(x))
            sd = float(np.nanstd(x))
        df[c + "_z"] = (x - mu) / sd if sd > 0 else np.nan
        used[c] = {"mean": mu, "std": sd}
    return df, used


def assert_standardized_present(df: pd.DataFrame, cols) -> None:
    """Rebuild-completeness guard: raise if any ``<col>_z`` was dropped or wrongly wiped.

    Checked **per column** (the exact regression phase 6 fixes — a silently dropped or
    all-NaN ``_z``): for every source covariate present in ``df``, its ``<col>_z`` must
    exist, and it must carry real data **unless the source itself has no variance**
    (a genuinely zero-variance or all-NaN source legitimately yields an all-NaN ``_z``).
    An all-NaN ``_z`` whose source *does* vary is a failure.
    """
    present_sources = [c for c in cols if c in df.columns]
    missing = [f"{c}_z" for c in present_sources if f"{c}_z" not in df.columns]
    if missing:
        raise RuntimeError(f"standardized columns missing after build: {missing}")
    wiped = []
    for c in present_sources:
        zc = f"{c}_z"
        source_varies = float(np.nanstd(df[c].to_numpy(dtype=float))) > 0
        if source_varies and not df[zc].notna().any():
            wiped.append(zc)
    if wiped:
        raise RuntimeError(
            f"standardized columns are all-NaN despite a varying source: {wiped}"
        )


def save_reference(reference: Reference, path) -> None:
    """Write a standardization reference to JSON (sorted, stable)."""
    with open(path, "w") as f:
        json.dump(reference, f, indent=2, sort_keys=True)


def load_reference(path) -> Reference:
    """Read a standardization reference written by :func:`save_reference`."""
    with open(path) as f:
        return json.load(f)
