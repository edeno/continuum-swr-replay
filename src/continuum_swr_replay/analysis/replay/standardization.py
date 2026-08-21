"""Pooled-global standardization of continuous covariates (`<col>_z`).

Standardizing a continuous predictor to unit SD makes a coefficient read as "effect
per 1 SD of predictor" and comparable across predictors. For these Poisson /
conditional-Poisson GLMs the standardized coefficient equals ``raw_coef * SD(x)``
exactly, so the persisted SD reference also documents how to convert a raw fit.

This is the **persisted-table** `_z` (means/SDs over a documented scientific
support in the full materialized table),
folded into the owning builders (event_arm side in ``prepare_data``, qvals side in
``qvals_table``) so a clean rebuild carries them. It is deliberately NOT the fit-time
`_z` the is_swr-contrast runners compute over their event/stop cohort — those are a
different scope and must not be repointed here.

The reference (``{col: {"mean": ..., "std": ..., "support": ...,
"n_reference_rows": ...}}``) is persisted next to each table so both the transform
and its scientific support are auditable; re-applying it reproduces the `_z` exactly.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

Reference = dict[str, dict[str, Any]]


def attach_standardized(
    df: pd.DataFrame,
    cols,
    reference: Reference | None = None,
    *,
    support_mask: Sequence[bool] | np.ndarray | pd.Series | None = None,
    support_label: str = "all_rows",
) -> tuple[pd.DataFrame, Reference]:
    """Append ``<col>_z`` pooled-global standardized columns to ``df`` in place.

    Parameters
    ----------
    df : pd.DataFrame
        Table to standardize (modified in place; also returned).
    cols : iterable of str
        Continuous covariates to standardize. Columns absent from ``df`` are skipped.
    reference : dict, optional
        ``{col: {"mean": m, "std": s, "support": label,
        "n_reference_rows": n}}``. If given, that reference is *applied* (so a
        rebuild reproduces a frozen transform); legacy references without the
        support metadata fail closed. If ``None``, the moments are computed from
        ``df`` (``np.nanmean`` / ``np.nanstd``) and returned with their support.
    support_mask : array-like of bool, optional
        Scientific support used to estimate the reference moments. The resulting
        transform is still applied to every row. For event-arm covariates this is
        the six leaf arms: structural-zero trunk rows do not identify reward/value
        slopes and therefore must not define the reported one-SD unit.
    support_label : str
        Persisted human-readable name for ``support_mask``.

    Returns
    -------
    (df, used_reference)
        ``used_reference`` maps every standardized column to the mean/SD actually used.
        A column with ``std == 0`` (or non-finite) yields an all-``NaN`` ``_z`` — the
        coefficient is not identified, matching the original one-off's behavior.
    """
    if support_mask is None:
        support = np.ones(len(df), dtype=bool)
    else:
        raw_support = np.asarray(support_mask)
        if raw_support.ndim != 1 or len(raw_support) != len(df):
            raise ValueError(
                "support_mask must be one-dimensional and match the table row count"
            )
        if pd.isna(raw_support).any():
            raise ValueError("support_mask must not contain missing values")
        support = raw_support.astype(bool)
    if not support.any():
        raise ValueError("standardization support must contain at least one row")

    used: Reference = {}
    for c in cols:
        if c not in df.columns:
            continue
        x = df[c].to_numpy(dtype=float)
        if np.isinf(x).any():
            raise ValueError(
                f"standardization source {c!r} contains non-finite infinity; "
                "use NaN only for documented missingness"
            )
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
            if not np.isfinite(mu) or not np.isfinite(sd):
                raise ValueError(
                    f"non-finite standardization reference for {c!r}: "
                    f"mean={mu}, std={sd}"
                )
            if sd < 0:
                raise ValueError(
                    f"standardization reference std for {c!r} must be nonnegative"
                )
            missing_metadata = {
                "support",
                "n_reference_rows",
            } - set(reference[c])
            if missing_metadata:
                raise ValueError(
                    f"standardization reference for {c!r} is missing required "
                    f"metadata {sorted(missing_metadata)}; regenerate it rather "
                    "than relabeling legacy moments"
                )
            label = str(reference[c]["support"])
            if not label:
                raise ValueError(f"standardization reference support for {c!r} is empty")
            if support_label != "all_rows" and label != support_label:
                raise ValueError(
                    f"standardization reference support for {c!r} is {label!r}, "
                    f"expected {support_label!r}"
                )
            raw_n_reference_rows = reference[c]["n_reference_rows"]
            try:
                n_reference_rows = int(raw_n_reference_rows)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"standardization reference row count for {c!r} must be a "
                    "positive integer"
                ) from exc
            if (
                isinstance(raw_n_reference_rows, (bool, np.bool_))
                or n_reference_rows <= 0
                or float(raw_n_reference_rows) != n_reference_rows
            ):
                raise ValueError(
                    f"standardization reference row count for {c!r} must be a "
                    "positive integer"
                )
        else:
            reference_values = x[support]
            finite_reference = np.isfinite(reference_values)
            if not finite_reference.any():
                raise ValueError(
                    f"standardization support for {c!r} contains no finite values"
                )
            mu = float(np.nanmean(reference_values))
            sd = float(np.nanstd(reference_values))
            label = support_label
            n_reference_rows = int(finite_reference.sum())
        df[c + "_z"] = (x - mu) / sd if sd > 0 else np.nan
        used[c] = {
            "mean": mu,
            "std": sd,
            "support": label,
            "n_reference_rows": n_reference_rows,
        }
    return df, used


def assert_standardized_present(
    df: pd.DataFrame,
    cols,
    *,
    support_mask: Sequence[bool] | np.ndarray | pd.Series | None = None,
) -> None:
    """Rebuild-completeness guard: raise if any ``<col>_z`` was dropped or wrongly wiped.

    Checked **per column** (catching a silently dropped or wiped ``_z``): for every source covariate present in ``df``, its ``<col>_z`` must
    exist, and it must carry real data **unless the source is a finite constant**
    (a genuine zero-variance source legitimately yields an all-NaN ``_z``). Two failures:
    an all-NaN ``_z`` whose source *does* vary, and a source that is *entirely NaN* (a
    failed upstream join — a canonical standardized covariate must carry data, so this is
    NOT waved through as if it were a legitimate zero-variance column).
    """
    if support_mask is None:
        support = np.ones(len(df), dtype=bool)
    else:
        support = np.asarray(support_mask)
        if support.ndim != 1 or len(support) != len(df) or pd.isna(support).any():
            raise ValueError(
                "support_mask must be one-dimensional, nonmissing, and match rows"
            )
        support = support.astype(bool)

    present_sources = [c for c in cols if c in df.columns]
    missing = [f"{c}_z" for c in present_sources if f"{c}_z" not in df.columns]
    if missing:
        raise RuntimeError(f"standardized columns missing after build: {missing}")
    bad_source = []  # source entirely NaN -> a failed upstream join, not zero-variance
    wiped = []       # _z all-NaN while the source has real variance
    for c in present_sources:
        zc = f"{c}_z"
        col = df[c].to_numpy(dtype=float)
        if np.isinf(col).any():
            raise RuntimeError(
                f"standardization source column contains infinity: {c}"
            )
        z = df[zc].to_numpy(dtype=float)
        if np.isinf(z).any():
            raise RuntimeError(f"standardized column contains infinity: {zc}")
        reference_values = col[support]
        if not np.isfinite(reference_values).any():
            bad_source.append(c)
            continue
        if float(np.nanstd(reference_values)) > 0 and not df[zc].notna().any():
            wiped.append(zc)
    if bad_source:
        raise RuntimeError(
            f"standardization source columns are entirely NaN (failed upstream join?): "
            f"{bad_source} -- a canonical standardized covariate must carry data"
        )
    if wiped:
        raise RuntimeError(
            f"standardized columns are all-NaN despite a varying source: {wiped}"
        )


def assert_standardized_matches_reference(
    df: pd.DataFrame,
    cols: Sequence[str],
    reference: Reference,
    *,
    support_label: str,
    rtol: float = 1e-12,
    atol: float = 1e-12,
) -> None:
    """Require persisted ``<col>_z`` values to match one frozen reference.

    This is the model-runner boundary for materialized tables.  Builder tests alone
    cannot protect a paper fit from reading an older table/reference pair whose
    columns have the right names but whose scientific support differs (for example,
    nine-arm rather than leaf-arm uncertainty moments).  Apply the reference to the
    raw columns, then compare every stored standardized value including missingness.

    Parameters
    ----------
    df : pandas.DataFrame
        Materialized table carrying both each raw source and its ``<col>_z``.
    cols : sequence of str
        Raw source columns required by the requested model predictors.
    reference : dict
        Frozen reference loaded from the owning builder's JSON sidecar.
    support_label : str
        Exact scientific support required by this consumer.
    rtol, atol : float
        Tight numerical tolerances for JSON round-trip differences only.
    """
    requested = tuple(cols)
    missing = [
        column
        for source in requested
        for column in (source, f"{source}_z")
        if column not in df
    ]
    if missing:
        raise ValueError(
            "materialized standardization is missing raw/standardized column(s) "
            f"{missing}"
        )
    if not requested:
        return

    expected, _ = attach_standardized(
        df.loc[:, requested].copy(),
        requested,
        reference=reference,
        support_label=support_label,
    )
    mismatched = []
    for source in requested:
        column = f"{source}_z"
        numeric_actual = pd.to_numeric(df[column], errors="coerce")
        invalid_actual = df[column].notna() & numeric_actual.isna()
        if invalid_actual.any():
            raise ValueError(
                f"materialized standardized column {column!r} contains "
                "nonnumeric values"
            )
        actual_values = numeric_actual.to_numpy(dtype=float)
        expected_values = expected[column].to_numpy(dtype=float)
        if not np.allclose(
            actual_values,
            expected_values,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        ):
            mismatched.append(column)
    if mismatched:
        raise ValueError(
            "materialized standardized values do not match the frozen "
            f"{support_label!r} reference: {mismatched}"
        )


def save_reference(reference: Reference, path) -> None:
    """Write a standardization reference to JSON (sorted, stable)."""
    with open(path, "w") as f:
        json.dump(reference, f, indent=2, sort_keys=True)


def load_reference(path) -> Reference:
    """Read a standardization reference written by :func:`save_reference`."""
    with open(path) as f:
        return json.load(f)
