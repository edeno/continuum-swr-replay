#!/usr/bin/env python
"""TEMPORARY, exploratory StateScript -> contingency-table parser.

Parse the per-port nominal reward probabilities, the contingency block index,
and the block change times out of the Spyglass ``StateScriptFile`` runtime log,
align the StateScript millisecond clock to the neural/DIO clock, and annotate
every stop in ``stop_table`` with the active block's nominal value structure.

The line-parse (``<ts> <name> = <int>``) and the DIO-offset alignment idea
(``calculate_time_offset`` -> a single constant additive offset to a reference
DIO event series) are ADAPTED -- not imported -- from
``trodes_to_nwb/convert_statescript.py`` on branch ``statescript``
(commit ``f3a0c1c``,
https://github.com/LorenFrankLab/trodes_to_nwb/blob/statescript/src/trodes_to_nwb/convert_statescript.py).
That parser is generic and knows nothing about ``contingency`` / ``leafProbs``
semantics; this module supplies that domain logic on top. The repo deliberately
does NOT depend on ``trodes_to_nwb`` -- only the ~20 lines that matter are
re-implemented here.

TEMPORARY -- replace with the canonical parsed-events table when it becomes
available. This is a scratch-tier (``_``-prefixed) module that exists only to
unblock the downstream rate analysis; do not build permanent infrastructure on
top of it. Consumers should bind to the produced column schema (see
``shared-contracts.md``), not to this parser, so the source can be swapped with
a one-line change.

Usage::

    uv run python scripts/exploratory/_statescript_contingency.py                 # all sessions
    uv run python scripts/exploratory/_statescript_contingency.py --max-per-animal 1  # smoke
"""

from __future__ import annotations

import argparse
import logging
import re

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import WELL_PATCH_MAPPING
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

# The documented contingency schedule values (fractions). 1.0 only appears in
# the first (exposure) session. Used to assert the parse, not to drive it.
VALID_P_REWARD = frozenset({0.0, 0.2, 0.5, 0.8, 1.0})

# Maximum tolerated alignment residual (s) per matched reference event.
MAX_ALIGN_RESIDUAL_S = 0.050

# Adapted from trodes_to_nwb@statescript: a runtime assignment line is
# ``<ts_ms> <name> = <int>``. Header declarations are ``#``-commented and so
# never match (the leading ``\d`` requires a timestamp).
_ASSIGN = re.compile(r"^\s*(\d+)\s+([A-Za-z]\w*)\s*=\s*(-?\d+)\s*$")
# A beam-break poke edge: ``<ts_ms> UP <port>`` (ports 1-6). Pumps (DIO 7-12)
# are controller outputs and are NOT logged as UP/DOWN, so pokes are the clean
# 1:1 StateScript<->DIO reference series for clock alignment.
_POKE_UP = re.compile(r"^\s*(\d+)\s+UP\s+(\d+)\s*$")


def parse_assignments(content: str) -> pd.DataFrame:
    """Extract every ``<ts_ms> name = int`` runtime assignment line.

    Parameters
    ----------
    content : str
        Full StateScript runtime log string.

    Returns
    -------
    pd.DataFrame
        Columns ``ts_ms`` (int), ``name`` (str), ``value`` (int), one row per
        runtime assignment line. ``#``-comments and non-assignment lines are
        ignored.
    """
    rows = [
        (int(m[1]), m[2], int(m[3]))
        for ln in content.splitlines()
        if (m := _ASSIGN.match(ln))
    ]
    return pd.DataFrame(rows, columns=["ts_ms", "name", "value"])


def parse_poke_times_ms(content: str) -> np.ndarray:
    """StateScript beam-break poke ``UP`` edge timestamps (ms), sorted.

    Parameters
    ----------
    content : str
        Full StateScript runtime log string.

    Returns
    -------
    np.ndarray, shape (n_pokes,)
        Poke-onset timestamps in milliseconds (StateScript clock).
    """
    ts = [int(m[1]) for ln in content.splitlines() if (m := _POKE_UP.match(ln))]
    return np.sort(np.asarray(ts, dtype=float))


def reconstruct_blocks(assignments: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct the contingency-block timeline from runtime assignments.

    Each ``contingency = k`` line marks a (re)statement of the active block; the
    ``leafProbs1..6`` lines that follow it set the six port probabilities
    (percent -> fraction). A block is the settled ``(contingency, prob-vector)``
    after a ``contingency`` line and before the next one. The log restates the
    same contingency many times (and updates the six leafProbs one line at a
    time, producing transient intermediate vectors); consecutive identical
    ``(contingency, prob-vector)`` segments are collapsed so each genuine
    change is one row.

    Parameters
    ----------
    assignments : pd.DataFrame
        Output of :func:`parse_assignments`.

    Returns
    -------
    pd.DataFrame
        Columns ``block_number`` (0-indexed), ``contingency`` (the logged
        index), ``t_start_ms`` (the ``contingency`` line's timestamp), and
        ``p_reward_port0..5`` (fractions). One row per distinct block.
    """
    mask = assignments["name"].isin(
        ["contingency", *[f"leafProbs{i}" for i in range(1, 7)]]
    )
    rows = assignments[mask].itertuples(index=False)

    segments: list[list] = []  # [t_start_ms, contingency, {leafProbs_name: pct}]
    current: list | None = None
    for ts_ms, name, value in rows:
        if name == "contingency":
            current = [ts_ms, value, {}]
            segments.append(current)
        elif current is not None:
            current[2][name] = value

    records: list[dict] = []
    last_key: tuple | None = None
    for t_start_ms, contingency, leaf in segments:
        if len(leaf) < 6:  # incomplete leafProbs set -> not yet a settled block
            continue
        vec = tuple(leaf[f"leafProbs{i}"] / 100.0 for i in range(1, 7))
        key = (contingency, vec)
        if key == last_key:
            continue
        last_key = key
        records.append(
            {
                "contingency": int(contingency),
                "t_start_ms": int(t_start_ms),
                **{f"p_reward_port{p}": vec[p] for p in range(6)},
            }
        )

    blocks = pd.DataFrame(records)
    blocks.insert(0, "block_number", np.arange(len(blocks), dtype=int))
    return blocks


def solve_time_offset(
    statescript_event_ms: np.ndarray,
    dio_event_sec: np.ndarray,
    *,
    match_tolerance_s: float = MAX_ALIGN_RESIDUAL_S,
) -> tuple[float, float, int]:
    """Solve the constant additive offset aligning the StateScript ms clock.

    StateScript timestamps are milliseconds at the same clock rate as the
    neural/DIO clock (verified: no drift/scale), while the DIO clock is Unix
    epoch seconds. So ``t_neural_sec = ts_ms / 1000 + offset``. Adapted from
    ``trodes_to_nwb.convert_statescript.calculate_time_offset``: find the offset
    that best registers the two event series. A 2-ms histogram of all pairwise
    differences gives a robust coarse peak; the offset is then refined to the
    median of the matched pairs.

    Parameters
    ----------
    statescript_event_ms : np.ndarray, shape (n_ss,)
        StateScript reference event timestamps (ms), e.g. poke ``UP`` edges.
    dio_event_sec : np.ndarray, shape (n_dio,)
        DIO reference event times (Unix s), e.g. ``Poke%`` ON edges in the
        epoch window.
    match_tolerance_s : float
        A StateScript event counts as matched if a DIO event lies within this.

    Returns
    -------
    offset : float
        Additive offset (s): ``t_neural_sec = ts_ms / 1000 + offset``.
    median_residual : float
        Median |residual| (s) over matched events.
    n_matched : int
        Number of matched events.

    Raises
    ------
    ValueError
        If either series is empty.
    """
    ss_s = np.asarray(statescript_event_ms, dtype=float) / 1000.0
    dio = np.sort(np.asarray(dio_event_sec, dtype=float))
    if ss_s.size == 0 or dio.size == 0:
        raise ValueError("both event series must be non-empty to align")

    diffs = (dio[:, None] - ss_s[None, :]).ravel()
    edges = np.arange(diffs.min(), diffs.max() + 0.002, 0.002)
    counts, edges = np.histogram(diffs, bins=edges)
    coarse_offset = float(edges[counts.argmax()])

    def _residuals(offset: float) -> tuple[np.ndarray, np.ndarray]:
        shifted = ss_s + offset
        idx = np.clip(np.searchsorted(dio, shifted), 1, dio.size - 1)
        left, right = dio[idx - 1], dio[idx]
        nearest = np.where(np.abs(right - shifted) < np.abs(left - shifted), right, left)
        return np.abs(nearest - shifted), nearest

    resid, nearest = _residuals(coarse_offset)
    matched = resid < match_tolerance_s
    if matched.any():
        coarse_offset = float(np.median(nearest[matched] - ss_s[matched]))
        resid, _ = _residuals(coarse_offset)
        matched = resid < match_tolerance_s

    return coarse_offset, float(np.median(resid[matched])) if matched.any() else np.inf, int(matched.sum())


def _best_patch(p_reward: np.ndarray) -> int:
    """Patch (1-3) with the highest two-port average nominal value.

    ``p_reward`` is indexed by port 0..5; ports map to wells 0..5 ->
    :data:`WELL_PATCH_MAPPING`. Patch value = mean of its two ports.
    """
    patch_means: dict[int, float] = {}
    for patch in (1, 2, 3):
        ports = [w for w, pa in WELL_PATCH_MAPPING.items() if pa == patch]
        patch_means[patch] = float(np.mean([p_reward[w] for w in ports]))
    return max(patch_means, key=patch_means.get)


def _patch_value_diff(p_reward: np.ndarray) -> float:
    """Best-patch value minus second-best-patch value (>= 0)."""
    means = []
    for patch in (1, 2, 3):
        ports = [w for w, pa in WELL_PATCH_MAPPING.items() if pa == patch]
        means.append(float(np.mean([p_reward[w] for w in ports])))
    means.sort(reverse=True)
    return means[0] - means[1]


def annotate_stops(
    stops: pd.DataFrame,
    blocks: pd.DataFrame,
    offset: float,
) -> pd.DataFrame:
    """Map each stop to its active block and the nominal value structure.

    Parameters
    ----------
    stops : pd.DataFrame
        One session's stops. Must carry ``stop_id``, ``stop_start`` (neural s),
        ``current_arm`` (the port 0..5 the animal is at), and ``reward_outcome``.
    blocks : pd.DataFrame
        :func:`reconstruct_blocks` output for the session.
    offset : float
        The additive offset from :func:`solve_time_offset` (s).

    Returns
    -------
    pd.DataFrame
        One row per ``stop_id`` with the
        ``shared-contracts.md#contingency-table-schema`` columns.
    """
    block_start_sec = blocks["t_start_ms"].to_numpy(dtype=float) / 1000.0 + offset
    port_cols = [f"p_reward_port{p}" for p in range(6)]
    port_mat = blocks[port_cols].to_numpy(dtype=float)

    stop_start = stops["stop_start"].to_numpy(dtype=float)
    # active block = last block whose start <= stop_start (-1 if before block 0).
    active = np.searchsorted(block_start_sec, stop_start, side="right") - 1

    records: list[dict] = []
    # trials_since_block_change: stop's 0-indexed rank within its block (stops
    # are port visits ~ trials; verified ~1:1 on a spot session).
    rank_within_block: dict[int, int] = {}
    order = np.argsort(stop_start, kind="stable")
    rank_lookup = np.full(len(stops), -1, dtype=int)
    for pos in order:
        b = int(active[pos])
        rank_lookup[pos] = rank_within_block.get(b, 0)
        rank_within_block[b] = rank_within_block.get(b, 0) + 1

    stops_reset = stops.reset_index(drop=True)
    for i, row in stops_reset.iterrows():
        b = int(active[i])
        if b < 0:  # stop precedes the first logged contingency -> no value
            continue
        p_reward = port_mat[b]
        current_arm = row["current_arm"]
        if pd.isna(current_arm):
            p_current = np.nan
            rpe = np.nan
        else:
            p_current = float(p_reward[int(current_arm)])
            rpe = float(row["reward_outcome"]) - p_current
        rec = {
            "stop_id": row["stop_id"],
            "block_number": int(blocks.loc[b, "block_number"]),
            "trials_since_block_change": int(rank_lookup[i]),
            **{f"p_reward_port{p}": float(p_reward[p]) for p in range(6)},
            "p_reward_current": p_current,
            "p_reward_best": float(np.max(p_reward)),
            "is_best_patch": (
                False
                if pd.isna(current_arm)
                else WELL_PATCH_MAPPING[int(current_arm)] == _best_patch(p_reward)
            ),
            "rpe": rpe,
            "value_diff_patch": _patch_value_diff(p_reward),
        }
        records.append(rec)

    return pd.DataFrame.from_records(records)


def _assert_invariants(table: pd.DataFrame) -> None:
    """Assert the schema invariants (do not weaken)."""
    port_cols = [f"p_reward_port{p}" for p in range(6)]
    vals = table[port_cols].to_numpy(dtype=float)
    finite = vals[np.isfinite(vals)]
    bad = sorted(set(np.round(finite, 3)) - VALID_P_REWARD)
    assert not bad, f"p_reward values outside the documented schedule: {bad}"
    vd = table["value_diff_patch"].dropna().to_numpy()
    assert (vd >= -1e-9).all(), "value_diff_patch must be >= 0"
    assert table["stop_id"].is_unique, "one row per stop_id"


def build_session_contingency_table(
    content: str,
    poke_times_ms: np.ndarray,
    dio_poke_on_sec: np.ndarray,
    stops: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """End-to-end per-session parse: log + DIO + stops -> contingency table.

    Returns the annotated table and a provenance dict (offset, residual,
    n_matched, n_blocks). Raises ``ValueError`` if alignment cannot be solved.
    """
    assignments = parse_assignments(content)
    blocks = reconstruct_blocks(assignments)
    if blocks.empty:
        raise ValueError("no contingency blocks parsed")

    offset, residual, n_matched = solve_time_offset(poke_times_ms, dio_poke_on_sec)
    if residual > MAX_ALIGN_RESIDUAL_S:
        raise ValueError(
            f"alignment residual {residual * 1e3:.1f} ms exceeds "
            f"{MAX_ALIGN_RESIDUAL_S * 1e3:.0f} ms ({n_matched} matched)"
        )

    table = annotate_stops(stops, blocks, offset)
    _assert_invariants(table)
    provenance = {
        "offset": offset,
        "residual_ms": residual * 1e3,
        "n_matched": n_matched,
        "n_blocks": len(blocks),
    }
    return table, provenance


# ── Real-data runner ──


def _epoch_to_int(epoch_name: str) -> int:
    """``"02_r1"`` -> ``2`` (the integer epoch StateScriptFile keys on)."""
    return int(epoch_name.split("_")[0])


def _load_session_inputs(nwb_file_name: str, epoch_name: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Fetch the StateScript log, its poke edges, and DIO poke ON edges.

    Returns ``(content, poke_times_ms, dio_poke_on_sec)`` restricted to the
    epoch's valid-time window. Spyglass imports are deferred (DataJoint opens a
    connection at import time).
    """
    from spyglass.common import DIOEvents, IntervalList, StateScriptFile

    epoch_int = _epoch_to_int(epoch_name)
    content = (
        StateScriptFile() & {"nwb_file_name": nwb_file_name, "epoch": epoch_int}
    ).fetch_nwb()[0]["file"].content
    poke_times_ms = parse_poke_times_ms(content)

    valid = (
        IntervalList()
        & {"nwb_file_name": nwb_file_name}
        & f"interval_list_name='{epoch_name}'"
    ).fetch1("valid_times")
    t0, t1 = float(valid[0][0]), float(valid[-1][-1])

    dio = (
        DIOEvents()
        & f"nwb_file_name='{nwb_file_name}'"
        & "dio_event_name LIKE 'Poke%'"
    ).fetch_nwb()
    on = []
    for d in dio:
        ts = np.asarray(d["dio"].timestamps)
        data = np.asarray(d["dio"].data)
        on.append(ts[data == 1])
    on = np.sort(np.concatenate(on)) if on else np.array([])
    on = on[(on >= t0) & (on <= t1)]
    return content, poke_times_ms, on


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help="Parse only the first N sessions per animal (smoke / spot-check).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    rc = PROCESSED_DATA_DIR / "replay_content"
    stop_table = pd.read_pickle(rc / "stop_table.pkl")

    session_cols = ["animal", "day", "epoch"]
    sessions = stop_table[session_cols].drop_duplicates().sort_values(session_cols)
    if args.max_per_animal is not None:
        sessions = sessions.groupby("animal", sort=False).head(args.max_per_animal)

    tables: list[pd.DataFrame] = []
    n_ok = n_fail = 0
    for animal, day, epoch in sessions.itertuples(index=False):
        nwb_file_name = f"{animal}{day}_.nwb"
        stops = stop_table[
            (stop_table["animal"] == animal)
            & (stop_table["day"] == day)
            & (stop_table["epoch"] == epoch)
        ]
        try:
            content, poke_ms, dio_on = _load_session_inputs(nwb_file_name, epoch)
            table, prov = build_session_contingency_table(
                content, poke_ms, dio_on, stops
            )
        except Exception as exc:  # noqa: BLE001 - log and skip; never crash the batch
            n_fail += 1
            logger.warning("SKIP %s %s: %s", nwb_file_name, epoch, exc)
            continue
        n_ok += 1
        logger.info(
            "OK %s %s: %d stops, %d blocks, offset=%.3f, residual=%.1f ms (%d matched)",
            nwb_file_name,
            epoch,
            len(table),
            prov["n_blocks"],
            prov["offset"],
            prov["residual_ms"],
            prov["n_matched"],
        )
        tables.append(table)

    if not tables:
        logger.error("No sessions parsed; nothing written.")
        return

    contingency_table = pd.concat(tables, ignore_index=True)
    out_dir = rc / "models_per_bin_rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "contingency_table.pkl"
    contingency_table.to_pickle(out)

    port_cols = [f"p_reward_port{p}" for p in range(6)]
    hist = (
        contingency_table[port_cols]
        .stack()
        .value_counts()
        .sort_index()
        .to_dict()
    )
    logger.info(
        "Wrote %d stops from %d sessions (%d failed) to %s",
        len(contingency_table),
        n_ok,
        n_fail,
        out,
    )
    logger.info("p_reward value histogram: %s", hist)
    logger.info(
        "block-count distribution: %s",
        contingency_table.groupby("stop_id")["block_number"].first().value_counts().sort_index().to_dict(),
    )


if __name__ == "__main__":
    main()
