"""Control run: the SWR-contrast content ladder with the at-well rule bypassed.

Answers one question that no comparison against an archived fit can: how much of
the difference between a shipped at-well content bundle and its predecessor is
the cohort rule, and how much is everything else that changed in between. It
runs ``scripts/replay_models/content/fit_swr_contrast_content.py`` under the CURRENT code and
the CURRENT tables with :func:`select_event_cohort` replaced by an identity
bypass, writing into a throwaway output root so no canonical artifact is
touched.

What it established on 2026-09-12, on the ``omitted`` stratum: the bypassed run
reproduces the pre-cohort archive EXACTLY (700/700 arm rows, 259/259 patch,
``assert_frame_equal(check_exact=True)`` once the four diagnostic columns added
since are dropped), so the archive is a valid baseline. Against the shipped
at-well bundle the fitted event set is unchanged on matched cells:
10,923 (session, rung, population) cells appear on both sides, and only
``C0_armfe`` cells differ in ``n_events``, so the rule removes no event any
other rung was fitting. That is a statement about the matched cells only -- 45
cells exist only after the refit and 27 only before, where a rung
reached a per-session fit on one side and not the other. Every coefficient
moves regardless, because the runner's fit-time standardization reference is
computed over the assembled cohort, so restricting the cohort changes the unit
of every ``_z`` covariate.

The output root must be a ``replay_content`` folder to the runner, which reads
its input tables and ``standardization_reference.json`` from the same root it
writes to: the run points ``$CSR_PROCESSED_DATA_DIR`` (the one place every data
root is resolved from, ``paths.processed_data_dir()``) at the output root's
parent, so the root must be named ``replay_content`` and its parent must not be
the live ``Processed-Data``. Symlinking every entry except the output directory
``models_swr_contrast`` is enough (a symlinked one would send the run's output
into the canonical directory); before the runner starts, the run refuses an
output directory that resolves under the live ``Processed-Data``::

    ROOT=/tmp/content_cohort_control/replay_content
    mkdir -p "$ROOT/models_swr_contrast"
    for entry in Processed-Data/replay_content/*; do
        [ "$(basename "$entry")" = models_swr_contrast ] && continue
        ln -sfn "$(realpath "$entry")" "$ROOT/$(basename "$entry")"
    done
    uv run python scripts/exploratory/_content_cohort_control.py "$ROOT" omitted

Then compare ``$ROOT/models_swr_contrast/swr_contrast_content_arm_omitted.pkl``
against the shipped bundle and against the archived one. (Before 2026-10-24 the
runner wrote into ``models/``, where the 2026-09-12 run above put its output.)

Unsupported one-off (leading underscore, per ``scripts/README.md``): it monkey-
patches a runner's module globals and is not part of any pipeline.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from continuum_swr_replay.paths import PROCESSED_DATA_DIR_ENV, ROOT_DIR, model_dir

REPO = Path(__file__).resolve().parents[2]


def bypass_event_cohort(
    frame: pd.DataFrame, cohort: str = "at_well", event_col: str = "event_id"
) -> tuple[pd.DataFrame, dict]:
    """Stand-in for ``select_event_cohort`` that keeps every event.

    Returns the same report shape the runner logs, labelled ``bypassed`` so the
    log cannot be mistaken for a canonical run.
    """
    counts = frame.groupby(frame["occupancy_state"].fillna("missing"), observed=True)[
        event_col
    ].nunique()
    tallies = {str(state): int(count) for state, count in counts.items()}
    n_events = int(frame[event_col].nunique())
    return frame.copy(), {
        "cohort": "bypassed",
        "rule": "CONTROL: no occupancy restriction",
        "n_events_before": n_events,
        "n_events_after": n_events,
        "counts_before": tallies,
        "counts_after": tallies,
    }


def use_output_root(out_root: Path) -> Path:
    """Make `out_root` the run's ``replay_content`` and return its output folder.

    Points ``$CSR_PROCESSED_DATA_DIR`` at ``out_root``'s parent, the one patch
    point every root is resolved from, then checks that the folder the runner
    will write to (``model_dir("swr_contrast")``) is a real directory and not
    under the live ``Processed-Data``.
    """
    live = (ROOT_DIR / "Processed-Data").resolve()
    out_root = out_root.resolve()
    if out_root.name != "replay_content" or out_root.parent == live:
        raise SystemExit(
            f"{out_root} must be a folder named replay_content outside {live} "
            "(see this module's docstring)"
        )
    os.environ[PROCESSED_DATA_DIR_ENV] = str(out_root.parent)
    out_dir = model_dir("swr_contrast")
    if out_dir != out_root / "models_swr_contrast":  # pragma: no cover - drift
        raise SystemExit(f"the runner would write to {out_dir}, not under {out_root}")
    if not out_dir.is_dir() or out_dir.is_symlink():
        raise SystemExit(
            f"{out_dir} must be a real directory and {out_root} must mirror "
            "replay_content (see this module's docstring)"
        )
    if out_dir.resolve().is_relative_to(live):
        raise SystemExit(f"{out_dir} resolves under the live {live}; refusing to write")
    return out_dir


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: _content_cohort_control.py <output-root> <outcome-stratum> "
            "[extra runner args...]"
        )
    # Before anything resolves a root: the runner reads and writes through
    # paths, which reads the variable this sets.
    use_output_root(Path(sys.argv[1]))
    runner_argv = ["fit_swr_contrast_content.py", "--outcome-stratum", *sys.argv[2:]]
    sys.path.insert(0, str(REPO))
    sys.argv = runner_argv

    import scripts.replay_models.content.fit_swr_contrast_content as runner
    from continuum_swr_replay.analysis.replay_models import contrast_aggregation

    # The cohort rule is applied where the cohorts are assembled.
    contrast_aggregation.select_event_cohort = bypass_event_cohort
    runner.main()


if __name__ == "__main__":
    main()
