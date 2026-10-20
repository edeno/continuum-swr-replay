"""Control run: the SWR-contrast content ladder with the at-well rule bypassed.

Answers one question that no comparison against an archived fit can: how much of
the difference between a shipped at-well content bundle and its predecessor is
the cohort rule, and how much is everything else that changed in between. It
runs ``scripts/content/fit_swr_contrast_content.py`` under the CURRENT code and
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

The output root must look like ``replay_content`` to the runner, which reads its
input tables and ``standardization_reference.json`` from the same root it writes
to. Symlinking every entry except ``models`` is enough::

    ROOT=/tmp/content_cohort_control
    mkdir -p "$ROOT/models"
    for entry in Processed-Data/replay_content/*; do
        [ "$(basename "$entry")" = models ] && continue
        ln -sfn "$(realpath "$entry")" "$ROOT/$(basename "$entry")"
    done
    uv run python scripts/exploratory/_content_cohort_control.py "$ROOT" omitted

Then compare ``$ROOT/models/swr_contrast_content_arm_omitted.pkl`` against the
shipped bundle and against the archived one.

Unsupported one-off (leading underscore, per ``scripts/README.md``): it monkey-
patches a runner's module globals and is not part of any pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from continuum_swr_replay.paths import model_dir

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


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: _content_cohort_control.py <output-root> <outcome-stratum> "
            "[extra runner args...]"
        )
    out_root = Path(sys.argv[1]).resolve()
    if not (model_dir("content_arm", out_root)).is_dir():
        raise SystemExit(
            f"{out_root}/models must exist and {out_root} must mirror "
            "replay_content (see this module's docstring)"
        )
    runner_argv = ["fit_swr_contrast_content.py", "--outcome-stratum", *sys.argv[2:]]
    sys.path.insert(0, str(REPO))
    sys.argv = runner_argv

    import scripts.content.fit_swr_contrast_content as runner

    runner.RC = out_root
    runner.select_event_cohort = bypass_event_cohort
    runner.main()


if __name__ == "__main__":
    main()
