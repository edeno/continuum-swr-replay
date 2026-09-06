"""Live documentation must spell covariates with their canonical names.

The 2026-08-21 naming scheme renamed several covariates. Old spellings that
survive in prose are worse than a stale path: a reader who greps the tables
for ``q_arm`` finds nothing, and a reader who trusts the doc cites a column
that no longer exists.

This test greps the *live* documentation surface -- ``docs/``, ``paper/``,
``CLAUDE.md`` and ``scripts/README.md`` -- for the retired spellings.

Two kinds of file are exempt, because their job is to record the old names:

* ``docs/design/model_history.md`` -- the model-history record.
* dated design snapshots, ``docs/design/<YYYY-MM-DD>-*.md`` (which includes the
  naming scheme itself, ``2026-08-21-covariate-naming-scheme.md``). These are
  point-in-time proposals, never revised; rewriting their formulas to today's
  vocabulary would falsify the record.

The covariate registry (``covariate_registry.py``) also carries the old
spellings as aliases, but it is code, not documentation, and is out of scope
here -- this test greps documentation only.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Retired spelling -> canonical replacement, for the failure message.
RETIRED_NAMES: dict[str, str] = {
    "is_last_rewarded_elsewhere": "is_last_rewarded_noncurrent_arm",
    "swr_x_is_last_rewarded_elsewhere": "swr_x_is_last_rewarded_noncurrent_arm",
    "is_after_omission": "is_omitted_outcome (from nullable is_rewarded_outcome)",
    "swr_x_q_arm": "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    "q_arm": "reward_belief_mean_candidate_arm_preoutcome",
    "arm_uncertainty": "reward_belief_variance_candidate_arm_preoutcome",
    "recent_switch_rate": "patch_switch_fraction_last5_completed_trips",
    "reward_lag{j}": "known_reward_outcome_lag_{j}",
}

# Most retired names are literals. ``reward_lag{j}`` is a whole family --
# ``reward_lag1`` through ``reward_lag10`` were all renamed -- so it gets an
# explicit pattern rather than a literal match on lag 1 alone.
_EXPLICIT_PATTERNS: dict[str, re.Pattern[str]] = {
    "reward_lag{j}": re.compile(r"\breward_lag\d+\b"),
}

# Word boundaries keep ``arm_uncertainty_z`` and the ``swr_x_`` compounds from
# matching the bare stems (the compounds are listed separately above so they are
# still caught).
_PATTERNS = {
    name: _EXPLICIT_PATTERNS.get(name, re.compile(rf"\b{re.escape(name)}\b"))
    for name in RETIRED_NAMES
}

_ALLOWLIST = {Path("docs/design/model_history.md")}
_DATED_DESIGN_SNAPSHOT = re.compile(r"^\d{4}-\d{2}-\d{2}-.*\.md$")


def _is_exempt(rel: Path) -> bool:
    if rel in _ALLOWLIST:
        return True
    return rel.parent == Path("docs/design") and bool(
        _DATED_DESIGN_SNAPSHOT.match(rel.name)
    )


def _scanned_files() -> list[Path]:
    files = [REPO / "CLAUDE.md", REPO / "scripts" / "README.md"]
    for root in ("docs", "paper"):
        files.extend(sorted((REPO / root).rglob("*.md")))
    return [f for f in files if f.is_file()]


def test_live_docs_use_canonical_covariate_names() -> None:
    hits: list[str] = []
    for path in _scanned_files():
        rel = path.relative_to(REPO)
        if _is_exempt(rel):
            continue
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            for name, pattern in _PATTERNS.items():
                if pattern.search(line):
                    hits.append(f"{rel}:{lineno}: `{name}` -> `{RETIRED_NAMES[name]}`")

    assert not hits, (
        "Retired covariate names in live documentation "
        f"({len(hits)} occurrence(s)):\n  " + "\n  ".join(hits)
    )
