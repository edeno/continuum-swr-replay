"""Where content-ladder fits land, and how their numbers are read.

The directory names a content fit publishes into are decided here, not in the
fitting runner, so a figure or QA script that finds fits imports these rules
without importing ``scripts/content/fit_replay_content_models.py`` (its argparse,
logging setup and heavy imports). The runner builds its CLI choices from the
same tuples, so a new choice cannot be added to the parser without also
entering the set of bundle names :func:`canonical_output_subdirs` returns.

:func:`scale_label` and :func:`frame_estimator` state how a fitted bundle's
likelihood-based numbers must be read, for the runner's printout and the
figures that plot the same comparison.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from continuum_swr_replay.analysis.replay.content_response import (
    ESTIMATOR_CHOICES,
    RESPONSE_CHOICES,
    default_response_for_estimator,
)
from continuum_swr_replay.paths import MODEL_SUBDIRS

# The CLI's accepted values, as constants: the canonical output-directory names
# are the product of these, so a new choice cannot be added to the parser without
# also entering the set of bundles a partial run is forbidden to overwrite.
EVENT_SOURCE_CHOICES = ("non_local", "SWR", "MUA")
BRAIN_AREA_CHOICES = ("HPC", "mPFC", "OFC")
HIERARCHY_CHOICES = ("arm", "patch")
SWR_SUBSET_CHOICES = ("all", "T", "F")
GRAIN_CHOICES = ("timebin", "event")


def resolve_contfrag_paths(
    event_source: str,
    brain_area: str,
    hierarchy_kind: str,
) -> tuple[str, str]:
    """Return ``(table_name, output_subdir)`` for a fit invocation.

    Pure helper extracted so the prep ↔ fitter filename contract has a
    direct unit-test target. ``event_source`` is one of
    ``"non_local"`` / ``"SWR"`` / ``"MUA"``; ``hierarchy_kind`` is
    ``"candidate_arm"`` / ``"patch"``.

    Non-local fits stay unsuffixed (their prep is single-area /
    single-source). ContFrag fits append ``_{brain_area}`` to the
    table name and ``_{source_lower}_{brain_area}`` to the output
    subdir so SWR/HPC, SWR/mPFC, MUA/HPC … never overwrite each
    other's ``M0.pkl`` / ``all_models.pkl``.
    """
    if event_source == "non_local":
        if hierarchy_kind == "patch":
            return "event_patch_table", MODEL_SUBDIRS["content_patch"]
        return "event_arm_table", MODEL_SUBDIRS["content_arm"]

    base_table = (
        f"event_patch_table_{event_source.lower()}"
        if hierarchy_kind == "patch"
        else f"event_arm_table_{event_source.lower()}"
    )
    table_name = f"{base_table}_{brain_area}"

    base_subdir = (
        MODEL_SUBDIRS["content_patch"]
        if hierarchy_kind == "patch"
        else MODEL_SUBDIRS["content_arm"]
    )
    output_subdir = f"{base_subdir}_{event_source.lower()}_{brain_area}"
    return table_name, output_subdir


def canonical_output_subdir(
    event_source: str,
    brain_area: str,
    hierarchy_kind: str,
    swr_subset: str,
    grain: str,
    response: str = "fraction",
    leaf_only: bool = False,
    estimator: str = "conditional_poisson",
) -> str:
    """Directory name a FULL-ladder run of one configuration publishes into.

    The base name comes from :func:`resolve_contfrag_paths`; the SWR subset, the
    event grain, the response scale and the candidate set each append a suffix,
    in that order, so an event-grain subset run has its own bundle rather than
    publishing over the timebin one.

    Parameters
    ----------
    event_source : str
        One of :data:`EVENT_SOURCE_CHOICES`.
    brain_area : str
        One of :data:`BRAIN_AREA_CHOICES` (ignored for ``non_local``).
    hierarchy_kind : str
        One of :data:`HIERARCHY_CHOICES`.
    swr_subset : str
        One of :data:`SWR_SUBSET_CHOICES`; ``"all"`` adds no suffix.
    grain : str
        One of :data:`GRAIN_CHOICES`; ``"timebin"`` adds no suffix.
    response : str, optional
        One of :data:`RESPONSE_CHOICES`; the shipped ``"fraction"`` adds no
        suffix. A count fit states its likelihood, AIC and SEs at the timebin
        grain rather than the event grain, so its rungs are not comparable with
        a fraction bundle's and must not replace them.
    leaf_only : bool, optional
        Whether the fit conditions on the six leaf arms instead of all nine
        candidates. A six-category conditional likelihood is on a different
        scale from a nine-category one, so a leaf-only ladder gets its own
        ``_leaf`` bundle.
    estimator : str, optional
        One of :data:`~continuum_swr_replay.analysis.replay.content_response.ESTIMATOR_CHOICES`.
        The Dirichlet-multinomial appends ``_dm`` LAST, after every other
        suffix, so a reader can strip it to find the conditional-Poisson bundle
        the fit is the overdispersed counterpart of. Its likelihood carries the
        multinomial coefficient the conditional one drops, so its AIC is not on
        the conditional bundle's scale and must never replace it. The response
        suffix is omitted where the response is the estimator's own default,
        which is what the ``_hard_count`` suffix would otherwise say twice.

    Returns
    -------
    str
        The bundle directory name, relative to the run's input directory.

    Raises
    ------
    ValueError
        If ``leaf_only`` is requested for the patch hierarchy, which has no
        leaf subset.
    """
    if leaf_only and hierarchy_kind == "patch":
        raise ValueError(
            "leaf_only is an arm-hierarchy candidate set; the three patches are "
            "not arms and have no leaf subset."
        )
    _table_name, subdir = resolve_contfrag_paths(
        event_source, brain_area, hierarchy_kind
    )
    if swr_subset != "all":
        subdir = f"{subdir}_swr_{swr_subset}"
    if grain == "event":
        subdir = f"{subdir}_event_grain"
    if response != default_response_for_estimator(estimator):
        subdir = f"{subdir}_{response}"
    if leaf_only:
        subdir = f"{subdir}_leaf"
    if estimator != "conditional_poisson":
        subdir = f"{subdir}_dm"
    return subdir


def canonical_output_subdirs() -> frozenset[str]:
    """Every canonical bundle name, over all CLI configurations.

    A partial ladder is fit on its own complete-case cohort, so it must not be
    written into ANY of these -- not just the one the current invocation would
    have used. Built from :func:`canonical_output_subdir` over the product of
    the CLI choice tuples, so it cannot drift from the names full runs produce.
    """
    return frozenset(
        canonical_output_subdir(
            source, area, hierarchy_kind, subset, grain, response, leaf_only, estimator
        )
        for source in EVENT_SOURCE_CHOICES
        for area in BRAIN_AREA_CHOICES
        for hierarchy_kind in HIERARCHY_CHOICES
        for subset in SWR_SUBSET_CHOICES
        for grain in GRAIN_CHOICES
        for estimator in ESTIMATOR_CHOICES
        # Only the conditional Poisson has a choice of response; the
        # Dirichlet-multinomial is defined on counts and nothing else.
        for response in (
            RESPONSE_CHOICES
            if estimator == "conditional_poisson"
            else (default_response_for_estimator(estimator),)
        )
        # The patch hierarchy has no leaf-only variant to guard.
        for leaf_only in ((False,) if hierarchy_kind == "patch" else (False, True))
    )


# The leaf-only belief chain, as documented in the usage block of
# scripts/content/fit_replay_content_models.py and in docs/models_table.md:
# M0-M4 refit on the six leaves alongside Mv/Mu/Mvu so every rung shares one
# six-category conditional likelihood.
BELIEF_CHAIN_MODEL_NAMES: tuple[str, ...] = (
    "M0",
    "M0a",
    "M1",
    "M2",
    "M3",
    "M4",
    "Mv",
    "Mu",
    "Mvu",
)


def partial_ladder_subdir(canonical_subdir: str, model_names: Sequence[str]) -> str:
    """Bundle name for a run that fits ``model_names`` rather than a full ladder.

    A partial ladder is fit on its own complete-case cohort, so it is keyed by
    the rungs it fits and never written next to (or over) a canonical bundle.

    Parameters
    ----------
    canonical_subdir : str
        The name :func:`canonical_output_subdir` gives the same configuration.
    model_names : Sequence[str]
        The requested rungs, in ladder order.

    Returns
    -------
    str
        The bundle directory name, relative to the run's input directory.
    """
    return f"{canonical_subdir}_{'-'.join(model_names)}"


def belief_chain_output_subdir() -> str:
    """Bundle directory the leaf-only belief chain (``Mv``/``Mu``/``Mvu``) writes.

    Derived from the same helpers the runner uses, so a reader of ``Mu.pkl``
    cannot drift from where ``fit_replay_content_models.py --leaf-only --models
    M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu`` publishes it.

    Returns
    -------
    str
        The bundle directory name, relative to ``PROCESSED_DATA_DIR /
        "replay_content"``.
    """
    return partial_ladder_subdir(
        canonical_output_subdir(
            event_source="non_local",
            brain_area="HPC",  # ignored for non_local (see resolve_contfrag_paths)
            hierarchy_kind="arm",
            swr_subset="all",
            grain="timebin",
            response="fraction",
            leaf_only=True,
        ),
        BELIEF_CHAIN_MODEL_NAMES,
    )


# How a fit's likelihood-based numbers should be read, by estimator. A count
# fit spends one observation per decoded timebin, of which an event supplies a
# median of ~32 that are near-copies of each other, so its llf/AIC/BIC are on a
# scale the data do not support. A fraction or dominant-choice fit spends one
# observation per event; its likelihood is a quasi-likelihood (the response is
# not a count), so AIC and LRT differences on it are event-scale quasi
# quantities rather than exact information criteria.
_SCALE_LABELS = {
    "conditional_poisson_fraction": "event-scale (quasi)",
    "conditional_logit_dominant": "event-scale (quasi)",
    "conditional_poisson_count": "timebin-scale",
    # The DM likelihood is a proper probability of the event's whole count
    # vector -- multinomial coefficient included -- so its AIC and LRT are exact
    # information criteria at the event grain rather than quasi ones. That same
    # constant is why they must not be differenced against a conditional fit's.
    "dirichlet_multinomial": "event-scale (full likelihood)",
}
_DEFAULT_SCALE_LABEL = "unlabelled scale"


def scale_label(estimator: object) -> str:
    """Return how likelihood-based numbers from ``estimator`` must be read.

    Parameters
    ----------
    estimator : object
        An estimator name from a fit result, or anything else (an absent or
        mixed column) for which no scale can be claimed.

    Returns
    -------
    str
        ``"event-scale (quasi)"``, ``"timebin-scale"``, or
        ``"unlabelled scale"`` when the estimator is unknown.
    """
    return _SCALE_LABELS.get(estimator, _DEFAULT_SCALE_LABEL)


def frame_estimator(frame: pd.DataFrame) -> object:
    """The single estimator a frame's rows were fit with, else ``None``."""
    if "estimator" not in frame.columns:
        return None
    values = set(frame["estimator"].dropna().unique())
    return values.pop() if len(values) == 1 else None
