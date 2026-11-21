"""The decode fit: a runner's record, its detector, the fit and the files it writes.

``scripts/decoding/run_decoding_bandit.py`` (the NonLocal 4-state detector) and
``run_decoding_contfrag.py`` (the ContFrag 2-state classifier) each hold a
:class:`DecodeRunner` naming what differs between them. :func:`perform_decoding`
builds that runner's detector (:func:`create_detector`) and fits it on one
epoch's spikes and position. What an epoch's decode does around the fit
(loading the session, the HPC decode's masks from the SWR source, a resume,
``data_over_time``, writing the files) reads ``analysis.event_definition`` and
``tables``, which this foundation package may not import, so it stays in
``scripts/decoding/_decode_runner.py:run_decode``.

``non_local_detector`` and ``decoding.transitions`` are imported inside
:func:`perform_decoding` and :func:`linear_environment`, so importing this
module loads no JAX.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr

from continuum_swr_replay.decoding.config import DecodingConfig
from continuum_swr_replay.parameters import PLACE_BIN_SIZE_1D, PLACE_BIN_SIZE_2D

if TYPE_CHECKING:
    from non_local_detector.environment import Environment

__all__ = [
    "DecodeRunner",
    "algorithm_params",
    "create_detector",
    "linear_environment",
    "perform_decoding",
]


@dataclass(frozen=True)
class DecodeRunner:
    """What one decode runner sets; everything else the two runners share.

    Two per-run arguments are the NonLocal runner's alone and are not here:
    ``non_local_threshold`` (``None`` for a runner without one, whose
    aggregation takes no threshold and whose ``data_over_time`` file name has
    no threshold part) and ``create_figurl`` (which calls ``figurl``).
    """

    detector_classes: Mapping[str, tuple[type, str]]
    """Brain area -> (detector class, ``"clusterless"`` or ``"sorted_spikes"``)."""
    config_class: type[DecodingConfig]
    detector_kwargs: Callable[[DecodingConfig], dict[str, Any]]
    """The detector's arguments beyond its environment and algorithm params:
    its discrete transitions, and any detector-level config."""
    aggregate: Callable[..., tuple[pd.DataFrame, xr.DataArray]]
    """``get_data_over_time`` or ``get_contfrag_data_over_time``."""
    load_saved: Callable[..., tuple[xr.Dataset, Any, Environment]]
    """``load_results`` or ``load_contfrag_results``, read on a resume."""
    file_suffix: str
    """Appended to ``{animal}{date}_{epoch}_{brain_area}`` in every file name."""
    hpc_log_likelihood: bool
    """Whether an HPC decode keeps its per-bin log-likelihood in ``results.nc``
    (``return_outputs="log_likelihood"``). The NonLocal runner does and the
    ContFrag runner does not."""
    figurl: Callable[..., None] | None
    """Draws the interactive view when ``create_figurl`` is set (NonLocal only)."""
    start_banner: str
    """The text centered in the log's ``START`` line."""
    decoding_label: str
    """Begins the ``<label> <brain_area>...`` log line before the fit."""
    model_line: str | None = None
    """A log line naming the model, after the git hash, or none."""


def algorithm_params(config: DecodingConfig, param_type: str) -> dict[str, Any]:
    """The detector's ``*_algorithm_params`` for `param_type`.

    ``"clusterless"`` is ``config.to_clusterless_params()`` and
    ``"sorted_spikes"`` is ``config.to_sorted_spikes_params()``.
    """
    to_params = {
        "clusterless": config.to_clusterless_params,
        "sorted_spikes": config.to_sorted_spikes_params,
    }[param_type]
    return {f"{param_type}_algorithm_params": to_params()}


def create_detector(
    detector_classes: Mapping[str, tuple[type, str]],
    detector_kwargs: Callable[[DecodingConfig], dict[str, Any]],
    brain_area: str,
    env: Environment,
    config: DecodingConfig,
) -> Any:
    """Build the detector for `brain_area`, unfitted.

    `detector_classes` and `detector_kwargs` are a :class:`DecodeRunner`'s
    (``runner.detector_classes``, ``runner.detector_kwargs``). Its discrete
    transitions come from `detector_kwargs`, never from the library's defaults.
    """
    if brain_area not in detector_classes:
        valid_areas = list(detector_classes.keys())
        raise ValueError(
            f"Invalid brain_area '{brain_area}'. Must be one of {valid_areas}"
        )

    detector_class, param_type = detector_classes[brain_area]
    return detector_class(
        environments=env,
        **algorithm_params(config, param_type),
        **detector_kwargs(config),
    )


def linear_environment(
    track_graph: nx.Graph, edge_order: list, edge_spacing: list
) -> Environment:
    """The unfitted environment of a 1-D decode: ``PLACE_BIN_SIZE_1D`` bins on
    `track_graph`, linearized in `edge_order` with `edge_spacing`."""
    # loads JAX, which importing this module must not
    from non_local_detector.environment import Environment

    return Environment(
        place_bin_size=PLACE_BIN_SIZE_1D,
        track_graph=track_graph,
        edge_order=edge_order,
        edge_spacing=edge_spacing,
    )


def perform_decoding(
    runner: DecodeRunner,
    logger: logging.Logger,
    spike_times: list,
    spike_waveform_features: list | None,
    position_info: pd.DataFrame,
    linear_edge_order: list,
    linear_edge_spacing: list,
    track_graph: nx.Graph,
    is_training: np.ndarray,
    brain_area: str,
    non_stationary: bool = False,
    is_1d: bool = True,
    is_missing: np.ndarray | None = None,
) -> tuple[xr.Dataset, Any, Environment]:
    """Fit `runner`'s detector for `brain_area` and decode.

    Raises ``ValueError`` when ``non_stationary`` is True: the library's
    non-stationary transitions are disabled (``NON_STATIONARY_REFUSAL``).

    ``is_training`` (shape (n_time,), on the position grid) selects the bins
    the encoding model is fitted on; ``is_missing`` (shape (n_time,), the same
    grid), when given, marks bins whose log-likelihood the decoder sets to 0,
    as an HPC decode's artifact mask does. An HPC decode fits on
    ``spike_waveform_features`` when given, and keeps its log-likelihood when
    ``runner.hpc_log_likelihood``.
    """
    # both load JAX, which importing this module must not
    from non_local_detector.environment import Environment

    from continuum_swr_replay.decoding.transitions import NON_STATIONARY_REFUSAL

    if non_stationary:
        raise ValueError(NON_STATIONARY_REFUSAL)

    if is_1d:
        env = linear_environment(track_graph, linear_edge_order, linear_edge_spacing)
    else:
        env = Environment(
            place_bin_size=PLACE_BIN_SIZE_2D,
        )

    config = runner.config_class()

    logger.info(f"{runner.decoding_label} {brain_area}...")

    detector = create_detector(
        runner.detector_classes, runner.detector_kwargs, brain_area, env, config
    )

    estimate_params = {
        "position_time": position_info.index.to_numpy(),
        "position": position_info[["head_position_x", "head_position_y"]].to_numpy(),
        "spike_times": spike_times,
        "time": position_info.index.to_numpy(),
        "discrete_transition_covariate_data": None,
    }

    # Apply the training mask (e.g. no_ripple) for every brain area — the
    # sorted-spikes detectors (mPFC/OFC) accept `is_training` too and otherwise
    # silently train on all bins, contrary to the logged training_type.
    estimate_params["is_training"] = is_training
    if is_missing is not None:
        estimate_params["is_missing"] = is_missing
    if brain_area == "HPC":
        if spike_waveform_features is not None:
            estimate_params["spike_waveform_features"] = spike_waveform_features
        if runner.hpc_log_likelihood:
            # Persist the per-timebin log-likelihood alongside the posterior so
            # the spike-evidence reality analysis
            # (scripts/event_evidence/analyze_event_spike_evidence.py) can read it instead
            # of recomputing the likelihood per event. Adds a
            # (n_time, n_state_bins) `log_likelihood` data_var to results.nc.
            estimate_params["return_outputs"] = "log_likelihood"

    results = detector.estimate_parameters(**estimate_params)

    return results, detector, env
