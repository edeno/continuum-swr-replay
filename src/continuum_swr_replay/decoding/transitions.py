"""Discrete-transition settings of the replay decoders, as explicit, sourced values.

The decoders used to leave their discrete-transition arguments to
``non_local_detector``'s defaults. Those defaults changed after April 2026 and
moved the non-local event count by +32 % on identical inputs. Every value now
lives here, and only here:

- :func:`nonlocal_transition_settings`: the NonLocal 4-state detector, setting
  P30-B (decision R10). It is a fixed-count Dirichlet prior whose mode (the
  M-step's zero-data estimate) gives dwell times from the literature, with a
  frozen No-Spike row whose exits follow base rates. Rationale, sources and
  validation:
  ``docs/design/2026-10-04-decoder-transition-prior.md``.
- :func:`contfrag_transition_settings`: the ContFrag 2-state classifier, at the
  values it has always decoded with, made explicit.

The NonLocal detector's states, in the library's order: 0 Local, 1 No-Spike,
2 Non-Local Continuous, 3 Non-Local Fragmented.

``non_local_detector``'s non-stationary (covariate-driven) transitions are not
used, and both decoders refuse ``--non_stationary``. Such detectors cannot be
saved (LorenFrankLab/non_local_detector#50), and their M-step adds the Dirichlet
transition prior at every time bin rather than once (#51).
"""

from typing import Any, Final

import numpy as np
from non_local_detector.discrete_state_transitions import (
    DiscreteStationaryCustom,
    DiscreteStationaryDiagonal,
)

from continuum_swr_replay.parameters import SAMPLING_FREQUENCY

BIN_S: Final[float] = 1 / SAMPLING_FREQUENCY
"""The decode's time bin (s): 2 ms at ``SAMPLING_FREQUENCY`` = 500 Hz."""

LOCAL_DWELL_S: Final[float] = 3.0
"""Expected stay in Local (s): the interval between awake events during stops.

Candidate replay events occur at 0.32-0.40 events/s while the rat is stopped
(Davidson, Kloosterman & Wilson 2009, Neuron, doi:10.1016/j.neuron.2009.07.027),
a mean interval of 2.5-3.1 s.
"""

NONLOCAL_DWELL_S: Final[float] = 0.1
"""Expected stay in each non-local state (s).

SWRs last 30-150 ms with a mode of 50 ms (Buzsáki 2015, Hippocampus,
doi:10.1002/hipo.22488; Sullivan et al. 2011, J Neurosci,
doi:10.1523/JNEUROSCI.0294-11.2011), and awake replay events last 40-1018 ms
(Davidson 2009). 100 ms covers single ripples and longer replays.
"""

NO_SPIKE_DWELL_S: Final[float] = 0.1
"""Expected stay in No-Spike (s): the post-ripple silence of about 100-150 ms.

English et al. 2014, J Neurosci, doi:10.1523/JNEUROSCI.2600-14.2014; Buzsáki
2015.
"""

PRIOR_VISITS: Final[int] = 30
"""Prior strength: 30 visits to each state.

Each learned row gets ``PRIOR_VISITS / 3`` pseudo-exits to each other state, so
the M-step adds ``sum(alpha - 1)`` = ``PRIOR_VISITS`` x its expected stay in bins
as pseudo-counts: 45,000 for Local and 1,500 for each non-local row at 500 Hz.
The Dirichlet's effective sample size, ``sum(alpha)`` (Morita, Thall & Müller
2008, Biometrics, doi:10.1111/j.1541-0420.2007.00888.x), is 4 more: 45,004 and
1,504. The design doc records the sensitivity over 10-100 visits.
"""

NO_SPIKE_EXIT_TO_LOCAL: Final[float] = 0.96
"""Share of the exits from No-Spike that go to Local.

Base rates: non-local content fills about 3.6 % of immobile time (0.36 events/s
x 0.1 s, Davidson 2009). No source reports which state follows a silence in
awake animals, so the next state follows the stationary distribution. The
remaining 0.04 is split equally between the two non-local states.
"""

NON_STATIONARY_REFUSAL: Final[str] = (
    "--non_stationary is disabled: non_local_detector's non-stationary discrete "
    "transitions cannot be saved (the fitted detector does not pickle, "
    "LorenFrankLab/non_local_detector#50) and add their Dirichlet transition prior "
    "at every time bin instead of once (#51). Decode with the stationary "
    "transitions pinned in continuum_swr_replay.decoding.transitions until both "
    "are fixed."
)
"""Why both decoders (NonLocal and ContFrag) refuse ``--non_stationary``."""


def nonlocal_transition_settings(
    sampling_frequency: float = SAMPLING_FREQUENCY,
) -> dict[str, Any]:
    """Discrete-transition keyword arguments for the NonLocal detectors (P30-B).

    Parameters
    ----------
    sampling_frequency : float, optional
        The decode's sampling frequency (Hz); one bin lasts
        ``1 / sampling_frequency`` s. Defaults to ``SAMPLING_FREQUENCY``
        (500 Hz).

    Returns
    -------
    dict
        Keyword arguments for ``NonLocalClusterlessDetector`` and
        ``NonLocalSortedSpikesDetector``. With each state's stay
        ``p_i = 1 - bin / dwell_i`` and ``e = PRIOR_VISITS / 3``:

        - ``discrete_transition_type``: ``DiscreteStationaryCustom(values=M)``,
          M of shape (4, 4): EM's starting matrix, and the fixed value of the
          frozen No-Spike row.

          - Local: ``[p_L, q_L, q_L, q_L]``, with ``q_L = (1 - p_L) / 3``.
          - No-Spike: ``[0.96 x, p_NS, 0.02 x, 0.02 x]`` with
            ``x = 1 - p_NS``, i.e. ``[0.0192, 0.98, 0.0004, 0.0004]`` at
            500 Hz: its exits split by ``NO_SPIKE_EXIT_TO_LOCAL``.
          - Non-Local Continuous: ``[q, q, p_NL, q]``; Non-Local Fragmented:
            ``[q, q, q, p_NL]``, with ``q = (1 - p_NL) / 3``.

        - ``discrete_transition_concentration``: ``1 + e``, i.e. ``e``
          pseudo-exits to each destination.
        - ``discrete_transition_stickiness``: ``e (4 p_i - 1) / (1 - p_i)`` for
          each state, shape (4,). With ``alpha = concentration +
          diag(stickiness)``, each learned row's prior mode (the M-step's
          zero-data estimate, its normalized ``alpha - 1``) is its row of M,
          and the M-step adds ``sum(alpha - 1) = PRIOR_VISITS * dwell_i / bin``
          pseudo-counts to it: 45,000 for Local and 1,500 for each non-local
          state at 500 Hz. The Dirichlet's effective sample size,
          ``sum(alpha)``, is 45,004 and 1,504. The No-Spike entry follows the
          same formula and has no effect, because that row is frozen.
        - ``discrete_transition_prior_weight``: ``0.0``, a fixed-count prior,
          not scaled to the amount of data.
        - ``frozen_discrete_transition_rows``: ``(1,)``. EM holds the No-Spike
          row at its row of M.

        Every call builds new arrays.
    """
    bin_s = 1 / sampling_frequency
    p_local = 1 - bin_s / LOCAL_DWELL_S
    p_no_spike = 1 - bin_s / NO_SPIKE_DWELL_S
    p_non_local = 1 - bin_s / NONLOCAL_DWELL_S
    q_local = (1 - p_local) / 3
    q_non_local = (1 - p_non_local) / 3
    no_spike_exit = 1 - p_no_spike
    to_each_non_local = (1 - NO_SPIKE_EXIT_TO_LOCAL) / 2
    matrix = np.array(
        [
            [p_local, q_local, q_local, q_local],
            [
                no_spike_exit * NO_SPIKE_EXIT_TO_LOCAL,
                p_no_spike,
                no_spike_exit * to_each_non_local,
                no_spike_exit * to_each_non_local,
            ],
            [q_non_local, q_non_local, p_non_local, q_non_local],
            [q_non_local, q_non_local, q_non_local, p_non_local],
        ]
    )
    stay = np.diag(matrix)
    exit_pseudo_counts = PRIOR_VISITS / 3
    return {
        "discrete_transition_type": DiscreteStationaryCustom(values=matrix),
        "discrete_transition_concentration": 1 + exit_pseudo_counts,
        "discrete_transition_stickiness": (
            exit_pseudo_counts * (4 * stay - 1) / (1 - stay)
        ),
        "discrete_transition_prior_weight": 0.0,
        "frozen_discrete_transition_rows": (1,),
    }


def contfrag_transition_settings() -> dict[str, Any]:
    """Discrete-transition keyword arguments for the ContFrag classifiers.

    These are the values the ContFrag decode has always used: the library's
    ContFrag defaults, which the decoder took before the pin (commit de15650) by
    passing ``discrete_transition_type=None``, made explicit. They are a 100 ms
    starting stay with a near-flat prior (concentration 1.1): weakly
    informative.

    Returns
    -------
    dict
        Keyword arguments for ``ContFragClusterlessClassifier`` and
        ``ContFragSortedSpikesClassifier``:

        - ``discrete_transition_type``:
          ``DiscreteStationaryDiagonal(diagonal_values=[0.98, 0.98])``, EM's
          starting matrix. A stay of 0.98 in Continuous and in Fragmented is an
          expected 100 ms at 2 ms bins (2 ms / (1 - 0.98)).
        - ``discrete_transition_concentration``: ``1.1``, i.e. 0.1
          pseudo-counts on every transition.
        - ``discrete_transition_stickiness``: ``[0.0, 0.0]``, no extra stay
          pseudo-counts.
        - ``discrete_transition_prior_weight``: ``0.0``, a fixed-count prior.
        - ``frozen_discrete_transition_rows``: ``None``; both rows are learned.

        Every call builds new arrays.
    """
    return {
        "discrete_transition_type": DiscreteStationaryDiagonal(
            diagonal_values=np.array([0.98, 0.98])
        ),
        "discrete_transition_concentration": 1.1,
        "discrete_transition_stickiness": np.array([0.0, 0.0]),
        "discrete_transition_prior_weight": 0.0,
        "frozen_discrete_transition_rows": None,
    }
