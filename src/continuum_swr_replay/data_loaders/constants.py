"""Constants for the bandit task data loading.

This module contains all constant values used across the data loading pipeline,
including NWB file lists, spatial mappings, and task structure definitions.

Attributes
----------
NWB_FILES : list of str
    The 91 NWB files ``get_epoch_info`` searches, from 5 animals: chimi, j16,
    peanut, senor, and wilbur. Which of their epochs are analysed is the epoch
    register's to say (``data_loaders/epoch_register.py``).
REGULAR_SESSION_DESCRIPTION : str
    The ``Session.session_description`` of a regular spatial-bandit session.
FIRST_EXPOSURE_DESCRIPTIONS : dict of {str: str}
    First-exposure sessions ``get_epoch_info`` admits under another description.
WELL_PATCH_MAPPING : dict of {int: int}
    Maps well node IDs (0-5) to patch IDs (1-3).
    Wells 0-1 are in patch 1, wells 2-3 in patch 2, wells 4-5 in patch 3.
TRACK_SEGMENT_TO_PATCH : dict of {int: int}
    Maps track segment IDs to patch IDs (1-3).
OMISSION_KNOWN_LATENCY_SEC : float
    Seconds after destination poke-in by which an omission is established.
BRAIN_AREAS : tuple of str
    The project's brain-area names; capitalization matters throughout.

Notes
-----
The spatial bandit task uses a 6-well environment organized into 3 patches.
Each patch contains 2 wells, and the animal navigates between wells along
track segments (edges in the track graph).

See Also
--------
get_epoch_info : Get information about all available epochs
load_data : Main data loading function that uses these constants

Examples
--------
>>> from continuum_swr_replay.data_loaders.constants import WELL_PATCH_MAPPING
>>> WELL_PATCH_MAPPING[0]  # Well 0 is in patch 1
1
>>> WELL_PATCH_MAPPING[4]  # Well 4 is in patch 3
3

"""

NWB_FILES = [
    "chimi20200212_.nwb",
    "chimi20200213_.nwb",
    "chimi20200214_.nwb",
    "chimi20200216_.nwb",
    "chimi20200217_.nwb",
    "chimi20200218_.nwb",
    "chimi20200220_.nwb",
    "chimi20200221_.nwb",
    "chimi20200222_.nwb",
    "chimi20200223_.nwb",
    "chimi20200224_.nwb",
    "chimi20200225_.nwb",
    "chimi20200227_.nwb",
    "chimi20200228_.nwb",
    "chimi20200304_.nwb",
    "chimi20200305_.nwb",
    "chimi20200306_.nwb",
    "chimi20200308_.nwb",
    "chimi20200310_.nwb",
    "chimi20200311_.nwb",
    "chimi20200312_.nwb",
    "chimi20200313_.nwb",
    "j1620210706_.nwb",
    "j1620210707_.nwb",
    "j1620210708_.nwb",
    "j1620210709_.nwb",
    "j1620210710_.nwb",
    "j1620210711_.nwb",
    "j1620210712_.nwb",
    "j1620210713_.nwb",
    "j1620210714_.nwb",
    "j1620210715_.nwb",
    "j1620210716_.nwb",
    "j1620210717_.nwb",
    "j1620210718_.nwb",
    "j1620210719_.nwb",
    "j1620210720_.nwb",
    "j1620210721_.nwb",
    "peanut20201125_.nwb",
    "peanut20201127_.nwb",
    "peanut20201128_.nwb",
    "peanut20201129_.nwb",
    "peanut20201201_.nwb",
    "peanut20201202_.nwb",
    "peanut20201203_.nwb",
    "peanut20201204_.nwb",
    "peanut20201205_.nwb",
    "peanut20201206_.nwb",
    "peanut20201208_.nwb",
    "peanut20201209_.nwb",
    "senor20201027_.nwb",
    "senor20201028_.nwb",
    "senor20201030_.nwb",
    "senor20201101_.nwb",
    "senor20201102_.nwb",
    "senor20201103_.nwb",
    "senor20201104_.nwb",
    "senor20201105_.nwb",
    "senor20201106_.nwb",
    "senor20201107_.nwb",
    "senor20201109_.nwb",
    "senor20201110_.nwb",
    "senor20201111_.nwb",
    "senor20201112_.nwb",
    "senor20201113_.nwb",
    "senor20201114_.nwb",
    "senor20201115_.nwb",
    "senor20201116_.nwb",
    "senor20201117_.nwb",
    "senor20201118_.nwb",
    "senor20201120_.nwb",
    "senor20201121_.nwb",
    "wilbur20210326_.nwb",
    "wilbur20210328_.nwb",
    "wilbur20210329_.nwb",
    "wilbur20210330_.nwb",
    "wilbur20210331_.nwb",
    "wilbur20210402_.nwb",
    "wilbur20210403_.nwb",
    "wilbur20210404_.nwb",
    "wilbur20210405_.nwb",
    "wilbur20210406_.nwb",
    "wilbur20210407_.nwb",
    "wilbur20210408_.nwb",
    "wilbur20210409_.nwb",
    "wilbur20210410_.nwb",
    "wilbur20210411_.nwb",
    "wilbur20210412_.nwb",
    "wilbur20210413_.nwb",
    "wilbur20210414_.nwb",
    "wilbur20210415_.nwb",
]
"""The 91 NWB files ``get_epoch_info`` searches, from 5 animals.

Animals and session counts:
- chimi: 22 sessions (Feb-Mar 2020)
- j16: 16 sessions (Jul 2021)
- peanut: 12 sessions (Nov-Dec 2020)
- senor: 22 sessions (Oct-Nov 2020)
- wilbur: 19 sessions (Mar-Apr 2021)

Not every file is analysed. The depletion sessions ("Spatial bandit task
(depletion)", a decaying reward schedule) are a different task variant that
``get_epoch_info`` leaves out by their description; one first-exposure
session, wilbur20210326, is admitted through ``FIRST_EXPOSURE_DESCRIPTIONS``.
Which run epochs are analysed, and why every other run epoch of these files
and of the sessions left out of this list is not, is recorded in one place,
the epoch register (``data_loaders/epoch_register.py``,
``epoch_register.csv``), which every stage reads. Its counts and its excluded
and pending entries, with their evidence, are generated into
``docs/epoch_register.md`` by ``scripts/checks/write_epoch_register_doc.py``; they
are not repeated here, so they cannot go stale.
"""

REGULAR_SESSION_DESCRIPTION = "Spatial bandit task (regular)"
"""``Session.session_description`` of a regular spatial-bandit session."""

FIRST_EXPOSURE_DESCRIPTIONS: dict[str, str] = {
    "wilbur20210326_.nwb": "Spatial bandit task (regular) - exploration session",
}
"""First-exposure sessions ``get_epoch_info`` admits under their own description.

Four epochs are an animal's first exposure to the task (200 trials, p(R) 100 %
then 50 %): j1620210706_02_r1, senor20201027_02_r1, chimi20200212_02_r1 and
wilbur20210326_03_r1. The project analyses all four. Three sessions carry the
regular description; wilbur20210326's is "Spatial bandit task (regular) -
exploration session", so it is admitted here, by name and description. Nothing
else is.
"""

WELL_PATCH_MAPPING = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 3,
    5: 3,
}
"""Mapping from well node IDs to patch IDs.

The 6-well environment is organized into 3 patches:
- Patch 1: Wells 0, 1
- Patch 2: Wells 2, 3
- Patch 3: Wells 4, 5
"""

WELL_ARMS: tuple[int, ...] = tuple(sorted(WELL_PATCH_MAPPING))
"""Leaf arm IDs that terminate at the six reward wells, in canonical order."""

POKE_EVENT_TO_WELL: dict[str, int] = {f"Poke{well + 1}": well for well in WELL_ARMS}
"""Map the six physical beam-break DIO channels to zero-based reward wells.

The DIO channel identifies which port was poked. Tracked position is a useful
quality-control measurement, but it is not authoritative at a poke boundary:
the position estimate can move onto an adjoining segment while the beam remains
broken. Keeping this mapping explicit prevents that tracking noise from
creating or deleting behavioral trials.
"""

PUMP_EVENT_TO_WELL: dict[str, int] = {f"Pump{well + 1}": well for well in WELL_ARMS}
"""Map reward-pump DIO channels to their zero-based reward wells.

Reward assignment is valid only when the pump channel matches the photogate
channel containing the pump onset.  Keeping this map separate from the poke
map makes that acquisition invariant explicit and testable.
"""

TRACK_SEGMENT_TO_PATCH = {
    0: 1,
    1: 1,
    6: 1,
    2: 2,
    3: 2,
    7: 2,
    4: 3,
    5: 3,
    8: 3,
}
"""Mapping from track segment IDs to patch IDs.

Track segments are edges in the graph connecting wells. Each segment
is associated with one of the 3 patches based on its location.
Segments 0, 1, 6 are in patch 1; segments 2, 3, 7 in patch 2;
segments 4, 5, 8 in patch 3.
"""

ALL_ARMS: tuple[int, ...] = tuple(sorted(TRACK_SEGMENT_TO_PATCH.keys()))
"""Every arm (track segment) id, ascending: six leaf arms then three trunk arms.

The canonical arm axis. Per-arm tables carry one row per entry in this
order, and per-arm matrices are indexed by position in it, so analysis and
visualization must share a single definition — reordering it silently
reinterprets every per-arm coefficient.
"""

LEAF_ARMS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
"""The leaf arms, ascending: each joins one reward well to its patch junction.

Arm ``i`` ends at well ``i``, so these are also the well (port) ids.
"""

TRUNK_ARMS: tuple[int, ...] = (6, 7, 8)
"""The trunk arms, ascending: each joins a patch junction to the centre."""

N_LEAF_ARMS: int = len(LEAF_ARMS)
"""The number of leaf arms, six."""

N_PORTS: int = len(WELL_ARMS)
"""The number of reward wells (ports), six."""


def port_columns(prefix: str) -> list[str]:
    """The per-port column names ``{prefix}_port_0`` .. ``{prefix}_port_5``.

    Parameters
    ----------
    prefix : str
        Such as ``"reward_belief_mean"``.

    Returns
    -------
    list[str]
        One name per port, in port order (length :data:`N_PORTS`).

    Examples
    --------
    >>> port_columns("reward_belief_mean")[:2]
    ['reward_belief_mean_port_0', 'reward_belief_mean_port_1']
    """
    return [f"{prefix}_port_{i}" for i in range(N_PORTS)]


OMISSION_KNOWN_LATENCY_SEC: float = 0.30
"""Seconds after destination poke-in by which an omission is established.

The rewarded boundary is a stimulus: in the refreshed 218-session behavioral
cache, 18,753 rewarded task visits had p99 pump latency 91 ms and maximum
latency 236 ms after destination arrival. Omissions have no negative-feedback
cue to timestamp, and the previous rule used destination poke-out -- a median
880 ms later, and a *behavioural* boundary rather than a stimulus one. Gating
on when the animal chose to leave is circular (dwell length plausibly covaries
with replay) and it excluded 99% of omission events from every outcome-gated
fit, collapsing the reward contrast those models exist to estimate.

0.30 s exceeds every reward latency observed in that corpus, so it cannot
declare an omission while a pump could still fire, and it retains 99.6% of
omission pokes with a median 0.58 s of usable dwell. Re-derive from the
reward-latency distribution if the hardware or task timing changes.

It lives with the loading constants, not the analysis ones, because the trial
builders on the loading side define the outcome clock with it and must not have
to import from ``analysis`` to do so.
"""

BRAIN_AREAS: tuple[str, ...] = ("HPC", "mPFC", "OFC")
