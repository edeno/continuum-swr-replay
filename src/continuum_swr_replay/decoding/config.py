"""The decoders' encoding-likelihood configuration and detector classes.

``run_decoding_bandit.py`` (the NonLocal 4-state detector) decodes with
:class:`NonLocalDecodingConfig` and ``run_decoding_contfrag.py`` (the ContFrag
2-state classifier) with :class:`DecodingConfig`. Their values are
``parameters``'. Which class decodes each brain area is
:func:`nonlocal_detector_classes` and :func:`contfrag_classifier_classes`.

``non_local_detector`` (JAX) is imported only inside those two functions, so
importing this module loads no JAX.
"""

from dataclasses import dataclass

from continuum_swr_replay.data_types import ClusterlessParams, SortedSpikesParams
from continuum_swr_replay.parameters import (
    BLOCK_SIZE,
    LOCAL_POSITION_STD,
    POSITION_STD,
    WAVEFORM_STD,
)


@dataclass(frozen=True)
class DecodingConfig:
    """Encoding-likelihood configuration shared by every detector family.

    ``position_std``, ``waveform_std``, and ``block_size`` configure the
    encoding-model likelihood (the ``*_algorithm_params`` dict) and apply
    to both the NonLocal 4-state detector and the ContFrag 2-state
    classifier. State-specific detector arguments live on subclasses —
    see :class:`NonLocalDecodingConfig`.
    """

    position_std: float = POSITION_STD
    block_size: int = BLOCK_SIZE
    waveform_std: float = WAVEFORM_STD

    def to_clusterless_params(self) -> ClusterlessParams:
        """Get parameters for clusterless decoding (HPC)."""
        return {
            "position_std": self.position_std,
            "waveform_std": self.waveform_std,
            "block_size": self.block_size,
        }

    def to_sorted_spikes_params(self) -> SortedSpikesParams:
        """Get parameters for sorted spikes decoding (mPFC/OFC)."""
        return {
            "position_std": self.position_std,
            "block_size": self.block_size,
        }


@dataclass(frozen=True)
class NonLocalDecodingConfig(DecodingConfig):
    """Decoding config for the NonLocal 4-state detector.

    Adds ``local_position_std``, a detector-level argument that controls
    the Local state's spatial representation. It is passed directly to the
    detector constructor (not through ``to_*_params``). The ContFrag
    2-state classifier has no Local state, so it uses the base
    :class:`DecodingConfig` instead of this subclass.
    """

    local_position_std: float = LOCAL_POSITION_STD


def nonlocal_detector_classes() -> dict[str, tuple[type, str]]:
    """Brain area -> (NonLocal 4-state detector class, its algorithm-params type).

    HPC decodes clusterless (spike waveform features), mPFC and OFC sorted
    spikes. The params type names the detector's ``*_algorithm_params``
    argument (``"clusterless"`` or ``"sorted_spikes"``).

    Returns
    -------
    dict of str to (type, str)
        ``{"HPC": (NonLocalClusterlessDetector, "clusterless"),
        "mPFC": (NonLocalSortedSpikesDetector, "sorted_spikes"),
        "OFC": (NonLocalSortedSpikesDetector, "sorted_spikes")}``.
    """
    from non_local_detector import (
        NonLocalClusterlessDetector,
        NonLocalSortedSpikesDetector,
    )

    return {
        "HPC": (NonLocalClusterlessDetector, "clusterless"),
        "mPFC": (NonLocalSortedSpikesDetector, "sorted_spikes"),
        "OFC": (NonLocalSortedSpikesDetector, "sorted_spikes"),
    }


def contfrag_classifier_classes() -> dict[str, tuple[type, str]]:
    """Brain area -> (ContFrag 2-state classifier class, its algorithm-params type).

    The 2-state model is [Continuous (RandomWalk), Fragmented (Uniform)]. As in
    :func:`nonlocal_detector_classes`, HPC is clusterless and mPFC and OFC are
    sorted spikes.

    Returns
    -------
    dict of str to (type, str)
        ``{"HPC": (ContFragClusterlessClassifier, "clusterless"),
        "mPFC": (ContFragSortedSpikesClassifier, "sorted_spikes"),
        "OFC": (ContFragSortedSpikesClassifier, "sorted_spikes")}``.
    """
    from non_local_detector import (
        ContFragClusterlessClassifier,
        ContFragSortedSpikesClassifier,
    )

    return {
        "HPC": (ContFragClusterlessClassifier, "clusterless"),
        "mPFC": (ContFragSortedSpikesClassifier, "sorted_spikes"),
        "OFC": (ContFragSortedSpikesClassifier, "sorted_spikes"),
    }
