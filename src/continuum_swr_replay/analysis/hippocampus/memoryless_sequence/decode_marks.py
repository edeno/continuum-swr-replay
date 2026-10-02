"""The HPC marks the saved decodes were fitted on, rebuilt from the raw marks.

The canonical decodes were made on 2026-04-01 at commit 0820d39d. Since then
``detect_coincident_spikes`` was corrected (64bf585, 2026-05-05) to count the
leading spike of each close cluster toward its coincident fraction, so the
current loader removes more spikes than the decode did (259 of about 358,000 on
``chimi20200225_02_r1_HPC``, every tetrode losing a few). The sliding-window
detector reads the decode's encoding model, so it must read the decode's spikes
too: this module reproduces the decode-time filtering, and only this detector's
runner uses it. ``load_data`` and every other analysis keep the corrected rule.

Decode-time filtering, in order:

1. every ``UnitMarks`` sort group of the epoch (the tetrode-only filter,
   f172084, also came later);
2. coincident-spike removal under the 0820d39d rule
   (:func:`decode_time_coincident_keep`);
3. spike times restricted to the decode's time span, inclusive at both ends,
   and electrodes left with no spikes dropped (:func:`decode_time_marks`).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label

DECODE_COMMIT = "0820d39d"
"""The commit the canonical decodes were made at."""

COINCIDENT_RULE = "0820d39d: leading spike removed with its cluster, not counted"
"""How the decode-time rule differs from the corrected one, for the manifest."""


def decode_time_coincident_keep(
    spike_times: list[NDArray[np.float64]],
    spike_closeness_threshold: float = 0.00004,
    max_coincident_fraction: float = 0.33,
) -> list[NDArray[np.intp]]:
    """Indices of the spikes the decode-time coincident filter kept.

    Spikes from all electrodes are pooled and sorted; a run of consecutive gaps
    shorter than ``spike_closeness_threshold`` is a cluster. As at 0820d39d,
    ``scipy.ndimage.label`` leaves each cluster's first spike unlabeled, so the
    fraction of electrodes in a cluster counts only its labeled spikes, while a
    cluster over ``max_coincident_fraction`` loses its first spike as well.

    Parameters
    ----------
    spike_times : list of ndarray, each shape (n_spikes,)
        Spike times in seconds, one array per electrode.
    spike_closeness_threshold : float
        Largest gap, in seconds, inside a cluster.
    max_coincident_fraction : float
        A cluster on more than this fraction of electrodes is removed.

    Returns
    -------
    list of ndarray of int, one per electrode
        Kept indices into that electrode's spikes, in the pooled sort's order,
        which is time order for sorted input.
    """
    n_electrodes = len(spike_times)
    times = np.concatenate([np.asarray(t, dtype=np.float64) for t in spike_times])
    electrode = np.repeat(np.arange(n_electrodes), [len(t) for t in spike_times])
    index = np.concatenate([np.arange(len(t)) for t in spike_times]).astype(np.intp)
    if times.size == 0:
        return [np.array([], dtype=np.intp) for _ in range(n_electrodes)]

    # the same default sort over the same concatenation as 0820d39d, so tied
    # times open their clusters with the same spike
    order = np.argsort(times)
    electrode, index = electrode[order], index[order]
    is_close = np.concatenate(
        [[False], np.diff(times[order]) < spike_closeness_threshold]
    )
    labels, n_labels = label(is_close)

    is_artifact_label = np.zeros(n_labels + 1, dtype=bool)
    labeled = labels > 0
    if labeled.any():
        pairs = np.unique(np.stack([labels[labeled], electrode[labeled]]), axis=1)
        n_per_label = np.bincount(pairs[0], minlength=n_labels + 1)
        is_artifact_label = n_per_label / n_electrodes > max_coincident_fraction
        is_artifact_label[0] = False
    is_artifact = is_artifact_label[labels]
    # the unlabeled spike that opens each artifact cluster goes with it
    is_artifact[:-1] |= is_artifact[1:]

    keep = ~is_artifact
    return [index[keep & (electrode == e)] for e in range(n_electrodes)]


def decode_time_marks(
    spike_times: list[NDArray[np.float64]],
    spike_waveform_features: list[NDArray[np.float64]],
    span_s: tuple[float, float],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """The decode's spike times and marks from an epoch's raw ``UnitMarks``.

    Parameters
    ----------
    spike_times : list of ndarray, each shape (n_spikes,)
        Raw spike times in seconds, one array per sort group.
    spike_waveform_features : list of ndarray, each shape (n_spikes, n_features)
        The matching raw marks.
    span_s : tuple of float
        First and last time of the decode's grid, in seconds; spikes on either
        end are kept.

    Returns
    -------
    spike_times, spike_waveform_features : list of ndarray
        Coincident spikes removed, restricted to ``span_s``, and electrodes with
        no spikes left dropped, as the decode received them.
    """
    start, end = span_s
    kept = decode_time_coincident_keep(spike_times)
    out_times, out_features = [], []
    for times, features, keep in zip(
        spike_times, spike_waveform_features, kept, strict=True
    ):
        times, features = np.asarray(times)[keep], np.asarray(features)[keep]
        in_span = (times >= start) & (times <= end)
        if not in_span.any():
            continue
        out_times.append(times[in_span])
        out_features.append(features[in_span])
    return out_times, out_features
