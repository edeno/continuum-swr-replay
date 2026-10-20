"""LFP and spike artifact detection for CA1 epochs.

- ``features``: the robust-unit features (S window spectra, J5 / L20 steps,
  spike lines and out-of-distribution marks).
- ``detect``: the constants and the four rules (S, T-ref, T-ca1, T-spk);
  ``detect_artifacts`` returns an epoch's unpadded cores with their evidence.

The padding table (``PADDING_S``) and the artifact mask live in
``analysis/events/artifact_masks.py``, and the per-epoch cores file and the
run manifest in ``analysis/events/artifact_intervals.py``; ``detect`` writes
through the latter.

``scripts/decoding/run_artifact_detection.py`` runs it per epoch;
``analysis/events/swr_source.py`` reads the artifact mask. Design note:
``docs/design/2026-10-09-artifact-detection.md``. The submodules are imported
by name, so reading a mask does not import the detector.
"""
