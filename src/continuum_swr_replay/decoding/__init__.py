"""What a decode is configured with, the files it writes, and how they are read.

A foundation package: it imports the standard library, numpy, xarray,
``non_local_detector`` and the other foundation modules (``parameters``,
``data_types``), and nothing from ``analysis``. The HPC decode's masks read the SWR
source, so they live above it, in ``analysis.event_definition.decode_masks``.

- ``config``: the encoding-likelihood configs (``DecodingConfig``,
  ``NonLocalDecodingConfig``) and the detector class that decodes each brain
  area (``nonlocal_detector_classes``, ``contfrag_classifier_classes``).
- ``transitions``: the decoders' discrete-transition settings.
- ``identifiers``: the epoch identifier, built (``format_epoch_identifier``,
  ``decode_identifier``) and read back (``parse_epoch_identifier``), with the
  defaults no identifier spells out (Kay, ``no_ripple``).
- ``files``: the one naming rule for a decode's files, and whether a decode's
  files exist.
- ``results``: the one ``results.nc`` reader, and an epoch's saved decode
  read back whole: its results, fitted detector and environment
  (``load_results``).
"""
