"""What a decode is configured with, the files it writes, and how they are read.

A foundation package: it imports the standard library, numpy, xarray,
``non_local_detector`` and the other foundation modules (``parameters``,
``data_types``), and nothing from ``analysis``. The HPC decode's masks read the SWR
source, so they live above it, in ``analysis.hippocampus.decode_masks``.

- ``config``: the encoding-likelihood configs (``DecodingConfig``,
  ``NonLocalDecodingConfig``).
- ``transitions``: the decoders' discrete-transition settings.
- ``files``: the one naming rule for a decode's files, and whether a decode's
  files exist.
- ``results``: the one ``results.nc`` reader.
"""
