"""The run infrastructure every runner shares.

A foundation package: it imports the standard library, numpy, pandas and the
other foundation modules (``tables``, ``data_loaders``, ``parameters``), and
nothing from ``analysis``. Importing it, or any of its modules, loads no dask:
``batch`` imports dask inside the functions that start a cluster.

- ``io``: atomic writes, strict JSON, content hashes, an epoch's output set.
- ``provenance``: the git checkout and environment a run ran from.
- ``cli``: run ids, the decoders' shared options, the epoch filters.
- ``batch``: the decoders' batch loop, the per-epoch task loop, per-epoch
  logs, the NWB file release and the run folder's lock.
"""
