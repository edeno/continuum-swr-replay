"""The run infrastructure every runner shares.

A foundation package: it imports the standard library, numpy, pandas and the
other foundation modules (``tables``, ``data_loaders``, ``parameters``), and
nothing from ``analysis``. Importing it, or any of its modules, loads no dask:
``batch`` imports dask inside the functions that start a cluster.

- ``io``: strict JSON, content hashes, an epoch's output set (atomic writes
  are ``tables.io.write_atomically``).
- ``provenance``: the git checkout and environment a run ran from, and the
  commit label and run id that name it.
- ``fit_provenance``: the sidecar beside a fitted artifact that records the
  input tables it was fit on, and the readers' check that they have not changed.
- ``staleness``: whether a measurement or a PFC bundle was made from the current
  inputs: a manifest's recorded digest and the count of records that differ.
- ``cli``: run ids, the decoders' shared options, the epoch filters.
- ``batch``: the decoders' batch loop, the per-epoch task loop, a run's
  manifest kept across a resume (``RunManifest``), per-epoch logs, the NWB file
  release and the run folder's lock.
- ``device``: where a runner's JAX computes (``default_jax_platforms``), the
  XLA flags of a deterministic GPU run (``xla_flags``) and one claimed card of
  a shared GPU host (``configure_jax_gpu``); it imports JAX only when called.
- ``threads``: pins the BLAS thread pools to one thread; it imports only the
  standard library, so a runner calls it above its numpy import.
"""
