"""The canonical tables' schema, row ids, on-disk format and parity checks.

A foundation package: its modules import numpy, pandas, pyarrow, each other
and the other foundation modules (``paths``, ``data_loaders``), and nothing
from ``analysis``, so any tool that needs a row key, the covariate registry or
a table read loads none of the analysis code.

- ``io``: the on-disk format (parquet with ``attrs``) and ``publish_canonical``,
  the one write door for a canonical table.
- ``covariate_registry``: the column vocabulary and registered row keys.
- ``versions``: the covariate-schema, behavioral-cache and manifest versions.
- ``session_keys``: the session key (animal, session_date, epoch).
- ``event_ids``: the content-hashed event and stop ids.
- ``standardization``: the z-scoring reference of the standardized columns.
- ``parity``: keyed table-to-table parity checks.
"""
