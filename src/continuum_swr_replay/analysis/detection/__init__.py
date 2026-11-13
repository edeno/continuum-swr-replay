"""The algorithms that produce event inventories from the recordings.

Each module here finds events, or prepares what a detector reads; whether the
inventories agree is ``detector_comparison``'s question, and which SWR
inventory the paper uses is ``event_definition.swr_source``'s.

- ``swr_detectors``: the four SWR detectors (Kay, Karlsson, Yu, Zugaro) and
  Carey's candidate detector on shared inputs (``run_swr_detectors``,
  ``run_carey_detector``, ``DETECTOR_SETTINGS``).
- ``ripple_tetrodes``: rule R-T, the CA1 band tetrodes kept per epoch.
- ``artifacts``: the LFP and spike artifact detector whose cores become the
  artifact mask.
- ``memoryless_sequence``: the Widloski & Foster sliding-window sequence
  detector, its shuffles, simulations and sweeps.
- ``detection_inputs``: what the detection runners read from a saved HPC
  decode and from each other.
- ``decode_masks``: the HPC decode's training and missing-data masks from the
  paper's SWR source.

Imports ``event_definition`` and the foundation; nothing here imports another
analysis package (``tests/repository/test_import_layering.py``). Submodules are
imported by name; nothing is re-exported here.
"""
