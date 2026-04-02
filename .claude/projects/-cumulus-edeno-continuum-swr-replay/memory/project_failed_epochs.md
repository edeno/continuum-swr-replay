---
name: Expected failed decoding epochs
description: Three senor epochs that fail HPC decoding due to missing Spyglass pipeline entries — expected and documented in constants.py
type: project
---

Three epochs fail during HPC decoding. These failures are expected and documented in `src/continuum_swr_replay/data_loaders/constants.py`:

1. **senor20201104_06_r3** — missing LFPBandV1 entry (no ripple detection data)
2. **senor20201104_08_r4** — missing LFPBandV1 entry (no ripple detection data)
3. **senor20201105_09_r4** — missing TrackGraph entry (no linearized position data)

**Why:** Upstream Spyglass pipeline entries were never populated for these epochs.

**How to apply:** 218/221 epochs succeed. These 3 can be excluded or skipped gracefully in batch processing.
