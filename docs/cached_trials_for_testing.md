# Loading cached trials + track_graph for testing (TEMPORARY)

> **This is a temporary workaround while Spyglass / NWB access is
> unavailable on this host.** The NWB files in `/stelmo/nwb/` were
> accidentally deleted and are being restored; while they are missing,
> `load_data()` can't fetch trials or the track graph from Spyglass.
>
> Once Spyglass is back, go straight to `load_data(nwb_file_name,
> epoch_name)` and delete this document.

## Why this exists

`scripts/prepare_replay_content_data.py` and any analysis that wants
real trial context need per-session `trials` and `track_graph`.
Normally both come from:

```python
from continuum_swr_replay.data_loaders import load_data
data = load_data(nwb_file_name="j1620210710_.nwb", epoch_name="02_r1")
trials = data["trials"]
track_graph = data["track_graph"]
```

Without Spyglass, those lines raise `DataJointError` or
`FileNotFoundError` on the missing NWB.

## Available cached sessions

The state-space-playground repo has pickled the full `load_data()`
output for seven j16 sessions under
`/cumulus/edeno/state-space-playground/cache/sessions/`:

- `j1620210710_02_r1_sortedHPC.pkl`
- `j1620210710_04_r2_sortedHPC.pkl`
- `j1620210710_06_r3_sortedHPC.pkl`
- `j1620210710_08_r4_sortedHPC.pkl`
- `j1620210710_10_r5_sortedHPC.pkl`
- `j1620210710_12_r6_sortedHPC.pkl`
- `j1620210710_14_r7_sortedHPC.pkl`

All have the matching `_HPC_80_data_over_time.pkl`, `_HPC_environment.pkl`,
and `_HPC_results.nc` already under `Processed-Data/`.

## Catch: pandas version skew

The playground venv is on pandas 3.0.2, which saves DataFrames with the
new `StringDtype(storage='python', na_value=nan)` labels. This repo's
venv is on pandas 2.3.3 and chokes on those labels with
`NotImplementedError: (<StringDtype(...)>, array([...]))` on
`pickle.load`.

Workaround: extract `trials` and `track_graph` in the playground venv,
re-pickle them as plain numpy dicts (no pandas dtypes involved in the
labels), then read back from this repo.

### Step 1 — one-shot conversion (run in playground venv)

```bash
cd /cumulus/edeno/state-space-playground
VIRTUAL_ENV= uv run --no-sync python - <<'PY'
import pickle
from pathlib import Path

src = Path("/cumulus/edeno/state-space-playground/cache/sessions")
dst = Path("/tmp/continuum_test_cache")
dst.mkdir(exist_ok=True)

for pkl in sorted(src.glob("*_sortedHPC.pkl")):
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    trials = data["trials"]
    track_graph = data["track_graph"]
    stem = pkl.stem.replace("_sortedHPC", "")
    # Dict-of-arrays bypasses pandas 3.0's StringDtype in the column
    # index — that's what pandas 2.x can't deserialize.
    cols = [str(c) for c in trials.columns]
    payload = {"columns": cols, **{c: trials[c].to_numpy() for c in cols}}
    with open(dst / f"{stem}_trials.pkl", "wb") as f:
        pickle.dump(payload, f, protocol=4)
    with open(dst / f"{stem}_track_graph.pkl", "wb") as f:
        pickle.dump(track_graph, f, protocol=4)
    print(f"{stem}: trials {trials.shape}, graph "
          f"{track_graph.number_of_nodes()}n/{track_graph.number_of_edges()}e")
PY
```

`VIRTUAL_ENV=` clears whichever venv your shell is in so `uv run` picks
up the playground's own `.venv`.

### Step 2 — load from this repo

```python
import pickle
import pandas as pd

session_id = "j1620210710_02_r1"
with open(f"/tmp/continuum_test_cache/{session_id}_trials.pkl", "rb") as f:
    d = pickle.load(f)
trials = pd.DataFrame({c: d[c] for c in d["columns"]})

with open(f"/tmp/continuum_test_cache/{session_id}_track_graph.pkl", "rb") as f:
    track_graph = pickle.load(f)
```

From here use `trials` / `track_graph` as if they came from
`load_data()`. The pipeline has been validated end-to-end on
`j1620210710_02_r1` with this path — 299 events, 591 stops, 2691
event-arm rows, 13773 sequence rows, no NaN in `current_arm` /
`prev_arm` / `next_arm`.

## Cleanup

Once Spyglass access is restored:

1. Replace the two `open()` calls above with
   `data = load_data(nwb_file_name, epoch_name)` + indexing.
2. Delete `/tmp/continuum_test_cache/`.
3. Delete this document.
