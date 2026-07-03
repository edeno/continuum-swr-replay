# fastplotlib interactive epoch viewer — how to run

**Verdict:** works on this server (headless A100s, Vulkan/wgpu device confirmed).
Full feasibility analysis + hypothesis tree: [`fastplotlib-feasibility.md`](fastplotlib-feasibility.md).

## What was installed
Into the project `.venv` (Python 3.11), **not** conda base:
```
uv pip install --python .venv/bin/python "fastplotlib[notebook]" sidecar imageio imageio-ffmpeg
.venv/bin/python -m ipykernel install --user --name continuum-swr-fpl --display-name "continuum-swr (fpl)"
```
To make it permanent, add `fastplotlib[notebook]` (+ `imageio`) to `pyproject.toml`.

## Files
- `notebooks/fastplotlib_epoch_viewer.py` — reusable module
  (`build_epoch_viewer`, `render_overview_png`, `aggregate_posterior`).
- `notebooks/fastplotlib_epoch_demo.py` — percent-format demo (open as a notebook).

## Option A — JupyterLab in a browser (recommended, most robust)
On the **remote server** (inside tmux):
```bash
cd /cumulus/edeno/continuum-swr-replay
.venv/bin/jupyter lab --no-browser --port 8888
```
On your **laptop**, forward the port (or let VSCode auto-forward it):
```bash
ssh -N -L 8888:localhost:8888 edeno@zephyr.cin.ucsf.edu
```
Open `http://localhost:8888/lab`, open `notebooks/fastplotlib_epoch_demo.py`
(right-click → Open With → Notebook), pick the **continuum-swr (fpl)** kernel, run.
The figure renders on the A100 and streams to your browser.

## Option B — VSCode native notebook over Remote-SSH (convenient, may be flaky)
Open `notebooks/fastplotlib_epoch_demo.py` as a notebook in VSCode, select the
**continuum-swr (fpl)** kernel, run. If the canvas does not appear / errors with a
`jupyter_rfb` / `RemoteFrameBufferModel` message, fall back to Option A.

## Option C — headless snapshot / movie (no display at all)
```bash
.venv/bin/python -c "from notebooks.fastplotlib_epoch_viewer import render_overview_png; \
print(render_overview_png('Processed-Data/<epoch>_HPC_results.nc', 'overview.png'))"
```
This forces the offscreen wgpu backend and writes a PNG — basis for batch movie
rendering on the GPU server.

## Verified
- wgpu enumerates 10 A100s as Vulkan adapters headless; GPU device creation OK.
- 1M-point line, per-frame update → ~75 fps (offscreen).
- Full real epoch (1.8M time bins, 4 GB `.nc`) → overview render in ~50 s (disk I/O bound).
- Numerical gotchas fixed in the module: (1) plot time **relative to epoch start**
  (absolute Unix timestamps lose precision in float32); (2) set
  `camera.maintain_aspect = False` for time-series; (3) place images via
  `world_object.local.scale_x` (top-level `.scale` silently no-ops).

## Interaction performance (remote streaming)
The ~75 fps figure was *offscreen render only*. Live interaction adds two costs:
GPU re-render of every line vertex + JPEG-encode/network of each frame. Levers,
biggest first:
1. **Decimate lines** — `max_line_points` in `build_epoch_viewer` (default 250k).
   4 states x 1.8M pts = 7.2M vertices/frame otherwise. Lower to 100k if still slow.
   Uses min/max-envelope decimation, so probability spikes are preserved.
2. **`tune_remote_performance(fig, quality=75, max_buffered_frames=1)`** — call after
   `fig.show()`. Lower `quality` (smaller frames) and `max_buffered_frames=1` (less lag).
3. **Smaller figure `size`** — fewer pixels to encode/stream.
4. **JupyterLab over VSCode** — its jupyter_rfb path is generally more efficient than
   VSCode's native notebook widget host.

## Navigation & axis locking

- **Left-drag** = scroll sideways through time. **Scroll-wheel** = zoom the time axis
  (toward the cursor). **Right-drag** = zoom box.
- A single `PanZoomController` drives both panels with `include_state={"x","width"}`,
  so navigation is **time-axis only**: panels stay time-locked and the y axes
  (probability 0-1, position) stay fixed — amplitudes never distort while you zoom/scroll.
- `build_epoch_viewer(..., link_x=False)` restores independent full pan/zoom per panel
  (incl. y) if you want to zoom amplitude.

## Not yet done (future work)
- Live verification of Options A/B (needs a browser session).
- True x-linked controllers across the two panels (currently zoom independently).
- Full-resolution posterior in a zoom window (overview is time-aggregated).
- Add LFP / ripple / position / spike-raster panels for a full epoch scrubber.
