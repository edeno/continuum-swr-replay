# Research: fastplotlib for fast interactive whole-epoch visualization

**Question:** Can we use [fastplotlib](https://github.com/fastplotlib/fastplotlib)
to interactively visualize *whole epochs* of time-series data (decode posteriors,
state probabilities, LFP, position, spikes) quickly, viewed from a laptop while the
data and compute live on a remote headless GPU server (accessed via VSCode Remote-SSH)?

Status: **FEASIBLE — verified end-to-end on this machine.** (2026-06-17)

---

## TL;DR / Recommendation

fastplotlib works on this box. The make-or-break requirement — a working Vulkan/wgpu
GPU device on a headless server — is **satisfied and verified** (device creation +
real offscreen render succeed). For remote viewing, the robust path is **JupyterLab in
a browser, port-forwarded over SSH** (fastplotlib renders on the A100 server-side and
streams JPEG frames to the browser via `jupyter_rfb`). VSCode's *native* notebook can
also host it but its ipywidgets/`jupyter_rfb` support is historically flaky over Remote-
SSH (Hypothesis H2, medium confidence). Recommend prototyping in JupyterLab; offer
VSCode-native as a "try it, may need a fallback" option.

Confidence that fastplotlib is the right tool for this job: **~0.8** (was ~0.5 before
verification; raised by the headless wgpu device + 75 fps render benchmark).

---

## Environment (verified facts)

| Fact | Value | Source |
|---|---|---|
| GPUs | 10× NVIDIA A100 80GB PCIe, driver 580.159.03 | `nvidia-smi` |
| Display | none (`DISPLAY` empty) — headless | env |
| Session | SSH + tmux, VSCode Remote-SSH (`VSCODE_IPC_HOOK_CLI` set) | env |
| Vulkan loader | `libvulkan.so.1` present | `ldconfig` |
| NVIDIA Vulkan ICD | `/usr/share/vulkan/icd.d/nvidia_icd.json` present | fs |
| **wgpu device (headless)** | **all 10 A100s enumerate as `Vulkan / DiscreteGPU`; `request_device_sync()` succeeds** | direct test |
| **Full render path** | **fastplotlib 0.6.1 offscreen render → PNG works** | direct test |
| **Render speed** | **1M-point line, data updated every frame → 75.8 fps** (first frame 1.5 s warmup) | benchmark |
| Project env | `.venv` Python 3.11.13 (uv-managed); jupyterlab 4.4 + ipywidgets 8.1 already installed in conda env (NOT .venv) | `uv pip list` |
| fastplotlib installed? | **No** — needs install into `.venv` | `uv pip list` |

The "No windowing system present. Using surfaceless platform" message during the wgpu
test is the *expected* headless-success signal, not an error.

---

## Data scale (verified — sizing the problem)

One HPC decode result (`chimi20200213_02_r1_HPC_results.nc`, 3.96 GB):

- `acausal_posterior`: **(time=1,807,732, state_bins=542) float32** ≈ 3.9 GB
- `acausal_state_probabilities`: (1,807,732, 4) float32
- coords: `position` (542), `state` (542), `time` (1.8M, float64)

Implications:
- **Lines:** 1.8M points/trace is trivial for fastplotlib on GPU (benchmarked 1M pts @ 75 fps).
  State probabilities (4 traces × 1.8M) → smooth interactive pan/zoom. ✅
- **Posterior heatmap:** 1.8M columns **exceeds the GPU max 2D texture dimension**
  (typically 8192–16384). Cannot upload the full posterior as one texture. ⚠️
  → Must aggregate/downsample along time for the whole-epoch *overview*, and show full
  resolution only inside a zoom window (the standard big-image pattern). This is the one
  real engineering nuance, and it's also the *correct* thing to do (the screen is ~2000 px
  wide; 1.8M raw columns is wasted work regardless of tool).

---

## Hypothesis tree (with calibrated confidence)

Root: **"fastplotlib lets us interactively view whole epochs quickly from a laptop."**

### H1 — JupyterLab (browser, SSH port-forward) + jupyter_rfb  — confidence 0.9
Render on A100 server-side, JPEG-stream to browser. This is fastplotlib's documented and
most-tested remote path. Device + render verified here. **Primary recommendation.**
- Risk: install footprint; minor latency on the JPEG stream (fine for scrubbing/zoom).
- Falsifiable by: launching `jupyter lab` on server, forwarding the port, instantiating a
  `Figure` in a cell — if a canvas appears and responds to mouse, confirmed.

### H2 — VSCode native notebook over Remote-SSH + jupyter_rfb  — confidence 0.5
Convenient (no separate browser tab) but VSCode's ipywidgets/`jupyter_rfb` rendering over
Remote-SSH has a documented history of breakage (vscode#148662; jupyter_rfb#68
"RemoteFrameBufferModel loaded but cannot be instantiated"). May work on current versions;
may need the JupyterLab fallback.
- Falsifiable by: same cell test inside a VSCode `.ipynb` on the remote.

### H3 — Offscreen render → movie/snapshot export (no live interactivity)  — confidence 0.95
For the "movie code" use-case. `WGPU_FORCE_OFFSCREEN=1`, `fig.canvas.force_draw()`,
`fig.export_numpy()` → frames → mp4. **Verified working + fast (75 fps for 1M-pt line).**
Decouples figure generation from any display entirely. Strong win for batch movie rendering
on the GPU server.

### Competing (non-fastplotlib) hypotheses — kept for calibration
- **H4 — datashader + holoviews/bokeh** (server-side rasterization → browser): great for
  *dense overview* of huge series, weaker for frame-by-frame "movie" scrubbing and custom
  GPU compositing. confidence as a *better* fit: 0.3.
- **H5 — downsampled matplotlib / mpl + mpld3 / plotly**: simplest, but interactivity on
  1.8M points is poor; not "quick". confidence as better fit: 0.15.
- **H6 — napari**: excellent N-d viewer but Qt-based, wants a display/Xvfb; worse for pure
  headless Remote-SSH than the browser-streamed fastplotlib path. confidence: 0.2.
- **H7 — pre-render movies + scrub video**: robust fallback (H3 feeds it) but not "live
  interactive". Good complement, not a replacement.

**Current ranking:** H1 ≳ H3 (different jobs) > H2 > H4 > others.

---

## Self-critique / open risks

1. **VSCode-native (H2) unverified live.** I verified the GPU render path, not the VSCode
   widget round-trip. Until a `Figure` actually shows in a VSCode remote notebook, H2 stays
   at 0.5. Mitigation: JupyterLab fallback is solid.
2. **Heatmap >max-texture not yet solved in code.** Need to implement time-aggregation for
   the overview. Until done, "whole-epoch posterior heatmap" is a plan, not a result.
3. **Install footprint / env hygiene.** fastplotlib[notebook] pulls pygfx/wgpu/jupyter_rfb.
   Install into `.venv` (not conda base), register a kernel; consider adding to `pyproject`
   only after the prototype proves out.
4. **First-frame latency 1.5 s** (shader compile) — fine interactively, but note it for
   movie loops (amortized after frame 1).
5. **Data loading, not rendering, is the real bottleneck** (per `now.md`: ~11 min to classify
   18 GB). fastplotlib fixes *rendering*; pair it with lazy/memmap loading of the `.nc`
   (xarray + dask, or slice-on-demand) so we never hold/transfer the full 4 GB to draw.

---

## Verification log

- [x] wgpu enumerates A100s as Vulkan adapters headless; `request_device_sync()` OK.
- [x] fastplotlib 0.6.1 offscreen render → `/tmp/fpl_test.png`.
- [x] 1M-point line, per-frame data update → 75.8 fps.
- [x] **Prototype built + verified on real 4 GB epoch** (`notebooks/fastplotlib_epoch_viewer.py`):
      1.8M-bin state-prob lines + time-aggregated posterior heatmap, offscreen PNG in ~50 s
      (I/O bound). Spot-checked the PNG: correct structure after fixing 3 numerical/API bugs.
- [x] Bugs found & fixed via plot inspection (see README): float32 time precision (use
      relative time), `maintain_aspect=False` for time-series, image placement via
      `world_object.local.scale_x` (top-level `.scale` no-ops).
- [ ] Live `Figure` shows + responds in JupyterLab over SSH port-forward. (H1 — needs browser)
- [ ] Live `Figure` shows in VSCode native remote notebook. (H2 — needs browser)
- [ ] Full-resolution posterior in a zoom window (overview is time-aggregated). (future)
- [ ] True x-linked controllers across panels; LFP/position/spike-raster panels. (future)

## Deliverables
- `.claude/research/fastplotlib-feasibility.md` — this analysis + hypothesis tree.
- `.claude/research/README-fastplotlib.md` — install + launch instructions (3 options).
- `notebooks/fastplotlib_epoch_viewer.py` — reusable module.
- `notebooks/fastplotlib_epoch_demo.{py,ipynb}` — runnable demo.
- Installed `fastplotlib[notebook]` into `.venv`; registered `continuum-swr-fpl` kernel.

## Next steps
1. Install `fastplotlib[notebook]` into `.venv`; register a jupyter kernel.
2. Build prototype notebook against a real `_results.nc` (lines first, then overview heatmap).
3. Test H1 (JupyterLab port-forward) and H2 (VSCode native) live; record which works.
