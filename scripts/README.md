# Scripts index

Analysis scripts grouped by domain. Run from the repo root, e.g.
`uv run python scripts/rate/fit_per_bin_rate_models.py`. Scripts import the package
(`continuum_swr_replay`) and read/write `Processed-Data/`; they do not import each other.

## `decoding/` — produce the decoded data + event/stop tables
- `run_decoding_bandit.py` — continuum decoding for the bandit task (HPC/mPFC/OFC).
- `run_decoding_contfrag.py` — decoding for the ContFrag (SWR/MUA) event sources.
- `regenerate_stop_table.py` — rebuild the per-stop immobility table.
- `build_timebin_tables.py` — build the per-(stop×bin) design tables.
- `prepare_contfrag_replay_content_data.py` — assemble ContFrag content tables.

## `rate/` — event-rate models + their plots
- `fit_per_bin_rate_models.py`, `fit_per_stop_rate_models.py` — Family B / A rate ladders.
- `fit_swr_contrast_rate.py` — the `is_swr`-contrast per-bin ladder (G0–G8 / S0–S10).
- `fit_uncertainty_rate.py`, `fit_cross_history_rate.py`, `fit_swr_power_model.py` — MU / cross-history / SWR-power models.
- `plot_per_bin_rate_models.py`, `plot_per_bin_rate_summary.py`, `plot_model_checking.py`, `plot_uncertainty_rate.py` — rate figures.
- `summarize_per_bin_rate_models.py` — generates `docs/per_bin_rate_results.md`.

## `content/` — replay content / choice models
- `fit_replay_content_models.py` — arm/patch content hierarchy (M0–M7 / P0–P4), per `is_swr` split.
- `fit_swr_contrast_content.py` — the HMM-harmonized content `is_swr`-contrast ladder (C / CP).
- `compare_event_source_models.py` — choice (clogit) model comparison.
- `analyze_content_occupancy.py`, `quantify_swr_overlap_geometry.py`, `generate_replay_content_figures.py`.

## `behavior/`
- `fetch_spatial_bandit_behavior.sh` — pull the canonical SpatialBanditTask behavior CSVs.

## `figures/` — publication figures
- `generate_figure01.py … generate_figure04.py`, `generate_all_figures.py` (orchestrator).
- `generate_supp_*.py`, `generate_event_gallery.py`, `run_event_gallery.sh`.
- `archive/` — retired figure scripts (not linted; kept for provenance).

## `qa/` — diagnostics / validation
- `check_data_availability.py`, `diagnose_trial_detection.py`, `compute_hpc_raw_stats.py`,
  `run_quality_metrics.py`, `analyze_event_spike_evidence.py` (spike-evidence reality check).

## `pfc/` — cross-area / staged analyses
- `run_pfc_analysis_bandit.py`, `run_stage5b_retrospective.py`, `run_stage6_learning.py`.

## `exploratory/` — unsupported one-offs
Scratch scripts (`_*.py`): diagnostics, ad-hoc plots, verification checks. Not part of the
audited pipeline; outputs go to `.claude/research/*_outputs/`. May be stale.
