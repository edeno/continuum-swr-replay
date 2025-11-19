# Baseline Figures for Regression Testing

This directory contains baseline PDF outputs from figure generation scripts, used for visual comparison during refactoring to ensure output consistency.

## Current Status

**Generated:** 2025-11-19
**Data Source:** `/cumulus/edeno/non_local_paper/Processed-Data` (old processed data, 16 GB copied)

### Available Baselines

- **figure02/**: ✅ Complete (22/22 PDFs, 817 KB)
- **figure03/**: ⚠️ Partial (13/16 PDFs, incomplete due to data quality issue)
- **figure04/**: ❌ Not generated (missing mPFC modulation files)

## Generation Process

### Scripts Used

Baseline-specific scripts were created to save PDFs instead of displaying interactively:

```bash
# Figure 02 baselines (complete)
uv run python scripts/generate_figure02_baseline.py

# Figure 03 baselines (partial)
uv run python scripts/generate_figure03_baseline.py
```

### Data Requirements

- **16 GB** of `*_HPC_80_data_over_time.pkl` files (182 files)
- Located in: `src/Processed-Data/`
- Source: `/cumulus/edeno/non_local_paper/Processed-Data`

### Known Issues

**Figure 03 Incomplete:**
- Generated 13/16 panels successfully
- Crashed at panel 14 with `TypeError: agg function failed [how->mean,dtype->object]`
- Root cause: Corrupted data column with concatenated NWB filenames instead of numeric values
- Error: `Could not convert string 'chimi20200213_.nwbchimi20200213_.nwb...' to numeric`
- This is a data quality issue in the old processed files, not a script issue

## Limitations

### Why These Are Incomplete

1. **Old Data:** These baselines use old processed data that will be replaced when decoding is re-run
2. **Missing Files:** Figure 04 cannot be generated (no mPFC modulation files exist)
3. **Data Quality:** Figure 03 has corrupted data preventing complete generation
4. **Temporary Status:** These baselines will be regenerated from fresh data

### Intended Use During Refactoring

**DO NOT** treat these as authoritative reference outputs. Instead:

- ✅ Use for visual spot-checking during development
- ✅ Compare structure and panel count
- ✅ Verify no major regressions (missing panels, broken layouts)
- ❌ Do NOT use for pixel-perfect comparison
- ❌ Do NOT block on missing figure03 panels 14-16

## Future Regeneration

After decoding is re-run with fresh data:

1. Copy new processed data to `src/Processed-Data/`
2. Run all figure scripts (original versions, not baseline-specific)
3. Save complete outputs to this directory
4. Document any differences from original baseline expectations
5. Update this README with new generation date and status

## Files in This Directory

```
tests/baselines/
├── README.md                          # This file
├── figure02/                          # Figure 02 baselines
│   ├── figure02_panel_01.pdf         # 22 complete panels
│   ├── ...
│   └── figure02_panel_22.pdf
├── figure03/                          # Figure 03 baselines (partial)
│   ├── figure03_panel_01.pdf         # 13 partial panels (of 16 total)
│   ├── ...
│   └── figure03_panel_13.pdf
└── figure04/                          # Figure 04 (empty - not generated)
```

## Baseline Script Modifications

The baseline-specific scripts (`generate_figure0X_baseline.py`) differ from the original scripts in two ways:

1. **Output directory:** PDFs saved to `tests/baselines/figure0X/` instead of displayed
2. **Save instead of show:** All `plt.show()` calls replaced with:
   ```python
   plt.savefig(OUTPUT_DIR / f"figure0X_panel_{i:02d}.pdf", bbox_inches="tight")
   plt.close()
   ```

These scripts are temporary and should be removed after fresh baselines are generated from the original scripts.
