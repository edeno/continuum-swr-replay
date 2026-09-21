# Yu et al. 2017 SWR detection — Frank lab MATLAB source

The consensus-ripple pipeline and the data-driven noise threshold used by
Yu JY, Kay K, Liu DF, Grossrubatscher I, Loback A, Sosa M, Chung JE, Karlsson MP,
Larkin MC, Frank LM (2017), *Distinct hippocampal-cortical memory representations for
experiences associated with movement versus immobility*, eLife 6:e27621,
doi:[10.7554/eLife.27621](https://doi.org/10.7554/eLife.27621).

**Nothing here is imported or executed by this project.** Read-only reference for the
port specified in
[docs/design/replay_detection_comparison/01-yu-swr-detector.md](../../docs/design/replay_detection_comparison/01-yu-swr-detector.md).

## Provenance

Recovered on 2026-09-21 from Anna Gillespie's home directory on `zephyr.cin.ucsf.edu`
(Frank lab). Neither file is in any git repository there or in any public Frank lab
repository (Bitbucket `franklab/filterframework_shared`, its `_open` mirror, or the
`droumis/FFPhy` fork on GitHub); the only public trace is the call site in
`DFFunctions/AG_extractRipplesJY.m`. The files were relayed through a Claude Remote
Control session (the driver as text, the estimator as an artifact file) and verified
against the original md5 sums:

| file | original path | mtime | bytes | original md5 | this copy |
|---|---|---|---|---|---|
| `jy_variableripthreshold_corecalculation.m` | `/home/anna/Src/Matlab/DayProcess/` | 2017-09-22 17:14:07 -0700 | 5606 | `2c2968b646cb9fe7afeabcc94ad84754` | byte-exact |
| `kk_extractconsensus2.m` | `/home/anna/Src/oldMatlabstuff/Functions/` | 2017-09-04 13:38:07 -0700 | 31103 | `fca0d5527b60fa086bb3f93a2984f779` | byte-exact |

**Licence: none stated.** Lab-internal code, vendored at the project owner's direction
(the project owner is a member of the lab). Do not redistribute without the authors'
permission.

## Files

- `jy_variableripthreshold_corecalculation.m` — Jai Yu's threshold estimator: fixed
  0.01-wide histogram on `[-10, 50]`, mode from an 11-point moving average of the counts,
  left half of the raw counts mirrored about the mode, threshold one bin past the point
  where the mirrored CDF reaches `1 - 1e-4`. Two alternatives (last empty mirrored bin;
  Gaussian fit at 2.5758 SD) are computed and discarded. Always draws a figure.
- `kk_extractconsensus2.m` — the consensus-trace and event-extraction driver, in Anna
  Gillespie's 2017 edit (function line renamed `ag_extractconsensus2`; header: "based on
  jai's version of kk_extractconsensus2"). Per-tetrode 4 ms Gaussian smoothing, optional
  per-tetrode z-scoring (`normeachtet`), `nanmedian` across tetrodes, immobility sample
  pooled over the day's run epochs, `extractevents` with the threshold above and the
  immobility mean as baseline, and an optional slow-baseline rejection (`adaptivefilt`).

## Caveats when reading

- `kk_extractconsensus2.m` reads `useenvelope`, `normeachtet`, `adaptivefilt`,
  `masterfolder`, and the `*text` cell arrays from `assign(varargin)`; **none has a
  default in the file**, and no caller of it exists anywhere in the trees searched, so
  the flag values used for the 2017 paper are not recoverable from this source. The
  2019 wrapper committed to Bitbucket, `AG_extractRipplesJY.m`, hard-codes the
  envelope, per-tetrode z-scoring, no adaptive filter, and endpoint-only immobility; it
  is the lab's consolidated version and pairs with the threshold function. The lab's
  production driver (`Trodes_dayprocess.m`) calls it with `nstd = []` and a 15 ms
  minimum duration, but every per-animal run script sets `extractRipplesJY = 0`, so
  the path is switched off in later production.
- The band filter those pipelines apply, `ripplefilter.mat` (2008-01-25, md5
  `ed158b07d679a38356b78e1e10e244da`), is byte-identical to the file shipped in
  `ripple_detection`.
- The reflection in the threshold function is written `abs(b) + 2m` for left bins `b`
  and mode `m`; that equals `2m - b` only when `b <= 0`. With a z-scored right-skewed
  envelope the mode is at or below zero and the formula is correct; a positive mode would
  mis-place the bins in `(0, m)`.
- `histc` bins are left-closed; the last bin counts values exactly equal to 50.
- `extractevents` (MEX; source in `droumis/FFPhy`, `DFFunctions/extractevents.cpp`)
  applies the minimum duration to the above-threshold run, then extends the event from
  the sample after the last at-or-below-baseline sample to the last above-baseline
  sample; with `allowstartend = 0` an event touching either end of the data is dropped.
- `jy_variableripthreshold.m`, which the estimator's header names as its caller, does not
  survive: a full case-insensitive search of the lab server's home tree (2026-09-21)
  returns only the `_corecalculation` file. `AG_extractRipplesJY.m` is the only surviving
  wrapper; do not repeat the search.
- The 0-0.5 Hz FIR used by `adaptivefilt` (`Hz0_5.mat`, `designeegfilt.m`) is not
  vendored; it is not needed for the port because the adaptive filter is not used.
