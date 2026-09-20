# Widloski & Foster 2025 — vendored reference code

Third-party MATLAB source, vendored verbatim so the replay-detection port can be
checked against the original line by line. **Nothing here is imported or executed
by this project.** It is read-only reference material.

## The paper

Widloski, J. & Foster, D. J. *Replay without sharp wave ripples in a spatial memory
task.* Nature Communications 16 (2025). doi:10.1038/s41467-025-65181-5

- Article: <https://www.nature.com/articles/s41467-025-65181-5>
- Open full text (PMC): <https://pmc.ncbi.nlm.nih.gov/articles/PMC12639102/>
- Preprint: <https://www.biorxiv.org/content/10.1101/2024.05.13.593931v1>
  (doi:10.1101/2024.05.13.593931)
- Author correction (reference errors only, no method changes):
  <https://www.nature.com/articles/s41467-026-72252-8> ·
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC13090353/>

## `zenodo_15199609/` — the paper's own scripts

Widloski, J. *Replay without sharp wave ripples in a spatial memory task* (software).
Zenodo, 11 April 2025. doi:10.5281/zenodo.15199609 — <https://zenodo.org/records/15199609>

Licence: **CC-BY-4.0**. Six analysis/figure drivers, unmodified.

`load_replayEvents_cellIDshuffle.m` is the only end-to-end implementation of the
detection pipeline in the deposit; lines 36–156 are the authoritative reference for
the port. The deposit ships **no helper functions and no `Analysis_Information.mat`**,
which is why the second directory exists.

## `foster_lab_matlab/` — the helper library

Source: <https://github.com/caitlinmallory/TimeCourseOrganizationOfHippocampalReplay>
pinned at commit `128513a656bdacfc11a0e0dfbcaf20e2c4515520` (2024-11-28).

Re-fetch the exact tree with:

```bash
git clone https://github.com/caitlinmallory/TimeCourseOrganizationOfHippocampalReplay
git -C TimeCourseOrganizationOfHippocampalReplay checkout 128513a
```

**Licence: none stated.** The repository carries no LICENSE file and reports
`license: null` via the GitHub API, i.e. default all-rights-reserved. It is vendored
here at the project owner's direction. Do not redistribute this directory without
first resolving permission with the authors.

This is a *different* paper from the same lab, so implementations are shared ancestry
but some functions have drifted (see the caveats below). It supplies the primitives
the Zenodo deposit omits: the decoder, the posterior summary statistics, the sequence
assembler, and the parameter file.

### Caveats when reading this directory

- `compute_filtering_binDecoding_cm.m` computes jumps as `abs(diff(x(:,2)))` — the
  **x-component only**, not the 2-D L2 norm that Widloski's Zenodo script gets from
  `compute_sequenceJumps`. Follow the Zenodo script and the paper, not this file.
- `load_AnalysisInformation_cm.m` sets `SWRFreqRange = [150,250]` with the comment
  `% John used 120-170`, while the paper states 100–220 Hz. Three different numbers;
  the paper governs for ripple detection.
- `load_replayEvents_cm.m` carries Widloski's parameters verbatim in a commented
  "John's criterion" block, but with `sequence_durationThr`/`replay_durationThr` of
  0.05 s. The Zenodo script and the paper both use 0.1 s — an older revision.
- Files suffixed `_cm` are the Mallory variants; where an unsuffixed original exists,
  the unsuffixed one is closer to Widloski's.

### Still missing everywhere

Called by the scripts above, defined in neither source and not found by GitHub-wide
code search. All are small and fully determined by the paper's definitions plus their
call signatures; the port reconstructs them:

| function | what it must do |
|---|---|
| `compute_sequenceJumps(COM)` | L2 norm of `diff(COM)`, length N−1 (callers pad with `jumps(end)`) |
| `compute_sequenceDispersion(COM)` | RMS distance of COM from the mean COM, in bins |
| `compute_sequenceDistance(COM)` | path length along the COM trajectory |
| `compute_events_times(ts_z, zThr, mergeThr, durThr)` | threshold-crossing intervals; returns `[start, end, peakTime, …]` |
| `compute_replaysConcatenation(decoder, ind)` | concatenates per-event COM traces into one array |
