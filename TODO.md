# Publication follow-ups

Work that is deferred on purpose. Each item says when it should be done.

## Paper metadata

- [ ] Replace the working title, and confirm the tentative author list and each
  author's affiliations in `CITATION.cff` and `paper/main.tex`.
  *When:* before the first draft goes to the co-authors.
- [ ] Choose the target journal and switch `paper/main.tex` to its document
  class and bibliography style.
  *When:* once the journal is chosen.

## Reproduction

- [ ] **Public reproduction from archived tables (Tier 1).** Archive the
  canonical tables and per-epoch derived files that the part summaries read
  (`paper/reported/<summary>.json`, `inputs`) on Zenodo, with a download script
  that checks each file's SHA-256, so the figures and summaries can be rebuilt
  without the lab's data. Fitted-model pickles are not archival inputs: the
  public path should refit the models from the tables.
  *When:* before submission, or when an outside reader needs to rerun the
  analyses.
- [ ] **Lab pipeline from NWB (Tier 2).** Document the decode, detection and
  table-building steps from Spyglass/NWB as a lab guide, and publish the NWB
  files on DANDI.
  *When:* with the data statement for submission.
- [ ] **Summary input paths and rebuild comparison.** Record input paths
  relative to the data root, not absolute paths (a build whose data root is
  outside the checkout records `/cumulus/...`; the current summaries were built in
  a checkout that links `Processed-Data`, so theirs are relative). Add a script that compares a rebuilt summary with the
  committed one within stated tolerances. Record each input file set's glob, not
  a label, so `make release-check` can re-check sets too (today it re-checks only
  single files). (Each build records the code that built it; a summary in
  `paper/reported/` is written only from committed code, and `make release-check`
  flags a stale one.)
  The builders read CSV inputs with pandas' default float parser, which is not
  exact and differs by machine: of the 1,492 float values in
  `paper/reported/inputs/detection/detection_sweep_summary.csv`, 125 on
  an arm64 laptop differ in the last digit from the exact parse
  (`float_precision="round_trip"`), and the cluster (x86_64) reads one of the
  detection summary's values differently from the laptop. So a summary is
  bit-identical only when rebuilt on the same kind of machine. Reading with the
  round-trip parser would make it exact everywhere and change those values in
  their last digit.
  *When:* together with Tier 1, or when a number in the paper cannot be traced
  back to a commit.

## Reported values

- [ ] **Scripts and figures that compute a reported statistic themselves.** Each
  summary's statistics are functions in `analysis/` (`docs/development.md`,
  "Reported numbers"). Figure 2's panels b and d and Figure 4d call them, and the
  `scripts/pfc/` reports take the summary's statistics from them (the
  marked-family summary's are `analysis/pfc/marked_family_statistics.py`'s); their
  audit-only tables are still computed in the scripts.
  These still hold their own copy; make each call the function, comparing its
  output before and after on the cluster:
  - events: `scripts/detection/analyze_event_spike_evidence.py` (per-epoch and
    per-animal medians written to CSVs nothing reads).

  *When:* at each figure's or report's next regeneration.

- [x] **Figure 4c: pooled or animal-balanced.** *Decided 2026-11-12: animal-balanced*,
  to match how the paper summarizes its other statistics. Panel c
  (`scripts/figures/generate_figure04.py`, `plot_population_average`) takes each
  unit's curve less its own value at -200 ms and reduces it unit -> session ->
  animal, the mean over animals with the SEM over animals
  (`analysis/pfc/population_statistics.animal_balanced_unit_curve`, the reduction the
  marked-family population curves use). It drew the mean over all units with the SEM
  over units before; over half the units are one animal's (wilbur), and the
  animal-balanced curve peaks elsewhere. The mpfc summary reports no value for this
  curve (its only 4c value is the unit count), so the figure copies no reported
  number. The choice is open to a different method later: the panel's reduction is
  that one call, so changing it means swapping the function.

- [ ] **Decide the open choices in the reported statistics.**
  `docs/known_issues/reported_statistics_choices.md` lists them: where each is
  made, what the code does, what is open, and what was measured on the current
  data. They were found while the statistics moved into `analysis/` and in the
  review of that code. Write each decision in that file.
  *When:* before the Results that cite the value are drafted.

- [ ] **Open decisions from the retired figure plan** (`paper/figure_claims_plan.md`,
  removed after `cad31719`; it used an older claim numbering, so "D" is its claim
  "our method beats SWR-triggered decoding"). Quoted as it wrote them; confirm
  each is still open, or close it:
  - "Close vs remote: scope of headline claims (default remote-clean)."
  - "Whether D needs a literal traditional-pipeline re-implementation or a
    derived contingency suffices."
  - "Does Fig 3d's 'reward zones' survive the remote-subset + autocorrelation
    controls, or get reframed as 'recent/valued locations'?"

  *When:* when Figures 2 and 3 are designed.

- [ ] **Open analyses from the retired notes.** Quoted from
  `paper/reported/overview.md`, `docs/replay_rate_content_swr_synthesis.md` and
  `docs/per_bin_rate_results.md` (retired; read them at `c74e56b8`). Confirm each,
  or close it:
  - Tests at the recording-day level: a p or q value over sessions treats the
    epochs of one recording day as independent, which they are not; until then,
    the animal counts are the level to read a claim at.
  - Property contrasts (duration, extent, speed, ...) are plain medians unless a
    value says matched duration; non-SWR events are shorter than SWR events.
  - The behavioral-state covariates on rate (value, uncertainty, switch rate,
    block): the `MU_uncertainty` and `is_swr`-contrast rate ladders report fit
    counts and separation shares, not coefficients; any re-derived coefficient
    must state its excluded (separated) share.
  - The `Mvu` conditioning diagnostics were count-response fits; re-derive them on
    the per-event fraction response.
  - The per-arm value-update (realized gain) re-run under the beta covariates has
    not been done.
  - The content arm hierarchy has no held-out cross-validation (model comparison
    rests on ΔAIC on the quasi-likelihood scale).
  - The choice fit (question 3.4) has no provenance sidecar.
  - Not built: the full marked point process (joint intensity, coupling and
    content); a graded ripple-power content rung.
  - Hypotheses that need a test: a continuum rather than two mechanisms (needs
    the peri-onset peak distribution and the joint model); reward shifting SWR
    coupling continuously; value-gating of with-SWR rate.

  *When:* before the Results are drafted.

## Checks

- [ ] **Type checking.** `mypy src/` reports about 400 errors in 90 files.
  Fix them, then add mypy to CI and pre-commit.
  *When:* as a separate cleanup, before mypy is relied on.
- [ ] **Pin GitHub Actions to commit SHAs.** The workflows use version tags
  (`@v4`), which the action's owner can move.
  *When:* before the repository is made public, or before a workflow gets a
  secret or deploys anything.
