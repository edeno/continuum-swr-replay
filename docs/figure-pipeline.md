# Figure pipeline

How each figure is made: the script that draws it, what the script reads, where
it writes, and the tests that guard it. Which questions and values each figure
shows is written once, in the question list
(`src/continuum_swr_replay/reported/questions.py`), and listed by figure and panel
at the end of [../paper/reported/questions.md](../paper/reported/questions.md); a
question placed in a figure whose panels are not designed yet is listed under it
unlettered. Each analysis's script is described in [../scripts/README.md](../scripts/README.md).

Run each script from the repository root. Each one writes to its output folder,
most as a PDF and a PNG; `plot_swr_contrast_coefs.py` and the two example
galleries (`generate_inventory_examples.py`, `generate_event_feature_examples.py`)
write PNGs only. `make figures` draws Figures 1–4 in one go
(`scripts/figures/generate_all_figures.py`); run it only after the full rebuild
([reproduce.md](reproduce.md#the-full-rebuild)), because Figure 4 reads the
Part 4 fits.

## Main figures

`paper/main.tex` draws each one from `paper/figures/figureNN/figureNN.pdf`.

| Figure | Panels | Script | Reads | Tests |
| --- | --- | --- | --- | --- |
| 1. The switching state-space model | a: states and transitions; b: how each state scores the spikes; c: position transitions; d: a simulated run and its decode; e: one example per kind of simulated spiking | `scripts/figures/generate_figure01.py` | nothing: it draws the simulated run of `analysis/detection/simulated_run.py` | — |
| 2. Non-local events and SWRs | a: one example event per SWR-overlap class; b: events by overlap class; c: non-local probability and SWR z-score around event starts; d: event start and end against the matched SWR; e: decoding error in the Local state | `scripts/figures/generate_figure02.py` | one example session (`--nwb-file-name`, `--epoch-name`; default `wilbur20210329_.nwb`, `02_r1`) through `load_data` and its decode; every analyzed epoch's `{epoch_id}_80_data_over_time.pkl` (panels a, c, e); the canonical event table `replay_content/event_table.parquet` (panels b, d: the counts and offsets the events summary cites, by the same functions) | `tests/scripts/figures/test_figure_script_fixes.py`, `tests/analysis/event_definition/test_swr_source_consumers.py` |
| 3. What SWR and no-SWR events are like | a–c: duration, spatial extent and replay speed, each kind's distribution and median, with the SWR minus no-SWR difference (duration-matched for b, c) and its interval; d: arms visited, of the nine; e: content time by distance from the animal, at a well and elsewhere; f: the ratio of no-SWR to SWR content time per cm along every arm, on the track; g: content time per cm from the center to the junctions and from the junctions to the wells, with each kind's share at a well; h: theta and slow and fast gamma per animal, at rest, in each kind of event and running | `scripts/figures/generate_figure03.py` | the canonical event table, `event_gamma/`, `brain_state_reference/reference.parquet`, `content_along_arms/` (`scripts/event_properties/measure_content_along_arms.py`) | `tests/analysis/event_properties/test_content_along_arms.py` |
| 4. mPFC responses to SWR and non-SWR events | a: one neuron's modulation by both event kinds; b: the population, both kinds, one sort order; c: population modulation, the mean over animals (unit → session → animal) ± SEM over animals; d: the share of neurons whose event term improves held-out deviance, and the best model per neuron | `scripts/figures/generate_figure04.py` | the example session's HPC decode, stops, events and SWRs; the mPFC modulation bundles, `{epoch_id}_mPFC_modulation.pkl` and their `.npy` arrays, from `scripts/pfc/run_pfc_analysis_bandit.py` | `tests/scripts/figures/test_generate_figure04.py` |

Figure 1 needs no data. Figures 2–4 need the lab's Spyglass database and the
processed-data root ([reproduce.md](reproduce.md#requirements)).

Figure 1 is the only main figure committed so far. A main figure that does not
exist yet is drawn in the PDF as a framed box naming its script. When a figure is
final, add it to `.gitignore` as an exception, as Figure 1 is
([paper/README.md](../paper/README.md#figures)).

## Supplementary figures

These are not in `paper/main.tex` yet. Each script's docstring describes its
panels.

| Figure | Script | Output |
| --- | --- | --- |
| Decoder validation: the 0.8 threshold, state fractions, calibration, triggered averages, event durations | `scripts/figures/generate_supp_decoder_validation.py` | `paper/figures/supp_decoder_validation/` |
| The companion to Figure 1: the Widloski & Foster sliding-window detector on the same simulated run, and on replays swept over path length and duration | `scripts/figures/generate_supp_memoryless_comparison.py` | `paper/figures/supp_memoryless_comparison/` |
| The analysis pipeline, as a schematic and as a data flow | `scripts/figures/generate_supp_pipeline_schematic.py`, `scripts/figures/generate_supp_pipeline_data_flow.py` | `paper/figures/qc_diagnosis/` |
| Examples from each cell of the detector comparison | `scripts/figures/generate_inventory_examples.py` | `paper/figures/inventory_comparison/<run id>/examples/` |
| Comparator-only and ours-only events, with their spike evidence | `scripts/figures/generate_event_feature_examples.py` | `paper/figures/event_features/<run id>/examples/` |
| Content against reward-belief uncertainty (run `content.fig.uncertainty`) | `scripts/figures/generate_uncertainty_content_figure.py` | `paper/figures/uncertainty_content/` |
| SWR against non-SWR coefficients, rate and content (run `content.fig.swr_contrast_coefs`) | `scripts/figures/plot_swr_contrast_coefs.py` | `paper/figures/swr_contrast/` |
| Content models under three estimators (conditional Poisson on counts, on fractions, Dirichlet-multinomial) on one cohort | `scripts/replay_models/content/compare_content_estimators.py` | `paper/figures/content_estimators/` |

The Part 3 figures that read the fits are drawn by `scripts/workflows/run_refits.py`,
which names each by its run (`rate.fig.*`, `content.fig.*`; `--dry-run` lists
them).

## Supplementary tables

| Table | Script | Reads | Output | Tests |
| --- | --- | --- | --- | --- |
| S1. Detection and decoding parameters in <!--v:survey.papers-->57<!--/v--> replay studies | `scripts/figures/generate_supp_literature_table.py` | `paper/tables/table_s1_literature_parameters.csv`, the survey (Supplementary Data 1), via `continuum_swr_replay.reported.literature` | `paper/tables/table_s1_literature.tex` | `tests/reported/test_literature_table.py` |

Table S1 needs no lab data. Its values are corrected in the survey CSV ([paper/README.md](../paper/README.md#supplementary-tables)).

## Captions

No caption draft is kept in the repository. Three were, beside their figures, until
2026-11-11 (`paper/figures/<topic>/<topic>_caption.md` for `memoryless_comparison`,
`content_estimators` and `uncertainty_content`; they are in git history). The
content-estimator and uncertainty-content scripts write theirs again when they run;
the memoryless comparison's was hand-written prose for an earlier design of that figure. Its
script checks the verdicts its panels commit to (those about the labeled recording through
`memoryless_sequence/recording_statistics.py`, those about the sweep's timing through
`memoryless_sequence/sweep_statistics.py`, both of which the detection summary reads too). A number in a caption is never typed by hand: in a
markdown draft it is an inline value marker, `<!--v:<summary>.key-->…<!--/v-->`,
which `make reported-values` fills, and in `paper/main.tex` it is the matching macro
([paper/README.md](../paper/README.md#numbers)).
