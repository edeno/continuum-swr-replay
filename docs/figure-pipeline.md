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
| 1. The switching state-space model | a: states and transitions; b: observation likelihoods; c: position transitions; d: a simulated decode | `scripts/figures/generate_figure01.py` | nothing: it simulates its data (`--seed`, `--n-neurons`) | — |
| 2. Non-local events and SWRs | a: one example event per SWR-overlap class; b: events by overlap class; c: non-local probability and SWR z-score around event starts; d: event start and end against the matched SWR; e: decoding error in the Local state | `scripts/figures/generate_figure02.py` | one example session (`--nwb-file-name`, `--epoch-name`; default `wilbur20210329_.nwb`, `02_r1`) through `load_data` and its decode; every analysed epoch's `{epoch_id}_80_data_over_time.pkl` | `tests/unit/test_figure_script_fixes.py`, `tests/unit/test_swr_source_consumers.py` |
| 3. Non-local events outside SWRs | a: an example event with no SWR; b: SWR z-score at Local and Non-Local times; c: replay speed by SWR overlap; d: replayed position by SWR overlap | `scripts/figures/generate_figure03.py` | as Figure 2 | `tests/unit/test_figure_script_fixes.py`, `tests/unit/test_swr_source_consumers.py` |
| 4. mPFC responses to SWR and non-SWR events | a: one neuron's modulation by both event kinds; b: the population, both kinds, one sort order; c: population mean modulation; d: the share of neurons whose event term improves held-out deviance, and the best model per neuron | `scripts/figures/generate_figure04.py` | the example session's HPC decode, stops, events and SWRs; the mPFC modulation bundles, `{epoch_id}_mPFC_modulation.pkl` and their `.npy` arrays, from `scripts/pfc/run_pfc_analysis_bandit.py` | `tests/unit/test_generate_figure04.py` |

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
| Our detector and the Widloski & Foster detector on simulated replays | `scripts/figures/generate_memoryless_comparison.py` | `paper/figures/memoryless_comparison/` |
| The analysis pipeline, as a schematic and as a data flow | `scripts/figures/generate_supp_pipeline_schematic.py`, `scripts/figures/generate_supp_pipeline_data_flow.py` | `paper/figures/qc_diagnosis/` |
| Examples from each cell of the detector comparison | `scripts/figures/generate_inventory_examples.py` | `paper/figures/inventory_comparison/<run id>/examples/` |
| Comparator-only and ours-only events, with their spike evidence | `scripts/figures/generate_event_feature_examples.py` | `paper/figures/event_features/<run id>/examples/` |
| Content against reward-belief uncertainty (run `content.fig.uncertainty`) | `scripts/figures/generate_uncertainty_content_figure.py` | `paper/figures/uncertainty_content/` |
| SWR against non-SWR coefficients, rate and content (run `content.fig.swr_contrast_coefs`) | `scripts/figures/plot_swr_contrast_coefs.py` | `paper/figures/swr_contrast/` |
| Content models under three estimators (conditional Poisson on counts, on fractions, Dirichlet-multinomial) on one cohort | `scripts/content/compare_content_estimators.py` | `paper/figures/content_estimators/` |

The Part 3 figures that read the fits are drawn by `scripts/qa/run_refits.py`,
which names each by its run (`rate.fig.*`, `content.fig.*`; `--dry-run` lists
them).

## Supplementary tables

| Table | Script | Reads | Output | Tests |
| --- | --- | --- | --- | --- |
| S1. Detection and decoding parameters in <!--v:survey.papers-->57<!--/v--> replay studies | `scripts/figures/generate_supp_literature_table.py` | `paper/tables/table_s1_literature_parameters.csv`, the survey (Supplementary Data 1), via `continuum_swr_replay.reported.literature` | `paper/tables/table_s1_literature.tex` | `tests/unit/test_literature_table.py` |

Table S1 needs no lab data. Its values are corrected in the survey CSV ([paper/README.md](../paper/README.md#supplementary-tables)).

## Captions

A caption draft sits beside its figure as `<topic>_caption.md` (for example
`paper/figures/memoryless_comparison/memoryless_comparison_caption.md`). A number
in a caption is written as an inline value marker, `<!--v:<summary>.key-->…<!--/v-->`,
which `make reported-values` fills, never typed by hand. When a caption moves into
`paper/main.tex`, its numbers become the matching macros
([paper/README.md](../paper/README.md#numbers)).
