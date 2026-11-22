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
write PNGs only. `make figures` draws Figures 1–3 and 5 in one go
(`scripts/figures/generate_all_figures.py`); run it only after the full rebuild
([reproduce.md](reproduce.md#the-full-rebuild)), because Figure 5 reads the
Part 4 fits.

## Main figures

`paper/main.tex` draws each one from `paper/figures/figureNN/figureNN.pdf`.

| Figure | Panels | Script | Reads | Tests |
| --- | --- | --- | --- | --- |
| 1. The switching state-space model | a: states and transitions; b: how each state scores the spikes; c: position transitions; d: a simulated run and its decode; e: one example per kind of simulated spiking | `scripts/figures/generate_figure01.py` | nothing: it draws the simulated run of `analysis/detection/simulated_run.py` | — |
| 2. Non-local events and SWRs | a: one example event per SWR-overlap class; b: events by overlap class; c: non-local probability and SWR z-score around event starts; d: event start and end against the matched SWR; e: decoding error in the Local state | `scripts/figures/generate_figure02.py` | one example session (`--nwb-file-name`, `--epoch-name`; default `wilbur20210329_.nwb`, `02_r1`) through `load_data` and its decode; every analyzed epoch's `{epoch_id}_80_data_over_time.pkl` (panels a, c, e); the canonical event table `replay_content/event_table.parquet` (panels b, d: the counts and offsets the events summary cites, by the same functions) | `tests/scripts/figures/test_figure_script_fixes.py`, `tests/analysis/event_definition/test_swr_source_consumers.py` |
| 3. What SWR and no-SWR events are like | The starting hypothesis is that the two are different kinds of event, so every panel draws them separately; the legend gives each kind's number of events. a: duration, each kind's distribution, its median and each animal's median; b: the mean share of an event in Continuous and in Fragmented, the animals' mean and each animal; c: spatial extent, as a; d: replay speed (each event's median over its time bins), as a, and under it the share of events at or below each speed (in a and d the end bin marked ≥ also holds the events beyond the axis); e: each kind's content time per cm along every arm, drawn on the track once per kind against what content spread evenly over the track would give, with the track's center, a junction and a well named; f: content time per cm from the center to the junctions and from the junctions to the wells, with each kind's share at a well, against what content spread evenly over the track would give (the dashed line and its share). The figure describes the content from the decode and the track alone: content against the animal (by distance from it, by which well; `content_along_arms.distance_from_animal_shares`) is left to the behavior figure. Slow and fast gamma during the events are a supplementary figure; ripple power, MUA, theta and speed around the events are Figure 2's. The number of arms an event's decoded position touches is in the events summary, in no panel | `scripts/figures/generate_figure03.py` | the canonical event table, `content_along_arms/` (`scripts/event_properties/measure_content_along_arms.py`), and one epoch's decode environment for the track drawn in f | `tests/analysis/event_properties/test_content_along_arms.py`, `tests/scripts/figures/test_figure_script_fixes.py` |
| 5. mPFC responses to SWR and non-SWR events | The two kinds of event are drawn separately and named as in Figures 2 and 3; here "SWR" is every detected SWR and "No SWR" every non-local event that overlaps none (the two kernels of the joint model). a: one unit's modulation by each kind, over its firing rate around each kind's event starts; b: every unit's modulation by each kind, the units in one order (the time of their SWR peak), on one color scale whose ends are marked; c: population modulation, the mean over animals (unit → session → animal) ± SEM over animals; d: the share of units whose event term improves held-out deviance beyond the matched control's null (bars pooled over units, dots the animals), and the best model per unit. Drawn to the standards of Figures 1–3: the publication style, one column grid, letters by row and column, `check_placement` and `check_journal_limits` | `scripts/figures/generate_figure05.py` | the example session's HPC decode, stops, events and SWRs; the mPFC modulation bundles, `{epoch_id}_mPFC_modulation.pkl` and their `.npy` arrays, from `scripts/pfc/run_pfc_analysis_bandit.py` | `tests/scripts/figures/test_generate_figure05.py` |

**The data subset of Figures 2 and 3** (and of the events summary,
`paper/reported/events.json`, and the gamma supplementary figure). Every event
panel is over the immobile non-local events that start inside a well visit,
from the first nose-poke in at a well to the final poke-out there:
`visit_context` `in_visit` or `spans_visit_end`, selected once, where the
canonical event table is read, by
`analysis/event_definition/cohort_selection.py:select_visit_context`. Figure 2
checks the whole table against the decode frames first, then selects. A stored
per-event output (participation, the content's time bins, the start distance,
the gamma per event) is read over the selected events
(`cohort_selection.select_events_in`); one that is a statistic over events (the
property bootstrap, the one-class-or-two tests) is made over them by its
producer. Time and SWRs count only inside a well visit
(`analysis/event_definition/well_visits.py:in_well_visit`, the visits read from
the session cache by `analysis/event_tables/prepare_data.py:well_visit_times`):
the quiet rest of the brain-state reference, of the rest-theta windows and of
the gamma reference, the immobile time bins of Figure 2j, and the SWRs and other
detectors' events of Figure 2c and 2e, which count when their first sample is
inside a visit (running, the theta-like benchmark, is the epoch's running
wherever it is). Each producer re-run over the subset records
`cohort_selection.ANALYZED_SUBSET` in its manifest, and one whose content depends
on the visits the digest of the session-cache entries it read them from
(`prepare_data.session_cache_record`); Figure 2 and the gamma figure refuse a
reference that records another subset or other visits than the current cache's
(`prepare_data.time_mask_staleness`), and the events summary marks the values
read from one pending. What another inventory leaves out (question 2b.2) is over
the selected non-local events and the non-local detector's exposure inside a
visit (`scripts/detector_comparison/measure_exposure_in_well_visits.py`, which
rebuilds the inventory comparison's exposure from its stored tables and refuses
one that is not the stored one).
The participation run (`event_participation/`) keeps a row for every event and
draws each event's Local-rest control wholly inside one well visit
(`analysis/event_evidence/participation.py:CONTROL_RULE`); its settings record
the subset and the rule, its inputs the session-cache digest, and the summary
marks the participation values pending when either is not current. The
held-out tetrode run takes its controls from the participation tables, so its
values say which controls they were made with.
The summary counts the events of every well-visit label, kept and left out
(`events.json`, `events_label_*`, `*events_in_subset`, `*events_left_out`).
The spike-evidence run (`event_spike_evidence/`) scores every event and every
matched quiet window; each scored event is one of the event table's, matched to
its `event_id` by epoch and start time
(`analysis/event_evidence/spike_evidence.py:with_event_ids`, an event without a
match is an error), and the summary reads the selected events and the quiet
windows whose first sample is inside a visit (`in_analyzed_subset`;
`*_left_out` counts the others). The held-out test split by SWR
(`scripts/event_evidence/split_state_evidence_by_swr.py`) reads the decoder's
immobile Non-Local bouts whose first sample is inside a visit, and every running
bout, each with its matched window (`state_evidence.mark_bouts_in_well_visits`);
the matched windows were
drawn from all of the epoch's time in the bout's regime, so some of the kept
bouts' windows start outside every visit, which its provenance counts
(`bouts`, with the bouts kept and left out per state, SWR label and regime;
`events.json`, `heldout_*_bouts_*`).

Figure 1 needs no data. Figures 2–4 need the lab's Spyglass database and the
processed-data root ([reproduce.md](reproduce.md#requirements)).

Figure 4, the events and behavior (Part 3 of the question list), has no script yet. Figures 1–3 and 5 are committed, and Figure 4's files will be tracked as soon as its script writes them (`.gitignore` excepts all five). A main figure that does not
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
| Slow and fast gamma during SWR and no-SWR events: each animal's level in each kind of event, between quiet-rest and running windows as long as the kind's events (was Figure 3's last panel) | `scripts/figures/generate_supp_event_gamma.py` | `paper/figures/supp_event_gamma/` |
| The held-out tetrode gain over local coding per spike of no-SWR and SWR events: a dot per animal, the animals' mean and its interval (was Figure 2's panel h; Figure 2g now compares the full and SWR-only decoders) | `scripts/figures/generate_supp_heldout_tetrode.py` | `paper/figures/supp_heldout_tetrode/` |
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
