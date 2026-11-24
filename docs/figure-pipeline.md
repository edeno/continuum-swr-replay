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
| 2. Non-local events and SWRs | a: the decode over a run and the rest after it, with the 2D track; b: six examples (inside an SWR, starts before it, ends after it, two with no SWR, an SWR with Local content); c: the SWRs by content; d: the non-local events by SWR overlap; e: for each SWR definition, MUA and the Widloski detector, the share of its events with non-local activity and the share of non-local events it finds; f: event start and end against the matched SWR; g: ripple power, MUA, theta power and speed around event starts; h: tetrode participation, the share of tetrodes active in the events against their length-matched Local-rest windows (moved from the supplement: the events are population events); i: where the non-local time sits in ripple power, its top strip confident Local against confident non-local time (each class's time over ripple power, from `state_by_signal/`); then in a row of their own, each with a small drawing of its test (`visualization/evidence_drawings.py`): j: do the spikes of a left-out tetrode point to the decoded place? On one axis, its better fit to the decoded place than to 1, the animal's place (the gain over local coding) and 2, another event's decoded place of the same width (the agreement minus the width-matched reference), for no-SWR and SWR events; k: does the decoder need non-local states outside SWRs? The full minus the SWR-only decoder inside and outside SWRs, and, on its own axis, outside SWRs in the no-SWR held-out events and in the other time (the gains over the local-only decoder are for the text). One averaging rule: each animal's value first, then the mean over animals, the animals as dots (c, d, e: each animal's share is the mean over its epochs of each epoch's share, the bar their mean; f: each animal's percent of its event-SWR pairs in a bin is the mean over its epochs of each epoch's percent, the bar their mean; g: each epoch's median and quartiles over its events, and its level between events, each animal's median over its epochs, then the mean over animals; i: each animal's share of bins decoded non-local at each level, and its share of non-local time below the threshold, is the mean over its epochs of each epoch's share, the bold line and the printed percent their mean); counts printed in a panel are totals. | `scripts/figures/generate_figure02.py` | one example session (`--nwb-file-name`, `--epoch-name`; default `parameters.EXAMPLE_NWB_FILE`, `EXAMPLE_EPOCH`): its decode frame (position and SWRs), its decode's results and environment (track graph), and its ripple trace and HPC spikes stored in `example_session/<epoch_id>/` (`scripts/decoding/store_example_session.py`, which alone reads the lab database) (panels a, b); `event_trigger_windows/` (`scripts/event_properties/measure_trigger_windows.py`: panel g, and the frames' event counts the table is checked against); the canonical event table `replay_content/event_table.parquet` (panels d, f); the event features run (c, e); `brain_state_reference` (g); `event_participation` (h); `nonlocal_by_signal` and `state_by_signal` (i); `event_tetrode_holdout` (j); `event_nonlocal_restriction_heldout_events`, with the restriction and per-bin facts runs it was made from (k). A stored output is refused, as the events summary marks its values pending, when it is not current; the canonical table's event counts by SWR overlap must equal the decode frames' | `tests/scripts/figures/test_generate_figure02.py`, `tests/analysis/event_definition/test_swr_source_consumers.py` |
| 3. What SWR and no-SWR events are like | The starting hypothesis is that the two are different kinds of event, so every panel draws them separately; the legend gives each kind's number of events. a: duration, each kind's distribution, its median and each animal's median; b: the mean probability of the Continuous and of the Fragmented state over an event's time bins, the animals' mean and each animal; c: spatial extent, as a; d: replay speed (each event's MAP line-fit speed: the slope, at least 0, of the track distance from its first decoded position against time), as a, and under it the share of events at or below each speed (in a and d the end bin marked ≥ also holds the events beyond the axis); e: each kind's content time per cm along every arm, drawn on the track once per kind against what content spread evenly over the track would give, with the track's center, a junction and a well named; f: content time per cm from the center to the junctions and from the junctions to the wells, the animals' mean over a thin line per animal, with each kind's share at a well, against what content spread evenly over the track would give (the dashed line and its share); g: the distance of the content from the animal (each event's mean over its bins), the percent of a class's events within each distance on a log axis, a line per animal and their mean, with 20 cm and 100 cm marked (each class's share under 20 cm, 20 to 100 cm and 100 cm or more are reported values; was the supplement's panel c). Content by which well the animal is at (`content_along_arms.distance_from_animal_shares`) is left to the behavior figure. Slow and fast gamma during the events are a supplementary figure; ripple power, MUA, theta and speed around the events are Figure 2's. The number of arms an event's decoded position touches is in the events summary, in no panel | `scripts/figures/generate_figure03.py` | the canonical event table, `content_mass_along_arms/` (`scripts/event_properties/measure_content_mass_along_arms.py`: each event's posterior mass at each place, the content of e and f), `event_speed/` (`scripts/event_properties/measure_event_speed.py`), and one epoch's decode environment for the track drawn in e | `tests/analysis/event_properties/test_content_along_arms.py`, `tests/analysis/event_properties/test_event_speed.py`, `tests/integration/test_event_speed_simulated.py`, `tests/scripts/figures/test_figure_script_fixes.py` |
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
the gamma reference, the immobile time bins of Figure 2i, and the SWRs and other
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
| The companion to Figure 1: the Widloski & Foster sliding-window detector on the same simulated run, and on replays swept over path length and duration | `scripts/figures/generate_supp_figure01.py` | `paper/figures/supp_figure01/` |
| Figure 3's supplement (S3). a, b: slow and fast gamma during SWR and no-SWR events: each animal's level in each kind of event, between quiet-rest and running windows as long as the kind's events (was Figure 3's last panel); c: the speed of each event's position distribution (the transport-slope speed: the slope, at least 0, of the earth mover's distance on the track between the Continuous-state distributions of two time bins against their lag, 10 to 20 ms), the percent of a class's events at or below each speed, a line per animal and their mean, over the events that have one (`event_speed/`); d: the share of each kind's content time at a remote well and at the animal's own well, each animal's two kinds joined by a line, and the animals' mean; e: the mean share of an event's content time at a remote well by the event's spatial extent, a line per animal and their mean; f: the share of each kind's events that represent no well, one well, and two or more (a well holding at least a tenth of the event's content time), the animals' mean and each animal (d to f from `content_mass_along_arms/`; d with the 95 % interval of each mean) | `scripts/figures/generate_supp_figure03.py` | `paper/figures/supp_figure03/` |
| Figure 2's supplement (S2), in five rows. a: the decoder's states calibrated by held-out spikes per spike inside the well visits (Local, Continuous, Fragmented on one axis); b: MUA around the start of No-Spike periods; c: how the immobile time inside the well visits is spread over non-local probability, a line per animal and their mean on a log axis, the event rule's cut (0.8) marked (`nonlocal_probability/`, `scripts/decoding/measure_nonlocal_probability.py`; the shares at most 0.2, between 0.2 and 0.8 and above 0.8 are reported values); d: how far the decoded place is from the animal against the Local probability over all analyzed time, a line per animal (the median over its sessions of each session's median) and their mean (the decoder-quality run's table; not a calibration); e: the share of events with no SWR, and the events, at other settings of the event rule (probability threshold, minimum duration 14 to 100 ms) and of the SWR detector (threshold, 0 to 6 z); f: each event against its SWR, nine rows (same start and end, within 10 ms; one bound apart: starts later, starts earlier, ends earlier, ends later; both apart: shorter at both ends, longer at both ends, starts and ends later, starts and ends earlier), each with a small drawing of the case, a bar of the mean over animals, a dot per animal and the percent; g: non-local time against MUA, in Figure 2i's format, its top strip confident Local against confident non-local time; h: Figure 2j's first comparison (over the animal's place) over all bins, bins with a spike and silent bins; i: Figure 2k's gain outside SWRs, then only farther than 100, 250 and 500 ms from every SWR (the shares of that gain and the gain at well-sampled places are reported values for the text, not drawn); j: Figure 2g's traces around event starts, a line per animal and class (each animal's median over its epochs of each epoch's median over its events; Figure 2g's line is their mean); k: the posterior's spread, the percent of events by the 95 % HPD length (each event's mean over its bins) on a log axis, a line per animal and their mean, with the 50 cm cut of the hpd_tight_* values marked; l: Figure 2k's decoder comparison with a fourth decoder, SWR-allowed (every state inside SWRs, only the local states outside), from `event_nonlocal_restriction_heldout_events` (the restriction run scores all four decoders). The low-rate places (the low-rate values), Figure 2j's second comparison by the width of the decoded place (the shared-content values) and the step test at the SWR threshold (the step values) are reported values with no panel. b, c, d, e, f and g follow Figure 2's averaging rule: b animal-balanced as Figure 2g, c's, d's and e's bold lines and f's bars the mean over animals of each animal's share, g as Figure 2i. Reads `state_calibration_in_visits/` (a, `scripts/event_evidence/summarize_state_calibration_in_visits.py`), `event_trigger_windows/` (b, j), `nonlocal_probability/` (c), `decoder_quality/calibration.parquet` (d), `event_rule_sweep/` (e, `scripts/event_properties/sweep_event_rule.py`), the canonical event table (f, k), `nonlocal_by_signal/` and `state_by_signal/` (g), `event_tetrode_holdout/` (h), and `event_nonlocal_restriction_heldout_events/` with the restriction and per-bin facts runs (i, l); a stored output is refused, as the events summary marks its values pending, when it is not current. Tests: `tests/scripts/figures/test_generate_supp_figure02.py` | `scripts/figures/generate_supp_figure02.py` | `paper/figures/supp_figure02/` |
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
