# Epoch register

**Status:** canonical, 2026-10-05. **Module:**
`src/continuum_swr_replay/data_loaders/epoch_register.py`. **Register:**
`src/continuum_swr_replay/data_loaders/epoch_register.csv`.

One register says which run epochs the project analyses and why every other one is
not. Every stage that enumerates epochs reads it, and the refresh audit checks every
artifact against it. It replaces the skip lists each stage used to keep (such as the
SWR runner's `KNOWN_NO_INPUT_EPOCHS`) and the exclusion notes in the `NWB_FILES`
docstring.

The register has one row per `(nwb_file_name, epoch)`: its `status`, its `reason`
(empty for an analysed epoch) and its `evidence`. An `epoch` of `*` covers every
epoch of a session none of whose run epochs is known (no NWB file, or no run epoch
in it). It was generated once, on 2026-10-05, from the exclusion audit and the
populate plan, both local to the analysis machine:

- the audit: `.claude/research/refresh_exclusions/exclusions_report.md` and
  `exclusions_table.csv`, with 557 rows: 553 run epochs and 4 sessions with none;
- the populate plan: `.claude/research/refresh_populate/populate_report.md` and
  `populate_matrix.csv`.

It is hand-maintained from then on.

**This doc is partly generated.** The rule, the reason codes, the counts and the
table of excluded and pending epochs sit between `<!-- BEGIN GENERATED -->` and
`<!-- END GENERATED -->` markers. They are rewritten from the module and the CSV by:

```bash
uv run python scripts/qa/write_epoch_register_doc.py           # rewrite them
uv run python scripts/qa/write_epoch_register_doc.py --check   # fail if stale
```

The rest is hand-written. `tests/unit/test_epoch_register_doc.py` fails when the
doc drifts from the register.

## The rule

<!-- BEGIN GENERATED: rule -->
> An epoch is analysed if and only if it is a run epoch (`^[0-9]+_r[0-9]+$`) of a
> regular spatial-bandit session in Spyglass (or of a listed first-exposure
> session), with parsed trials (`noPrePostTrialTimes`), and every decode input
> resolves to exactly one row: position map, position info and linearization, ripple
> band, CA1 ripple times, HPC marks. Every other run epoch of a project session is
> excluded with exactly one enumerated reason code and its evidence, in one register
> that every stage reads and the audit checks.
<!-- END GENERATED: rule -->

When several reasons hold for one epoch, the register records the first in the
rule's order: no data (`NO_NWB`, `NO_RUN_EPOCH`), then the task (`DEPLETION_TASK`),
then Spyglass (`NOT_INGESTED`, `NO_TRIAL_INTERVALS`), then the decode inputs
(`PENDING_POPULATE`, `MISSING_INPUT`). The evidence names the others. For example,
chimi20200303, chimi20200307, peanut20201207 and senor20201119 are depletion
sessions whose trial parsing is also incomplete: they are `DEPLETION_TASK`, so
`NO_TRIAL_INTERVALS` holds for 8 of the 12 sessions with incomplete trial parsing.

wilbur20210416 is not a depletion session. Its raw NWB describes it as "Spatial
bandit task (regular) - after depletion". It was never ingested, so it is
`NOT_INGESTED`, like wilbur20210401. That one is a regular session Spyglass ingested
only in part, and its re-ingestion is blocked.

The listed first-exposure session is wilbur20210326, admitted by
`FIRST_EXPOSURE_DESCRIPTIONS` in `data_loaders/constants.py`. Four epochs are an
animal's first exposure to the task (200 trials, p(R) 100 % then 50 %):
j1620210706_02_r1, senor20201027_02_r1, chimi20200212_02_r1 and
wilbur20210326_03_r1. All four are included. Only wilbur20210326's session
description, "Spatial bandit task (regular) - exploration session", needs the
allowlist.

## Statuses

| status | meaning |
|---|---|
| analysed | In the cohort: every stage processes the epoch. |
| pending | Not yet analysed; its reason is `PENDING_POPULATE`. Only a `--pending-only` run of the decoders, the SWR runner or the Widloski runner processes it. |
| excluded | Never analysed, for its reason. |

## Reason codes

<!-- BEGIN GENERATED: reason-codes -->
| code | status | definition |
|---|---|---|
| `NO_NWB` | excluded | No NWB file in either raw store and no Spyglass entry: there are no data. |
| `NO_RUN_EPOCH` | excluded | The session's NWB file holds no run epoch. |
| `DEPLETION_TASK` | excluded | A depletion session: a decaying reward schedule, a task variant, not the regular spatial bandit. |
| `NOT_INGESTED` | excluded | The session is not (fully) ingested in Spyglass. |
| `NO_TRIAL_INTERVALS` | excluded | Trial parsing failed: the epoch has no `noPrePostTrialTimes` interval. |
| `PENDING_POPULATE` | pending | A decode input is missing from Spyglass and a planned populate adds it; analysed once its decodes exist and its entry is updated. |
| `MISSING_INPUT` | excluded | A decode input is missing from Spyglass and no fix is planned. |
<!-- END GENERATED: reason-codes -->

## Counts

<!-- BEGIN GENERATED: counts -->
| status | reason | entries |
|---|---|---|
| analysed | — | 218 |
| excluded | `NO_NWB` | 3 |
| excluded | `NO_RUN_EPOCH` | 1 |
| excluded | `DEPLETION_TASK` | 238 |
| excluded | `NOT_INGESTED` | 15 |
| excluded | `NO_TRIAL_INTERVALS` | 42 |
| pending | `PENDING_POPULATE` | 37 |
| excluded | `MISSING_INPUT` | 3 |
| **total** | | **557** |

An entry is one run epoch, except the 4 session-wide `*` entries of sessions with no known run epoch: 553 run epochs and 4 session-wide entries in all.

| animal | analysed | pending |
|---|---|---|
| chimi | 45 | 1 |
| j16 | 58 | 0 |
| peanut | 13 | 32 |
| senor | 58 | 3 |
| wilbur | 44 | 1 |
| **all** | **218** | **37** |
<!-- END GENERATED: counts -->

Of the 91 `NWB_FILES`, 46 are depletion sessions. The other 45 hold every analysed
and pending epoch: 44 regular sessions and wilbur20210326. The other 18 sessions in
the register are not in `NWB_FILES`.

## How each stage reads it

`analysed_epochs()` returns `get_epoch_info()`'s rows whose register status is
`analysed`. `pending_epochs()` returns the `pending` ones alone. `epochs_to_run`
picks one of the two, by a stage's `--pending-only` flag.

Each raises `EpochRegisterError` when `get_epoch_info()` and the register disagree:

- `get_epoch_info()` lists an epoch the register does not;
- `get_epoch_info()` does not list an epoch of the requested status.

`pending_epochs()` also raises when the register holds no pending epoch.

The cohort is pinned in the register, not in Spyglass. A populate makes a pending
epoch decodable, and so lists it in `get_epoch_info()`. The epoch is analysed only
once its decodes exist and its entry is changed to `analysed`.

| stage | reads |
|---|---|
| NonLocal and ContFrag decoders (`decoding_common.run_batch`) | `epochs_to_run(args.pending_only)`. `--pending-only` refuses the canonical `Processed-Data` folder as `--save_dir`. |
| SWR runner (`run_swr_detection.py`) | `epochs_to_run(args.pending_only)`; no skip list of its own |
| Widloski runner (`run_memoryless_detection.py`, `--all` / `--animal`) | `epochs_to_run(args.pending_only)`. An epoch it selects with no decode under `--root` is run and recorded as failed. A decode of any other epoch is left out and recorded as skipped. `--epoch-id` runs what it names. |
| replay table builders (`prepare_data.find_sessions`, used by `prepare_replay_content_data.py`, `build_timebin_tables.py`, `regenerate_stop_table.py`) | `analysed_epochs()`; an analysed epoch with no decode stops the build |
| ContFrag table builders (`prepare_contfrag_replay_content_data.py`, `refresh_contfrag_behavioral_covariates.py`) | `analysed_epochs()`; for HPC an analysed epoch with no decode stops the build |
| PFC runner (`run_pfc_analysis_bandit.py`) | `analysed_epochs()`, then the PFC-level condition below |
| HPC raw stats (`scripts/qa/compute_hpc_raw_stats.py`) | `analysed_epochs()` |
| decode-folder readers (see below) | `analysed_decode_files()`, from the register alone |
| audit (`scripts/qa/audit_refresh_consistency.py`) | the register (`--register`), checked against every artifact and `get_epoch_info()` |

The decode-folder readers read decode files rather than Spyglass:

- `load_data_over_time` and `load_contfrag_data_over_time` (`analysis/hippocampus/` and
  `analysis/contfrag/data_aggregation.py`), used by Figures 2 and 3 and the decoder
  validation supplement;
- `run_state_evidence.py`;
- `run_quality_metrics.py`;
- `analyze_content_occupancy.py`;
- `analyze_event_spike_evidence.py`.

They take `analysed_decode_files(directory, suffix)`, which names the analysed
epochs' files instead of globbing, so a decode of any other epoch in the folder is
never read. The two `data_aggregation` loaders also require every analysed epoch's
file. The helper reads only the register, not the database.

`scripts/figures/generate_event_gallery.py` still globs the decode folder. It relies on
the canonical folder holding only analysed epochs' decodes, which the audit's
`decode_folder` check enforces.

`scripts/qa/measure_pfc_null_geometry.py` globs the PFC runner's `*_mPFC_modulation.pkl`
bundles, not the decode folder, so the `decode_folder` check does not cover it. It relies
on the PFC runner, which writes bundles only for the analysed PFC cohort (see "PFC
subset"). No check removes a bundle left in that folder from an earlier cohort.

`scripts/qa/check_data_availability.py` still reads `get_epoch_info()` itself. It
diagnoses which inputs load for every epoch Spyglass lists, pending ones included.

The audit runs these checks on the register:

- `epoch_sets`: every artifact holds exactly the analysed epochs. An absence passes
  only with a reason under `--exceptions`. An epoch an artifact holds but the
  register does not analyse fails, named with its status and reason. So does any
  unregistered epoch.
- `decode_folder`: the decode folder (`--root`) holds HPC NonLocal and ContFrag
  decode files only of analysed epochs. A file of a pending, excluded or unregistered
  epoch fails, even one of a partly saved decode.
- `epochs_registered`: every epoch `get_epoch_info()` lists is registered, and every
  analysed epoch is listed.
- `pending_epochs`: lists the pending epochs and their evidence, which
  `not_checked` names.

## Adding the pending epochs

1. The Spyglass populates (the controller's, from the populate plan).
2. Decode the pending epochs alone, into a staging folder, with both decoders:
   `run_decoding_bandit.py --pending-only --save_dir <staging>`, and the same for
   `run_decoding_contfrag.py`. The decoders refuse `--pending-only` into the
   canonical folder, so no reader sees a pending decode there.
3. Run the SWR runner with `--pending-only`. Run the Widloski runner with
   `--all --pending-only --root <staging>`. Under a run id whose settings match the
   main run's, each adds only the pending epochs to that run's manifest.

   From here until step 4, the audit's `epoch_sets` check reports the pending epochs
   in the SWR and Widloski manifests as epochs the register does not analyse. That is
   expected. It clears when step 4 sets their entries to `analysed`.
4. Check the staged decodes. Then move them into the canonical folder in the same
   change that sets their entries to `analysed`, with an empty reason. In that
   change, regenerate this doc with `scripts/qa/write_epoch_register_doc.py`.
5. Rebuild the tables and rerun the PFC runner and the audit. These read only
   analysed epochs, so they now include the new ones.

## PFC subset

The PFC cohort is the analysed epochs whose session has curated cortical spike
sorting (`CuratedSpikeSorting`, mountainsort4). An analysed epoch without that
sorting is left out of the PFC analysis at the PFC level. It is not an epoch
exclusion and has no register entry.

The PFC runner writes this out to `pfc_cohort_<brain_area>.json` in its output
folder, holding:

- the analysed and PFC cohort counts;
- the epochs it ran;
- each left-out epoch with that reason.

`ran` is per writer. Each invocation overwrites the file when it starts, with its own
sessions. A batch split across parallel partitions (`--sessions`) leaves the last
partition's share only. The union is the bundles on disk. The counts and `left_out`
are the same whichever partition writes last.

The mPFC and OFC ContFrag builds have their own cohort rule: an analysed epoch
with no decode for the area is outside that area's cohort. That rule is not
recorded.

## Changing the register

- **A populate succeeds.** Follow "Adding the pending epochs".
- **A planned populate fails.** Change the entry to `excluded` / `MISSING_INPUT`,
  with the failure as evidence, and regenerate this doc.
- **A new session or epoch appears in Spyglass.** `analysed_epochs()` raises until
  it is registered.

`read_register` refuses a malformed file:

- an epoch registered twice;
- a session-wide `*` entry beside epoch entries;
- an unknown status or reason;
- a reason that does not match its status;
- an unanalysed entry without evidence.

## Excluded and pending epochs

The source for the evidence is the exclusion audit and the populate plan.
`populate_plan.py` and `populate_report.md` are in
`.claude/research/refresh_populate/`. One row per session and evidence, its epochs
comma-separated, `*` for a session-wide entry. A literal `|` inside a cell is
written `\|`.

<!-- BEGIN GENERATED: entries -->
| session | epochs | status | reason | evidence |
|---|---|---|---|---|
| chimi20200302_.nwb | * | excluded | `NO_NWB` | No chimi20200302*.nwb in /stelmo/nwb/raw or /nimbus/nwb/raw2, and no Spyglass Nwbfile or Session row. Whether a raw .rec exists elsewhere is unknown. |
| peanut20201126_.nwb | * | excluded | `NO_NWB` | No peanut20201126*.nwb in /stelmo/nwb/raw or /nimbus/nwb/raw2, and no Spyglass Nwbfile or Session row. Whether a raw .rec exists elsewhere is unknown. |
| senor20201108_.nwb | * | excluded | `NO_NWB` | No senor20201108*.nwb in /stelmo/nwb/raw or /nimbus/nwb/raw2, and no Spyglass Nwbfile or Session row. Whether a raw .rec exists elsewhere is unknown. |
| chimi20200301_.nwb | * | excluded | `NO_RUN_EPOCH` | The raw NWB (/stelmo/nwb/raw/chimi20200301.nwb) holds one epoch, 01_s1 (sleep), so no run epoch exists; never ingested (Spyglass Session: 0 rows). |
| chimi20200303_.nwb | 02_r1, 04_r2 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; no alison_behav.TrialsInfoByEpoch row for the epoch. Also no noPrePostTrialTimes interval for any of the session's run epochs and no alison_behav.NoPrePostTrialIntervals row for the session (trial parsing incomplete; listed as excluded since c39f0e9). |
| chimi20200303_.nwb | 06_r3 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. Also no noPrePostTrialTimes interval for any of the session's run epochs and no alison_behav.NoPrePostTrialIntervals row for the session (trial parsing incomplete; listed as excluded since c39f0e9). |
| chimi20200304_.nwb | 02_r1, 04_r2 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| chimi20200305_.nwb | 02_r1, 04_r2, 06_r3 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| chimi20200306_.nwb | 02_r1, 04_r2, 06_r3 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| chimi20200307_.nwb | 02_r1 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. Also no noPrePostTrialTimes interval for any of the session's run epochs and no alison_behav.NoPrePostTrialIntervals row for the session (trial parsing incomplete; listed as excluded since c39f0e9). |
| chimi20200307_.nwb | 04_r2, 06_r3 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; no alison_behav.TrialsInfoByEpoch row for the epoch. Also no noPrePostTrialTimes interval for any of the session's run epochs and no alison_behav.NoPrePostTrialIntervals row for the session (trial parsing incomplete; listed as excluded since c39f0e9). |
| chimi20200308_.nwb | 02_r1, 04_r2, 06_r3 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| chimi20200310_.nwb | 02_r1, 04_r2 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| chimi20200311_.nwb | 02_r1, 04_r2, 06_r3, 08_r4 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| chimi20200312_.nwb | 02_r1, 04_r2, 06_r3, 08_r4 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| chimi20200313_.nwb | 02_r1, 04_r2, 06_r3, 08_r4 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| j1620210714_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5, 12_r6 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| j1620210715_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| j1620210716_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| j1620210717_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| j1620210718_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| j1620210719_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| j1620210720_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| j1620210721_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5, 12_r6 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| peanut20201203_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| peanut20201204_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| peanut20201205_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| peanut20201206_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| peanut20201207_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5, 12_r6 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; no alison_behav.TrialsInfoByEpoch row for the epoch. Also no noPrePostTrialTimes interval for any of the session's run epochs and no alison_behav.NoPrePostTrialIntervals row for the session (trial parsing incomplete; listed as excluded since c39f0e9). |
| peanut20201208_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| peanut20201209_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| senor20201109_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| senor20201110_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| senor20201111_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| senor20201112_.nwb | 02_r1, 03_r2, 05_r3, 07_r4, 09_r5, 11_r6 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| senor20201113_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| senor20201114_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| senor20201115_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| senor20201116_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| senor20201117_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 09_r5, 11_r6 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| senor20201118_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| senor20201119_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; no alison_behav.TrialsInfoByEpoch row for the epoch. Also no noPrePostTrialTimes interval for any of the session's run epochs and no alison_behav.NoPrePostTrialIntervals row for the session (trial parsing incomplete; listed as excluded since c39f0e9). |
| senor20201120_.nwb | 02_r1, 04_r2, 06_r3, 07_r4, 09_r5, 11_r6 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| senor20201121_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| wilbur20210404_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. Also no CA1 RippleTimesV1 (only group sb_wilbur_ripple); decoding never attempted it. |
| wilbur20210405_.nwb | 02_r1, 04_r2, 06_r3, 10_r5, 12_r6 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| wilbur20210405_.nwb | 08_r4 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; no alison_behav.TrialsInfoByEpoch row for the epoch. |
| wilbur20210406_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| wilbur20210407_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| wilbur20210408_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| wilbur20210409_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| wilbur20210410_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| wilbur20210411_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| wilbur20210412_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| wilbur20210413_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| wilbur20210414_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| wilbur20210415_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `DEPLETION_TASK` | Session.session_description 'Spatial bandit task (depletion)'; alison_behav.TrialsInfoByEpoch decay_percent=80. |
| wilbur20210401_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5, 12_r6, 14_r7, 16_r8 | excluded | `NOT_INGESTED` | Partly ingested, and re-ingestion is blocked. The raw NWB (/nimbus/nwb/raw2/wilbur20210401.nwb) is a regular session with 8 run epochs, but Spyglass holds only its Nwbfile row (no Session, IntervalList or trials) and its tracked copy /stelmo/nwb/raw/wilbur20210401_.nwb is missing. Blocked: re-ingestion needs a destructive reinsert (or a hash-identical regenerated copy), Comrie's alison_behav trial parsing, and a refit of Comrie's RL model, without which wilbur's q-value coverage falls below the 90 % gate (populate_report.md B1-B3). |
| wilbur20210416_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5, 12_r6, 14_r7 | excluded | `NOT_INGESTED` | Never ingested (Spyglass Session: 0 rows). Its raw NWB (/stelmo/nwb/raw/wilbur20210416.nwb) session_description is 'Spatial bandit task (regular) - after depletion', the session run after the depletion block, which get_epoch_info would not admit by its description even if it were ingested. |
| chimi20200215_.nwb | 02_r1, 04_r2, 06_r3, 07_r3 | excluded | `NO_TRIAL_INTERVALS` | Trial parsing incomplete: no noPrePostTrialTimes interval for any of the session's run epochs and no alison_behav.NoPrePostTrialIntervals row for the session (experimenter-confirmed, commit c39f0e9). Session.session_description 'Spatial bandit task (regular)'. |
| chimi20200219_.nwb | 02_r1, 04_r2, 06_r3 | excluded | `NO_TRIAL_INTERVALS` | Trial parsing incomplete: no noPrePostTrialTimes interval for any of the session's run epochs and no alison_behav.NoPrePostTrialIntervals row for the session (experimenter-confirmed, commit c39f0e9). Session.session_description 'Spatial bandit task (regular)'. |
| chimi20200226_.nwb | 02_r1, 04_r2, 06_r3 | excluded | `NO_TRIAL_INTERVALS` | Trial parsing incomplete: no noPrePostTrialTimes interval for any of the session's run epochs and no alison_behav.NoPrePostTrialIntervals row for the session (experimenter-confirmed, commit c39f0e9). Session.session_description 'Spatial bandit task (regular)'. |
| peanut20201124_.nwb | 02_r1, 04_r2, 06_r3, 07_r4, 09_r5 | excluded | `NO_TRIAL_INTERVALS` | Trial parsing incomplete: no noPrePostTrialTimes interval for any of the session's run epochs and no alison_behav.NoPrePostTrialIntervals row for the session (experimenter-confirmed, commit c39f0e9). Session.session_description 'Spatial bandit task (regular)'. |
| peanut20201130_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5, 12_r6, 14_r7, 16_r8 | excluded | `NO_TRIAL_INTERVALS` | Trial parsing incomplete: no noPrePostTrialTimes interval for any of the session's run epochs and no alison_behav.NoPrePostTrialIntervals row for the session (experimenter-confirmed, commit c39f0e9). Session.session_description 'Spatial bandit task (regular)'. |
| senor20201029_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5 | excluded | `NO_TRIAL_INTERVALS` | Trial parsing incomplete: no noPrePostTrialTimes interval for any of the session's run epochs and no alison_behav.NoPrePostTrialIntervals row for the session (experimenter-confirmed, commit c39f0e9). Session.session_description 'Spatial bandit task (regular)'. |
| senor20201031_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5, 12_r6, 14_r7 | excluded | `NO_TRIAL_INTERVALS` | Trial parsing incomplete: no noPrePostTrialTimes interval for any of the session's run epochs and no alison_behav.NoPrePostTrialIntervals row for the session (experimenter-confirmed, commit c39f0e9). Session.session_description 'Spatial bandit task (regular)'. |
| wilbur20210327_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5, 12_r6, 14_r7 | excluded | `NO_TRIAL_INTERVALS` | Trial parsing incomplete: no noPrePostTrialTimes interval for any of the session's run epochs and no alison_behav.NoPrePostTrialIntervals row for the session (experimenter-confirmed, commit c39f0e9). Session.session_description 'Spatial bandit task (regular)'. |
| chimi20200212_.nwb | 02_r1 | pending | `PENDING_POPULATE` | First-exposure session (200 trials, p(R) 100% then 50%). No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on chimi20200217, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 1 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201125_.nwb | 02_r1 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 1 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201125_.nwb | 04_r2 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 3 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201125_.nwb | 06_r3 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 5 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201125_.nwb | 08_r4 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 7 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201125_.nwb | 10_r5 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 9 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201125_.nwb | 12_r6 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 11 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201125_.nwb | 14_r7 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 13 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201127_.nwb | 02_r1 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 1 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201127_.nwb | 04_r2 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 3 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201127_.nwb | 06_r3 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 5 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201127_.nwb | 08_r4 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 7 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201127_.nwb | 10_r5 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 9 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201127_.nwb | 12_r6 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 11 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201128_.nwb | 02_r1 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 1 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201128_.nwb | 04_r2 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 3 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201128_.nwb | 08_r4 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 7 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201128_.nwb | 10_r5 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 9 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201128_.nwb | 12_r6 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 11 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201128_.nwb | 14_r7 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 13 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201128_.nwb | 16_r8 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 15 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201129_.nwb | 02_r1 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201129, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 1 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201129_.nwb | 04_r2 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201129, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 3 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201129_.nwb | 08_r4 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201129, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 7 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201129_.nwb | 12_r6 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201129, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 11 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201129_.nwb | 14_r7 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201129, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 13 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201129_.nwb | 16_r8 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201129, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 15 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201201_.nwb | 02_r1 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 1 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201201_.nwb | 04_r2 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 3 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201201_.nwb | 10_r5 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 9 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201202_.nwb | 08_r4 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 7 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201202_.nwb | 12_r6 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 11 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| peanut20201202_.nwb | 14_r7 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on peanut20201124, LFPBandV1 ripple and theta, 21 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 13 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| senor20201102_.nwb | 08_r4 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on senor20201030, LFPBandV1 ripple and theta, 22 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 7 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| senor20201104_.nwb | 04_r2 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on senor20201030, LFPBandV1 ripple and theta, 22 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 10 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| senor20201105_.nwb | 02_r1 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on senor20201030, LFPBandV1 ripple and theta, 22 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 1 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| wilbur20210326_.nwb | 03_r1 | pending | `PENDING_POPULATE` | First-exposure session (200 trials, p(R) 100% then 50%). Its description, 'Spatial bandit task (regular) - exploration session', is admitted by FIRST_EXPOSURE_DESCRIPTIONS. No CA1 RippleLFPSelection or RippleTimesV1; every other input exists. Planned fix (populate_plan.py phases 5-6): CA1 RippleLFPSelection on the 14-electrode ripple band (decision D2), RippleTimesV1. |
| senor20201104_.nwb | 06_r3, 08_r4 | excluded | `MISSING_INPUT` | LFPV1.make fails in spyglass common_interval.censor ('Interval_list must contain all timestamps') for the existing LFPSelection (populate 2026-10-09), so no LFP, ripple band or CA1 ripple times; likely why LFPV1 was never made |
| senor20201105_.nwb | 09_r4 | excluded | `MISSING_INPUT` | IntervalPositionInfo head_position spatial series empty for 'pos 8 valid times' (IntervalLinearizedPosition.make fails, populate 2026-10-09); also no CA1 RippleTimesV1 |
<!-- END GENERATED: entries -->
