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
in it). It was generated once, on 2026-10-05, from the exclusion audit
(`.claude/research/refresh_exclusions/exclusions_report.md` and
`exclusions_table.csv`, 557 run epochs, local) and the populate plan
(`.claude/research/refresh_populate/populate_report.md` and `populate_matrix.csv`,
local). It is hand-maintained from then on.

This doc mirrors the module and the CSV. `tests/unit/test_epoch_register_doc.py`
fails when they drift. Its contract: the rule is the blockquote under "The rule";
"Reason codes" holds one row per code; "Counts" holds one row per status and reason,
then the total; "Excluded and pending epochs" holds one row per session and
evidence, its epochs comma-separated. A literal `|` inside a cell is written `\|`.

## The rule

> An epoch is analysed if and only if it is a run epoch (`^[0-9]+_r[0-9]+$`) of a
> regular spatial-bandit session in Spyglass (or of a listed first-exposure
> session), with parsed trials (`noPrePostTrialTimes`), and every decode input
> resolves to exactly one row: position map, position info and linearization,
> ripple band, CA1 ripple times, HPC marks. Every other run epoch of a project
> session is excluded with exactly one enumerated reason code and its evidence, in
> one register that every stage reads and the audit checks.

When several reasons hold for one epoch, the register records the first in the
rule's order: no data (`NO_NWB`, `NO_RUN_EPOCH`), then the task (`DEPLETION_TASK`),
then Spyglass (`NOT_INGESTED_BLOCKED`, `NO_TRIAL_INTERVALS`), then the decode inputs
(`PENDING_POPULATE`, `MISSING_INPUT`). The evidence names the others. For example,
chimi20200303, chimi20200307, peanut20201207 and senor20201119 are depletion
sessions whose trial parsing is also incomplete: they are `DEPLETION_TASK`, so
`NO_TRIAL_INTERVALS` holds for 8 of the 12 sessions with incomplete trial parsing.

The listed first-exposure session is wilbur20210326, admitted by
`FIRST_EXPOSURE_DESCRIPTIONS` in `data_loaders/constants.py`. Four epochs are an
animal's first exposure to the task (200 trials, p(R) 100 % then 50 %):
j1620210706_02_r1, senor20201027_02_r1, chimi20200212_02_r1 and
wilbur20210326_03_r1. All four are included. The first two are analysed. The last
two are pending a populate. Only wilbur20210326's session description, "Spatial
bandit task (regular) - exploration session", needs the allowlist.

## Statuses

| status | meaning |
|---|---|
| analysed | In the cohort: every stage processes the epoch. |
| pending | Not yet analysed; its reason is `PENDING_POPULATE`. Only the decoders, given `--include-pending`, process it. |
| excluded | Never analysed, for its reason. |

## Reason codes

| code | status | definition |
|---|---|---|
| `NO_NWB` | excluded | No NWB file in either raw store and no Spyglass entry: there are no data. |
| `NO_RUN_EPOCH` | excluded | The session's NWB file holds no run epoch. |
| `DEPLETION_TASK` | excluded | A depletion session (a decaying reward schedule) or the session run after the depletion block: a task variant, not the regular spatial bandit. |
| `NOT_INGESTED_BLOCKED` | excluded | A regular session that Spyglass ingested only in part, whose re-ingestion is blocked. |
| `NO_TRIAL_INTERVALS` | excluded | Trial parsing failed: the epoch has no `noPrePostTrialTimes` interval. |
| `PENDING_POPULATE` | pending | A decode input is missing from Spyglass and a planned populate adds it; analysed once its decodes exist and its entry is updated. |
| `MISSING_INPUT` | excluded | A decode input is missing from Spyglass and no fix is planned. |

## Counts

| status | reason | epochs |
|---|---|---|
| analysed | — | 218 |
| excluded | `NO_NWB` | 3 |
| excluded | `NO_RUN_EPOCH` | 1 |
| excluded | `DEPLETION_TASK` | 245 |
| excluded | `NOT_INGESTED_BLOCKED` | 8 |
| excluded | `NO_TRIAL_INTERVALS` | 42 |
| pending | `PENDING_POPULATE` | 40 |
| **total** | | **557** |

The 218 analysed epochs are the epochs decoded in the October 2026 re-decode: chimi
45, j16 58, peanut 13, senor 58, wilbur 44. The 40 pending epochs are populate
groups G1 to G4 of the populate plan. With them, the cohort would be 258. Group G5,
wilbur20210401, is `NOT_INGESTED_BLOCKED`.

Of the 91 `NWB_FILES`, 46 are depletion sessions (221 run epochs). The other 45
hold the 218 analysed and 40 pending epochs: 44 regular sessions and wilbur20210326.
Today 42 of the 44 regular sessions contribute analysed epochs. chimi20200212 and
peanut20201125 have no PositionIntervalMap row for any run epoch. The other 18
sessions in the register (74 run epochs and 4 session-wide entries) are not in
`NWB_FILES`.

## How each stage reads it

`analysed_epochs(include_pending=False)` returns `get_epoch_info()`'s rows whose
register status is `analysed`, plus `pending` when asked. It raises
`EpochRegisterError` in two cases:

- `get_epoch_info()` lists an epoch the register does not;
- `get_epoch_info()` does not list an epoch the register analyses, or (with
  `include_pending`) one it holds pending.

The cohort is pinned in the register, not in Spyglass. A populate makes a pending
epoch decodable, and so lists it in `get_epoch_info()`. The epoch is analysed only
once its decodes exist and its entry is changed to `analysed`.

| stage | reads |
|---|---|
| NonLocal and ContFrag decoders (`decoding_common.run_batch`) | `analysed_epochs(include_pending=args.include_pending)`; `--include-pending` decodes the pending epochs after their populate |
| SWR runner (`run_swr_detection.py`) | `analysed_epochs()`; no skip list of its own |
| Widloski runner (`run_memoryless_detection.py`, `--all` / `--animal`) | `analysed_epochs()`. An analysed epoch with no decode is run and recorded as failed. A decode of an epoch not analysed is left out and recorded as skipped. `--epoch-id` runs what it names. |
| replay table builders (`prepare_data.find_sessions`, used by `prepare_replay_content_data.py`, `build_timebin_tables.py`, `regenerate_stop_table.py`) | `analysed_epochs()`; an analysed epoch with no decode stops the build |
| ContFrag table builders (`prepare_contfrag_replay_content_data.py`, `refresh_contfrag_behavioral_covariates.py`) | `analysed_epochs()`; for HPC an analysed epoch with no decode stops the build |
| PFC runner (`run_pfc_analysis_bandit.py`) | `analysed_epochs()`, then the PFC-level condition below |
| HPC raw stats (`scripts/qa/compute_hpc_raw_stats.py`) | `analysed_epochs()` |
| audit (`scripts/qa/audit_refresh_consistency.py`) | the register (`--register`), checked against every artifact and `get_epoch_info()` |

`scripts/qa/check_data_availability.py` still reads `get_epoch_info()` itself. It
diagnoses which inputs load for every epoch Spyglass lists, pending ones included.

The audit runs three checks on the register:

- `epoch_sets`: every artifact holds exactly the analysed epochs. An absence passes
  only with a reason under `--exceptions`. An epoch an artifact holds but the
  register does not analyse fails, named with its status and reason. So does any
  unregistered epoch.
- `epochs_registered`: every epoch `get_epoch_info()` lists is registered, and every
  analysed epoch is listed.
- `pending_epochs`: lists the pending epochs and their evidence, which
  `not_checked` names.

## PFC subset

The PFC cohort is the analysed epochs whose session has curated cortical spike
sorting (`CuratedSpikeSorting`, mountainsort4): 71 of the 218, as the current mPFC
bundles hold them. An analysed epoch without that sorting is left out of the PFC analysis at the
PFC level. The runner prints the epoch with that reason. It is not an epoch
exclusion and has no register entry. The same holds for the mPFC and OFC ContFrag
builds: an analysed epoch with no decode for the area is outside that area's
cohort.

## Changing the register

- **A populate succeeds.** Decode the epoch (`--include-pending`) and check its
  decodes. Then change its entry to `analysed` with an empty reason, and update
  this doc. The stages pick it up from then on.
- **A planned populate fails.** Change the entry to `excluded` / `MISSING_INPUT`,
  with the failure as evidence.
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
`.claude/research/refresh_populate/`.

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
| wilbur20210416_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5, 12_r6, 14_r7 | excluded | `DEPLETION_TASK` | Raw NWB (/stelmo/nwb/raw/wilbur20210416.nwb) session_description 'Spatial bandit task (regular) - after depletion', the session run after the depletion block; never ingested (Spyglass Session: 0 rows). |
| wilbur20210401_.nwb | 02_r1, 04_r2, 06_r3, 08_r4, 10_r5, 12_r6, 14_r7, 16_r8 | excluded | `NOT_INGESTED_BLOCKED` | The raw NWB (/nimbus/nwb/raw2/wilbur20210401.nwb) is a regular session with 8 run epochs, but Spyglass holds only its Nwbfile row (no Session, IntervalList or trials) and its tracked copy /stelmo/nwb/raw/wilbur20210401_.nwb is missing. Blocked: re-ingestion needs a destructive reinsert (or a hash-identical regenerated copy), Comrie's alison_behav trial parsing, and a refit of Comrie's RL model, without which wilbur's q-value coverage falls below the 90 % gate (populate_report.md B1-B3). |
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
| senor20201104_.nwb | 06_r3 | pending | `PENDING_POPULATE` | LFPV1 was never populated for its LFPSelection (lfp_tets_senor, '06_r3 noPrePostTrialTimes'), so it has no LFPBandV1 ripple or theta band and no CA1 RippleTimesV1; its decode failed at get_ripple (logs/senor20201104_06_r3_HPC.log). Planned fix (populate_plan.py phases 3-6): LFPV1, LFPBandV1 ripple and theta, CA1 RippleLFPSelection, RippleTimesV1. |
| senor20201104_.nwb | 08_r4 | pending | `PENDING_POPULATE` | LFPV1 was never populated for its LFPSelection (lfp_tets_senor, '08_r4 noPrePostTrialTimes'), so it has no LFPBandV1 ripple or theta band and no CA1 RippleTimesV1; its decode failed at get_ripple (logs/senor20201104_08_r4_HPC.log). Planned fix (populate_plan.py phases 3-6): LFPV1, LFPBandV1 ripple and theta, CA1 RippleLFPSelection, RippleTimesV1. |
| senor20201105_.nwb | 02_r1 | pending | `PENDING_POPULATE` | No PositionIntervalMap row, so get_epoch_info cannot list it, and no CA1 RippleLFPSelection or RippleTimesV1. Every other input exists: IntervalPositionInfo default_decoding, IntervalLinearizedPosition on senor20201030, LFPBandV1 ripple and theta, 22 HPC UnitMarks sort groups, DIO, Comrie's trials. Planned fix (populate_plan.py phases 1, 5, 6): PositionIntervalMap populate (predicted 'pos 1 valid times'), CA1 RippleLFPSelection, RippleTimesV1. |
| senor20201105_.nwb | 09_r4 | pending | `PENDING_POPULATE` | IntervalLinearizedPosition was never populated for its selection ('pos 8 valid times', default_decoding, track graph senor20201030), so get_position_info falls back to a TrackGraph 'senor20201105' that does not exist; its CA1 RippleTimesV1 is also unpopulated although its CA1 RippleLFPSelection exists. Planned fix (populate_plan.py phases 2 and 6): IntervalLinearizedPosition, RippleTimesV1. |
| wilbur20210326_.nwb | 03_r1 | pending | `PENDING_POPULATE` | First-exposure session (200 trials, p(R) 100% then 50%). Its description, 'Spatial bandit task (regular) - exploration session', is admitted by FIRST_EXPOSURE_DESCRIPTIONS. No CA1 RippleLFPSelection or RippleTimesV1; every other input exists. Planned fix (populate_plan.py phases 5-6): CA1 RippleLFPSelection on the 14-electrode ripple band (decision D2), RippleTimesV1. |
