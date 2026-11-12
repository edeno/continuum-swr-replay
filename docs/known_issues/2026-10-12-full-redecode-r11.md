# The canonical decodes were replaced twice in October 2026

**Status:** done and audited. The final audit passed on 2026-10-12 (17 of 17 checks;
[report](../../.claude/docs/reviews/2026-10-12-full-pipeline-refresh.md), section 1). Open: a number
written before the refresh comes from the April decodes, or from tables built on them. It is stale
until it is rerun or requoted (see [What is stale until rerun](#what-is-stale-until-rerun)).
**Dates:** first swap 2026-10-05; second swap 2026-10-10 18:33 to 2026-10-11 00:17.
**Severity:** high for any quoted number. Every non-local event changed, and so did every
`event_id`. Exact-parity checks against earlier tables cannot pass.

## What happened

| when | canonical decodes after the swap | where the replaced files went |
|---|---|---|
| 2026-04-01 | the April decodes, commit `0820d39d`, 218 epochs | (the starting point) |
| 2026-10-05 | the first re-decode ("R1"), launch commit `643bbe0`, 218 epochs | `Processed-Data/_pre_redecode_backup_20261003/` |
| 2026-10-10 to 2026-10-11 | the R11 decodes, commit `f62b6ce`, 242 epochs | `Processed-Data/_pre_r11_backup_20261010/` |

Each swap renamed files inside one filesystem. Inode, size and modification time were checked
to be unchanged. Each backup has a `SHA256SUMS` file for the files moved into it.

The canonical decodes, tables and refits now rest on the R11 decodes. The outputs listed under
[What is stale until rerun](#what-is-stale-until-rerun) do not. The design is in
[2026-10-03-full-pipeline-refresh-design.md](../design/2026-10-03-full-pipeline-refresh-design.md).

## The backups

Both backups are kept. The refresh's cleanup list leaves them out
([report](../../.claude/docs/reviews/2026-10-12-full-pipeline-refresh.md), section 8).

- **`Processed-Data/_pre_redecode_backup_20261003/`** holds the April decodes of 218 epochs:
  1,744 files, 8 per epoch (NonLocal and ContFrag `results.nc`, `detector.pkl`,
  `environment.pkl` and `data_over_time.pkl`). It also holds the copy taken before the refresh of
  the logs, `replay_content`, `contfrag_replay_content` and the figures. `SHA256SUMS` covers the
  April files. `STAGED_SHA256SUMS` covers the first re-decode's files that replaced them.
  `MISSING_APRIL_DECODES.md` names the one epoch whose April files are not in it (see below).
- **`Processed-Data/_pre_r11_backup_20261010/`** holds the first re-decode's files that were
  canonical from 2026-10-05 to 2026-10-10: 1,744 files for the same 218 epochs. `SHA256SUMS` covers
  them. `STAGED_SHA256SUMS` covers the 1,936 R11 files that replaced them (242 epochs, 8
  each). `EPOCHS.txt` is the swap plan (207 analysed epochs and 35 pending ones going in, 218 old
  ones going out, 13 `s_block` epochs staying out). `swap_actions.jsonl` is the swap log.
- **The 13 `SUSTAINED_NOISE` epochs.** Their R11 decodes (104 files) stay in
  `Processed-Data/_staging_redecode_20261010/`. They are not canonical.

## What changed

The April decode and the R11 decode differ in the inputs below
([design](../design/2026-10-03-full-pipeline-refresh-design.md), "What the re-decode changes",
for the April values).

| change | first in |
|---|---|
| Coincident-spike fix: the coincidence count now includes each cluster's leading spike (`64bf585`). About 0.07 % more HPC marks are dropped (259 of about 358,000 on chimi20200225_02_r1) | first re-decode |
| P30-B discrete transitions, pinned in `decoding/transitions.py` (R10; [note](../design/2026-10-04-decoder-transition-prior.md)). The library's defaults are not used | first re-decode |
| Local state is a 2.5 cm Gaussian, not the exact position (`ded292e`) | first re-decode |
| Non-local events need 7 bins at the 500 Hz grid, not 9 (R9; `minimum_sample_count(time, 0.015)` is 7 on every epoch) | first re-decode |
| Artifact bins are missing data: `is_missing` is set, so the bin's log-likelihood is 0 | R11 |
| Encoding training leaves out the Kay inventory's events and the artifact mask. Before, it left out Spyglass's stored ripples | R11 |
| The SWR source, for the decode and for every analysis, is the Kay 2.0 inventory, run `056eee0dcb5a`, with tetrode rule R-T and a mask-excluded normalisation. It replaces Spyglass's `SWR_event` | R11 |
| The artifact mask is artifact run `fbbff144a123` | R11 |

Other changes since April, from the same design table: tetrode-only HPC marks (`f172084`),
position trimmed to the valid interval (`e5a0def`), `non_local_detector` at `2d730c3`, and
`ripple_detection` pinned at `68e1784`.

Each R11 decode records both run ids in `attrs["decode_masks"]` and the commit in
`attrs["provenance"]`.

## The cohort

The cohort is the epoch register's ([../epoch_register.md](../epoch_register.md)):

- **242 analysed epochs** in 45 NWB files. That is 207 of the earlier 218, plus 35 epochs that
  were not in the earlier cohort and were decoded in the refresh.
- **13 epochs are excluded as `SUSTAINED_NOISE`.** The artifact detector finds an S block
  (sustained ripple-band noise) in each. 11 of them were in the earlier 218.
- 255 epochs were decoded in all. The SWR run `056eee0dcb5a` holds all 255.

## Effect on the results

The size and the cause of each change are in the
[audit report](../../.claude/docs/reviews/2026-10-12-full-pipeline-refresh.md), section 3. In
short:

- The code changed no fitted result. With the same decodes and SWR labels, current code rebuilds
  the golden tables and rate fits exactly. It changed two descriptives: the content-quadrant
  definition and the gallery's count of SWRs that do not overlap a non-local event.
- The SWR source (rule R-T with the mask normalisation) and the 35 added epochs move the SWR and
  non-SWR reward rate ratios.
- The R11 decode loses short, mostly non-SWR events with own-arm content. It moves the
  SWR × current-arm content term (CS1), the current-arm main effect (C1) and the
  Dirichlet-multinomial `log_phi_is_swr`.
- Several earlier "before" values came from older scripts, not from the pre-refresh bundles
  (report, section 3, caveat 1).

Event ids are not stable across decodes. An old-to-new comparison must match events on
(epoch, `event_start_s`, `event_end_s`), not on `event_id`.

## What is stale until rerun

These are the items in the report's section 5 that name an output that was not rerun on the R11
decodes (item numbers are the report's):

- **Hand-maintained docs.** 78 claims quote pre-refresh numbers; 8 change a stated conclusion
  (item 1, report section 6). They wait for the project lead's review of the report.
- **Held-out spike evidence per decoder state (Task 12).** Only a five-epoch pilot on pre-R11
  decodes exists. Do not quote it as a result of the refresh (item 36).
- **Half-tetrode held-out labels (Task 13).** The 20 GPU decodes were never run (item 37).
- **R13 robustness and sensitivity reruns.** Not run: the sensitivity without epochs over 2 %
  masked, the masked-time table, and the reruns without the mask, with every artifact threshold
  25 % lower and higher, and with 1 s masked after each step (item 38).
- **Spike-evidence ("reality") outputs.** They predate the R11 decodes.
  `paper/figures/reality/event_spike_evidence_per_animal.csv` still labels j16 as "j"
  (item 39).
- **PFC sigma-sensitivity bundles** (item 20) and **`pfc_marked_model_family/`** (item 29).

## One epoch has no April decode

The eight April files of `chimi20200212_02_r1_HPC` were overwritten with placeholders on
2026-10-03 at 14:09 by a mutation-testing run during Task 2. The project lead accepted the loss.
The placeholders were renamed `*.placeholder_20261003` in `Processed-Data/`. A ZFS snapshot from
2026-10-03 at 00:05 (`/cumulus/.zfs/snapshot/20261003`) predates the damage.

The epoch was not in the earlier 218. It is a first-exposure epoch that was added in the
refresh, and the report's old-to-new comparisons use the epochs both cohorts share, so none
of them involves it. `MISSING_APRIL_DECODES.md` says its April-decode tables are intact in the
golden baseline.

## Also recorded here

33 at-well stops in `chimi20200218_06_r3` disagree on reward between `qvals_table` and the stop
table. They did so before the refresh too, in the old and the new tables alike (report, section 5,
item 42). The report gives no cause.
