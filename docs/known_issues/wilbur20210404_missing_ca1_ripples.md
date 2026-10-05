# wilbur20210404: a depletion session with no CA1 ripple extraction

**Status:** not an exclusion of its own. wilbur20210404 is a **depletion** session,
never in the cohort. Its 5 run epochs (02_r1, 04_r2, 06_r3, 08_r4, 10_r5) are
`excluded` / `DEPLETION_TASK` in the epoch register
([../epoch_register.md](../epoch_register.md)). Corrected on 2026-10-05 by the
exclusion audit (`.claude/research/refresh_exclusions/exclusions_report.md`,
section 3.4). The earlier version of this note, written 2026-07-10, named the wrong
mechanism.

## Why it is not analysed

- `Session.session_description` is "Spatial bandit task (depletion)". Comrie's
  `alison_behav.TrialsInfoByEpoch` gives `decay_percent` = 80, a decaying reward
  schedule.
- `get_epoch_info` has admitted only regular sessions since the repository's first
  commit, so it never listed this session.
- Decoding therefore never ran on it. It has no decode files, logs or table rows,
  in April or in October 2026.
- The earlier claim was that `run_decode` caught a `get_ripple` error and silently
  dropped these epochs. That never happened in this repository.

## The ripple fact, which still holds

wilbur20210404 is the only project session whose ripple times exist only under
`group_name = "sb_wilbur_ripple"`, with no `CA1` extraction (`RippleTimesV1`: 11
rows). That is a different channel and reference set from the `CA1` extraction
every analysed epoch uses. `get_ripple` (`data_loaders/lfp.py`) restricts on
`group_name = "CA1"`, and it raises a `ValueError` naming the group names present
when no single CA1 row matches.

If depletion sessions were ever brought into the analysis, this session would also
need a matching CA1 `RippleTimesV1` extraction. Using `sb_wilbur_ripple` would mix
ripple definitions.
