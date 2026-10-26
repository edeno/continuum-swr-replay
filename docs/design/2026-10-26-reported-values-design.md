# Reported values: every number read from a computed file

Date: 2026-10-26. Status: Part 2 built (`paper/reported/part2.json`, `scripts/reported/`); Parts 1, 3 and 4 not yet.

## Why

Numbers in this project get typed by hand, often by an AI assistant: into the claims table, the results docs,
figure captions and chat replies. Typed numbers drift from the analyses and sometimes are wrong or made up. A
recent example is claim 19a's spike-evidence "≈ 0.73", which comes from a one-session prototype and was never
updated after the 242-epoch run. The lead wants every reported number to come from a file that a script
computed, so that nothing can be stated that was not computed.

The model is [edeno/statespacecheck-paper](https://github.com/edeno/statespacecheck-paper):
- each figure's analysis writes a summary JSON at full precision;
- that repository's emitter script (`emit_reported_values.py`) turns the summaries into `manuscript/reported_values.tex`, a file of
  `\newcommand` macros;
- the manuscript types no numbers;
- all rounding lives in one module (`reported_values.py` and `number_format.py`), and a value that cannot be
  printed at the requested precision without losing information raises (`_exact`).

This repository already has a small version of the pattern: `scripts/qa/write_epoch_register_doc.py` fills the
`<!-- BEGIN GENERATED: name -->` … `<!-- END GENERATED: name -->` blocks of `docs/epoch_register.md` from the
register.

## Goals

- Every number reported anywhere in the repository comes from one of four summary files, one per part of the
  paper. The parts are those of [docs/analysis_map.md](../analysis_map.md) and
  [paper/claims_table.md](../../paper/claims_table.md):
  1. the model;
  2. the events;
  3. behavior;
  4. other brain areas.
- Each value says where it is used: paper, supplement, a figure, or several of these.
- One emitter writes every place a number appears:
  - `paper/reported_values.tex` for the manuscript (LaTeX, in this repository; not started yet);
  - the generated blocks of the claims table and the results docs;
  - the numbers that figure captions use.
- The claims table is the list of what gets reported: each claim row's numbers are values in its part's summary.
- In chat, the assistant quotes a value from a summary file and names the file and the key. A number not in a
  summary is described as not computed, never worked out on the spot.

## Non-goals

- **No tests for now.** The lead deferred them. The design should keep them easy to add later, for example
  "the emitted files match the summaries" and "a summary is older than its inputs".
- No re-running of fits or decodes. Summaries read outputs the analyses already wrote. Where a reported quantity
  needs a new computation (the first case is below), that is an analysis change made on its own and then read
  by the summary.
- No finer split than the four parts (no per-subsection files).

## Anti-goals

- **No silent overwrite.** When a typed number in a doc disagrees with the computed value, the computed value
  replaces it (decided by the lead), and every such replacement is listed for the lead: the claim, the old
  number, the new number and its source. A disagreement means one of them was wrong, so the lead must see each
  one.
- **No second home for a number.** A value is computed in one summarize step only, not again in a figure script
  or doc writer. Figure captions read the summary.
- **No rounding outside the emitter.**

## The pieces

### 1. Summary files, one per part

- **Content:** the reported values at full precision. Each value carries:
  - its key;
  - a short description;
  - its unit or kind (count, fraction, ratio, nats, cm, …);
  - the claims it supports (claims-table numbers);
  - where it is used (paper, supplement, figure name);
  - its source: the run id, table or file it was read from, and the selection (column, population, detector)
    that produced it.
- **File provenance:** the code commit and whether the tree was dirty, and each input file's path with a content
  hash or size and modification time.
- **Location and committing:** builder's call (below). The lead's goal implies the manuscript and the docs can
  be regenerated from the summaries without `Processed-Data/`, as statespacecheck-paper commits its summaries.
- **Who writes them:** one summarize step per part, reading existing outputs. Shared code goes in the package,
  not in scripts (see the import rules in `scripts/README.md` and `tests/unit/test_import_layering.py`).

### 2. The emitter

One command reads the four summaries and writes:
- **`paper/reported_values.tex`:** one `\newcommand` per value, each with a comment naming its summary key.
  LaTeX macro names cannot contain digits, so the part goes in as a word (for example `\PartTwo…`); the exact
  scheme is the builder's call.
- **The marked values in the claims table and results docs:** each claim row is one markdown table row, so
  multi-line blocks cannot sit inside its cells. Values are marked inline instead, for example
  `<!--v:part2.kay_swr_no_nonlocal_pct-->17 %<!--/v-->`: the emitter rewrites only the text between the two
  markers, and the prose around it stays hand-written. Multi-line generated blocks (the
  `write_epoch_register_doc.py` style) are still used outside tables. *(The assistant's call; the builder may
  change the marker syntax.)*
- **The numbers figure captions use:** figure scripts read values through the same loader and the same
  formatting function, so a caption and the prose cannot round a number differently.

Rounding policy lives in the emitter only. Exact counts print in full. Proportions print as whole percents unless
a claim needs more. Effect sizes print to two significant figures. The builder may refine these and should write
the policy down in the emitter's docstring, as statespacecheck-paper does.

### 3. The chat habit

Add a rule to the project `CLAUDE.md`: when stating a result number, read it from a summary file and name the
file and key. If the number is not in a summary, say so instead of computing it inline.

### 4. Moving the existing numbers

The claims table and results docs (`docs/replay_rate_content_swr_synthesis.md`, `docs/per_bin_rate_results.md`,
and others) already hold many typed numbers. They move one part at a time:
1. When a part's summary exists, its typed numbers become generated values.
2. Each typed number is compared with the computed one first. A disagreeing number is replaced by the computed
   one and listed for the lead (anti-goal above).
3. A typed number that no current output can back (for example claim 19a's one-session "≈ 0.73") is replaced by
   a value computed from the current run of that analysis where one exists (for 19a, the 242-epoch spike-evidence
   run). Where no analysis exists yet (claims marked "no script"), the number is replaced by a visible "not
   computed" mark. Both kinds are listed for the lead. *(Decided by the lead: replace, and tell the lead.)*

**Which docs move** *(the assistant's call)*: the claims table, and the living results docs in `docs/` that
report fitted or measured numbers: `replay_rate_content_swr_synthesis.md`, `per_bin_rate_results.md`, and any
other undated results doc found while migrating a part. Design specs in `docs/design/` are not moved: they
record why a method was chosen and what it showed then.

**Existing report writers stay** *(the assistant's call)*: the scripts that already write run reports
(`detection/report_inventory_comparison.py`, `detection/report_event_features.py`, the `report.md` and
`captions.md` they write, and similar) are upstream run reports, not reported values. Summaries read their
outputs. They are not rewired to read the summaries, which would make the dependency circular. Paper figure
captions that state numbers (`paper/figures/*/…caption…`) do move, to read the summary.

Numbers in dated point-in-time records are left alone: the `.claude/docs/reviews/` reports, the
`.claude/research/` notes, and dated docs such as `docs/reference_comparison_2026-09-17.md`. They record what
was true then.

## The first build: all of Part 2, starting with claim 23

Claim 23 is "the other side of the overlap". The non-local detector judges some detected SWRs to have local
content, under every SWR definition, and even events of the other detectors that overlap a non-local event have
stretches that are not non-local. The lead wants it reported precisely.

**Definitions (decided by the lead):**
- **Local content** inside an event = the share of the event's time bins in which the decoder puts the Local
  state above 0.8.
- **No-Spike content** = the share of bins with No-Spike above 0.8.
- **Non-local** = the share of bins with non-local (Continuous + Fragmented) above 0.8. This is the existing
  `ss_frac_rows_nonlocal` rule.
- **Uncertain** = bins where none of these is above 0.8. The four shares sum to 1 for each event. This last
  category is the assistant's addition, so the shares close; the lead has not reviewed it.

**Reporting (decided by the lead):**
- **Main figure and paper:** Kay SWRs only.
- **Supplement:** the other SWR definitions (Karlsson, Yu, Zugaro, Carey), MUA, and the Widloski detector under
  both of its cohorts (`memoryless`, cohort `published`; `memoryless_criteria`, cohort `replay_criteria`),
  reported separately. Every Widloski value stays marked pending the Phase 2 pilot checks until those pass.

**Values for each detector:**
- the share of its events that overlap no non-local event;
- of those, how they divide by content: mostly Local, mostly No-Spike, or neither (the rule for "mostly" is the
  builder's to propose);
- for its events that do overlap a non-local event, the distribution over events of the Local, No-Spike,
  non-local and uncertain shares.

**Inputs that exist:**
- the detector-comparison run `afeb897d850f`, whose `directional_by_*` tables count which events overlap a
  non-local event (the per-event flag is the `pair_membership` table of run `30cf1cc432e9`);
- the event-features run `30cf1cc432e9`, all 242 epochs `ok`, which gives `ss_frac_rows_nonlocal` and
  `ss_local_prob_mean` per event.

**What is missing:** the Local-above-0.8 and No-Spike-above-0.8 shares per event. Mean probabilities are not the
lead's definition. Computing these shares is a small analysis change, made on its own before the summary reads
it. Either:
- add two columns to `ss_features` in `analysis/inventory_comparison/features.py` beside `ss_frac_rows_nonlocal`,
  and rerun the event-features run (the last full run took about 3 h of wall time on 2026-10-18);
- or compute them in a separate pass over the same intervals and decodes.

That choice is the builder's.

**An oddity to check, not to report around:** under Zugaro, every animal's overlapping events read 100 %
non-local (`features_ss_by_animal.csv`, population `matched_by:state_space`). That may be a property of Zugaro's
events or a bug. The builder check found Zugaro's events short (median about 19 rows, 38 ms, against 43 rows for
Kay, on a subset of epochs), which would make most of them fully non-local; that fits a property of the detector,
but it is not yet checked. Find out before Zugaro's values are reported.

**A provenance note:** run `30cf1cc432e9`'s `manifest.json` names decode paths under
`/cumulus/edeno/continuum-swr-replay-phase4-run/`, a worktree removed on 2026-10-26. Its `Processed-Data` was a
symlink to the main tree's, so the same files are in `Processed-Data/`. The sha256 values in the manifest can
confirm it.

**Scope (decided by the lead):** the first build is all of Part 2, not only claim 23: the shared pieces
(summary format, emitter, `paper/reported_values.tex`, the inline markers, the `CLAUDE.md` rule), then every
Part 2 claim row (2a–2d) with its summary values, its typed numbers moved, and the replacement list for the
lead.

## After Part 2

Parts 1, 3 and 4 follow, in an order the lead chooses. Each starts by reading its claims-table rows for the
values they state, then builds its summary, then moves its typed numbers (section 4).

## Left to the builder

- File format (JSON is the default), file names, and where the summaries live, as long as they can be committed
  and the docs and manuscript can be regenerated without `Processed-Data/`.
- The macro naming scheme.
- The summary schema's exact field names, and a schema version.
- How summarize steps are invoked: one script with a part argument, or one script per part. They are not
  `run_refits.py` runs unless that is simpler.
- How figure scripts read caption values.
- The rule for "mostly Local / mostly No-Spike" for SWRs with no non-local event.
- Whether the Local and No-Spike shares are added to the event-features run or computed in a separate pass.
