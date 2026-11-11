# Questions and their numbers: one list, one map, interpretation in the Results

**Status:** draft for the lead, 2026-11-10. Nothing here is built yet. Every
count and every finding in this document was read from the files named beside it,
at commit `3bcca2e7`. The inventory script is
`.claude/docs/plans/questions-and-numbers/inventory.py` (run from the repository
root). An independent fact-check of this document against the repository found
errors, which are corrected here.

**Counting rule.** A value tagged with several claims counts once per tag, not
once per claim.

No result value is quoted. Where a claim's text holds a number, the key stands in
its place, as `[part2.key]`.

## The problem

The project reports about 2,000 numbers. They are stored once, in
`paper/reported/part{0..4}.json`, and the manuscript cites them through
`paper/reported_values.tex`. That chain is sound and stays as it is.

What is not sound is everything around it. Seven hand-written documents say which
question each number answers, which figure shows it, and what it means, and each
repeats parts of the others:

| Document | Lines | Value markers | What only it holds |
|---|---|---|---|
| `paper/claims_table.md` | 170 | 498 keys | the argument for each claim; narrative order; spike-support logic; considered and dropped |
| `docs/analysis_map.md` | 215 | 7 keys | run names; claims with no script; old run ids and question labels |
| `docs/figure-pipeline.md` | 70 | 1 key | what each figure reads; its tests; the caption procedure |
| `paper/reported/overview.md` | 419 | 614 keys | the assistant's reading of the numbers, by question |
| `docs/replay_rate_content_swr_synthesis.md` | 765 | 456 keys | a Part 3 research log |
| `docs/per_bin_rate_results.md` | 296 | 224 keys | the rate ladder, term by term |
| `paper/figure_claims_plan.md` | 273 | 0 | an older plan, with an older claim numbering |

(Keys are counted as distinct `<!--v:partN.key-->` markers per file, including
markers with a style suffix, `<!--v:partN.key@style-->`.)

Nothing checks the copies against each other, and they have drifted. The findings
below are the evidence.

## Findings from the inventory

Each is a fact with its source. None is resolved here; the decisions are listed
at the end.

1. **Figure tags are free text and do not all name a figure.** Values carry
   `used_in` tags set in the builders (`src/continuum_swr_replay/reported/part{n}.py`).
   `Value.__post_init__` (`reported/values.py:93`) only requires the tags to be
   non-empty. These tags name no figure in `docs/figure-pipeline.md`:
   - `figure: main, left out`: 20 values, claims 2 and 21 (`part2.py:571`, `left_out_values`);
   - `figure: main, other side of the overlap`: 14 values, claims 4 and 23 (`part2.py:390`, `other_detector_values`);
   - `figure: spike support`: 37 values, claims 6, 7, 7c (`part2.py:909`, `:1021`);
   - `figure: one class or two`: 3 values, claim 24 (`part2.py:1104`);
   - `figure: 3, content occupancy`: 12 values, each tagged with claims 13 and 14 (`part2.py:1512`);
   - `figure: content_estimators`: 80 values, all claim 19a, 78 of them also 19f
     and 19g (`part3.py:2294`). A script draws it
     (`scripts/content/compare_content_estimators.py`), but `figure-pipeline.md`
     does not list it.
2. **Figure 3's tags and its panel list disagree.** `figure-pipeline.md` gives
   Figure 3 four panels: a example, b SWR z-score (claim 24), c replay speed
   (claim 15), d replayed position.
   - The 8 ripple-power values tagged `figure: 3b` (`part2.py:748–752`,
     `brain_state_values`) carry claims 5 and 24. Panel 3b fits claim 24. The
     question is whether claim 5 (theta and speed) belongs on them. Its theta,
     speed and gamma values are tagged `supplement` only.
   - Claims 14, 16 and 16b: 48 values are tagged `figure: 3`, no panel
     (`part2.py:1210`, `_PROPERTIES`).
   - The claims table places claim 4 in "Main Fig 3 Venn" (`claims_table.md:43`).
     Figure 3 has no Venn panel.
3. **The claims table contradicts itself about the held-out test split by SWR.**
   (Corrected 2026-11-10: an earlier version of this finding said the values came
   from a script not in the repository. That was wrong.) Claim 7's SWR split has
   10 values (`part2.json`, keys `heldout_*`), tagged `paper` and not pending.
   Their source is
   `/cumulus/edeno/continuum-swr-replay/Processed-Data/_refresh_20261016/state_evidence/nl_bouts_by_swr_stats.csv`.
   - **The code side.** The Part 2 builder's docstring (`part2.py:15–17`) says
     the tracked `scripts/detection/split_state_evidence_by_swr.py` writes that
     file. The script was added in `e78273b5` (2026-10-30). `part2.json` records
     the run's `nl_bouts_by_swr.provenance.json`, by checksum only.
   - **The claims table:**
     - line 49's comparison column names the same tracked script;
     - line 49's placement column names an untracked research script,
       `.claude/research/refresh_followup/f1_scripts/state_evidence_by_swr.py`
       (absent, and gitignored at `.gitignore:227`);
     - lines 112 and 119–120 say the values are pending.
   - **Not known from here:** whether the CSV was written by the tracked script.
     The provenance file that says so is on the cluster, which is not mounted on
     this machine. Follow-up in `TODO.md` ("Reported values").
   - No value in any of the five summaries has a `pending` note.
4. **Two summaries have no exact code version.** `part3.json` records commit
   `c96c9133ef6c-dirty` and `part4.json` records `5609943e9472-dirty`: both were
   built from uncommitted changes.
   - `part0.json` (commit `1577f7209200`) predates two later commits to
     `reported/part0.py` (`d4d42542`, `882033bf`).
   - Part 0 needs no lab data, so it was rebuilt in memory at `3bcca2e7`. Its 32
     values and its inputs are identical to `part0.json`.
5. **Tags with no claim or figure.**
   - **Claim tags with no row in the claims table:**
     - `introduction`: the 32 Part 0 values. These are fine: they belong to the
       introduction and to Table S1.
     - `synthesis Layer 2`: 10 Part 3 values (`power_*`: the ripple-power model's
       coefficients on behavioral covariates).
   - **Values tagged only with a document,** no `paper`, `supplement` or figure:
     25 in all.
     - 21 are tagged `docs: replay_rate_content_swr_synthesis`: the 10
       `power_*` values, and 11 claim-19c `rate_contrast_*` values
       (`part3.py:1854`).
     - 4 are tagged `docs: rate_models`.
     - `Value`'s docstring allows only `"paper"`, `"supplement"` or
       `"figure: <name>"` (`values.py:59`). Retiring the synthesis document leaves
       these values with no use.
6. **The old plan uses another numbering.** `figure_claims_plan.md` calls the
   local-kernel re-decode "#9" (`:144`). The claims table calls it #8 and uses #9
   for "remote events are the clean population". The old plan's claim C says
   non-SWR events are "a different kind of event" that serves "a different
   function" (`:13–14`, `:107`). The claims table lists that claim as dropped
   (`claims_table.md:141` onward).
7. **Questions whose values do not answer them.** No check catches these.
   - **No values:** claims 1, 3, 8, 10, 11 and 25.
     - The claims table marks 3, 8 and 11 GAP.
     - It marks claim 10 "have (in-sample); GAP (held-out + immobile)", so the
       in-sample part it calls "have" has no values in any summary.
     - Claim 1 is the example figure; claim 25 is framing.
   - **Claim 12 (false-positive rate)** has 99 values. 97 are from the
     simulated-replay comparison, and 2 (`pilot_*`) are from one real epoch,
     chimi20200225_02_r1. The claims table still marks it "Supp · GAP", because
     there is no real-data false-positive rate.
   - **Claim 9 (remote events)** has 33 values, all Part 1 held-out values also
     tagged claim 7. None is about remote events.
   - The proposed status rule ("computed" means at least one value) would call
     claims 9 and 12 computed. So each question's status must be set by hand, and
     reviewed against what its values measure.
8. **Numbers typed by hand.** The claims table has typed numbers that are
   reported values, not markers:
   - the epoch count of `part2.evidence_epochs` at lines 49, 50 and 100;
   - a typed animal count ("in all … animals") at lines 125–126, and the typed animal denominator in
     counts such as `<marker>/5` (for example lines 152 and 156).

   No check finds typed numbers yet.
9. **Analysis inputs inside figure folders.** 43 Part 2 values read files under
   `paper/figures/content_occupancy/` (12) and `paper/figures/reality_all/` (31).
   Both folders are gitignored (`.gitignore:208`, `:209`) and absent on this
   machine, so these values can be rebuilt only where those outputs exist.

## The paper's outline (lead, 2026-11-10)

One part per main figure:

1. **Part 1 / Figure 1:** explain the model.
2. **Part 2 / Figure 2:**
   - it works;
   - it finds events that other methods do not;
   - the events are real;
   - what the events are like.
3. **Part 3 / Figure 3:** the events relate to behavior.
4. **Part 4 / Figure 4:** the events relate to mPFC.

This replaces the claims table's grouping. The question list's `part` field uses
these parts, with Part 2 split into its four sub-questions.

**Mapping of the current questions.** The assistant drafted it; the lead
confirmed the placement of the properties and of "it works" (2026-11-10):

| Part | Questions |
|---|---|
| 1, the model | none with numbers (Figure 1 is simulated) |
| 2, it works | 10, 11, 12a, 12b, 3 (moved from the claims table's Part 1) |
| 2, finds events others miss | 1, 2, 21, 4, 23, 22, 24, 25 |
| 2, the events are real | 5, 6, 7, 7b, 7c, 8, 9, 13 |
| 2, what the events are like | 14, 15, 16, 16b, 17, 18 |
| 3, behavior | 19a–19h (19h: the `power_*` values, decision 4) |
| 4, mPFC | 20 |

`docs/figure-pipeline.md` still describes the current Figures 2 and 3 (events and
SWRs; events outside SWRs). Its update waits until Figure 3 is settled.

## Decided (lead, 2026-11-10)

**Answers to the decisions listed at the end:**
- **Figure 3 (decision 2):** its contents are not settled. No panel is assigned
  yet; values keep their current tags.
- **Claim 7's SWR split (decision 3):** the lead asked for the values to be
  marked pending, with a follow-up. That was said before finding 3 was
  corrected. The follow-up (`TODO.md`) decides pending or not, from the cluster
  provenance file.
- **Considered and dropped (decision 6):** keep tracking it. Each dropped claim
  becomes a question with status `dropped` and its reason, in `questions.py`.
- **Nothing goes into the manuscript yet.** No Results, Methods or Discussion
  text is drafted for now. Retiring the interpretation documents is deferred
  (Order of work, step 5).
- **Milestones (decision 7):** a draft going to co-authors, submission, and a
  release.
- **Rebuild (decision 8):** Parts 3 and 4 are rebuilt from committed code before
  the Results are drafted, if possible. It needs the cluster.
- **Question wording (decision 9):** kept as drafted for now.
- **Decisions 1, 4 and 5:** the lead took the assistant's recommendations
  (2026-11-10). They are applied in Order of work, step 2.
  - **Figure tags (decision 1):**
    - `main, left out` → Figure 2 (questions 2, 21);
    - `main, other side of the overlap` → Figure 2 (4, 23);
    - `spike support` → Figure 2 (6, 7, 7c);
    - `3, content occupancy` → Figure 2 (14; the old tag pointed at Figure 3,
      which is now behavior);
    - `one class or two` → the supplement (24);
    - `content_estimators` → the supplement (19a, 19f, 19g), a figure that
      exists and is added to `figure-pipeline.md`.
    - Panel letters wait until Figure 2 is designed.
  - **Values only a document uses (decision 4):**
    - `power_*` becomes Part 3 question **19h**, tagged `supplement`;
    - the 15 claim-19c model diagnostics are tagged `supplement`.
  - **Questions 9 and 12 (decision 5):**
    - **9:** the 16 remote-event spike-support values (`non_swr_remote_*`,
      `swr_remote_*` in `part2.json`, now tagged 6, 7, 7c or 7b) also get tag 9. The Part 1 held-out
      values lose tag 9 and keep tag 7.
    - **12** splits in two:
      - **12a**, the simulation comparison (computed, the current 99 values);
      - **12b**, the false-positive rate on real data (not computed).

- **Numbers are stored only in the summaries,** and the manuscript cites them only
  through `reported_values.tex`. This is the statespacecheck-paper chain. We keep
  the summaries by part, not by figure: 1,633 of 2,021 values have no figure tag.
  Also, the builders read analysis outputs, mostly tables (with the exceptions in
  finding 9), rather than computing inside the figure scripts.
- **Interpretation goes into the manuscript's Results,** not into markdown.
- **We keep a list of questions, each with its numbers,** to build the Results
  from.

## Design

### 1. The question list: `src/continuum_swr_replay/reported/questions.py`

This is one hand-written module, the only place a question's text, figure and
status are written. It follows the pattern of `analysis/replay/model_specs.py`.

Each question has:

- `id`: the current claim id (`"2"`, `"15"`, `"19a"`, …), so the values' existing
  `claims` tags stay valid;
- `part`: `"1"`, `"2a"`, …, `"4"`, as in the claims table;
- `question`: the question, written as a question and not as its answer;
- `panels`: figure-panel ids (below), possibly none;
- `status`: `computed`, `partly computed`, `not computed`, `no numbers` (an
  example figure or framing) or `dropped` (from the claims table's "considered
  and dropped"), with a one-line reason when not `computed`.

The module also lists the **figures and panels** (id, title, script). These are
the only valid figure tags. `figure-pipeline.md` keeps one hand-written section
per figure, in statespacecheck-paper's format, and a test checks that every
figure id has a section.

`Value` then rejects a claim id or figure tag that the module does not list. So a
builder with a stray tag fails when it runs, not months later in a document.

### 2. The generated view: `paper/reported/questions.md`

`make reported-values` writes it, and it replaces `values_index.md`
(`reported/emit.py:values_index` is rewritten to produce it). It is
grouped by part, then question. Each question shows:
- its status and figure panels;
- the source tables;
- every value: the macro to cite, the value as the emitter formats it, and its
  description.

Shape of one entry (values filled by the emitter, not shown here):

> ### 15. How does replay speed differ between SWR and non-SWR events?
> Figure 3c · computed · source: event table (`part2.json`)
>
> | Macro | Value | What it is |
> |---|---|---|
> | `\PartTwoSpeedMedianNonSwr` | … | median replay speed (median over the event) of immobile non-SWR non-local events |
> | `\PartTwoSpeedMedianSwr` | … | median replay speed (median over the event) of immobile SWR-overlapping non-local events |
> | … | | |

A question with no values shows its status and reason only. A section at the end
lists every value per figure panel.

### 3. Interpretation: the Results section of `paper/main.tex`

`main.tex` already has the four Results subsections, matching the four parts.
- **Drafts.** The readings now in `overview.md` and the argument in
  `claims_table.md` move there as drafts, marked so they render visibly and cannot
  reach a submission. A check refuses to build the submission while a draft mark
  is left.
- **Markers become macros.** `<!--v:part2.speed_median_swr-->` becomes
  `\PartTwoSpeedMedianSwr`, mechanically, by `emit.macro_name`.
- **The lead checks the drafts.** The readings were written by the assistant
  (`overview.md:5–6` says so). The lead checks each draft paragraph against its
  question's numbers before removing its draft mark.

### 4. Provenance

**What statespacecheck-paper does:**
- Each summary records two digests:
  - `source_tree_sha256`, a SHA-256 of every `.py` file in its package;
  - `uv_lock_sha256`.
- Its emitter refuses when the two summaries' provenance blocks differ from each
  other (`require_matching_source_provenance`, `reported_values.py:1025`). It
  does not compare them with the current code.
- It has no check for uncommitted changes.

**Requirement (lead, 2026-11-10):** the rule must not slow development down.

**How often code changes.** Commits from 2026-10-10 to 2026-11-10 that touch the
package modules each part's builder loads:

| Part | Modules loaded | Commits | Commits that change code (docstrings and comments ignored) | Days with commits |
|---|---|---|---|---|
| 0 | 6 | 27 | 21 | 7 |
| 1 | 5 | 26 | 20 | 7 |
| 2 | 49 | 117 | 100 | 14 |
| 3 | 62 | 129 | 113 | 16 |
| 4 | 38 | 100 | 83 | 15 |

"Modules loaded" is `sys.modules` after importing `reported.part{n}`.
"Change code" compares the docstring-stripped AST before and after each commit.
The script is `.claude/docs/plans/questions-and-numbers/astchurn.py`.

A rule that blocks whenever code has changed since a build would block on most
working days, even per part, even ignoring docstrings, and Parts 1–4 rebuild only
on the data machine. So the rule records provenance always, reports staleness
always, and blocks only at milestones:

1. **Record exactly, at no cost, at every build.**
   - Each summary records:
     - the commit;
     - whether the tree had uncommitted changes, and if so the diff of the tracked
       files the builder loaded;
     - a SHA-256 of each package module the builder actually loaded (from
       `sys.modules`, so no list has to be kept up to date), over its
       docstring-stripped AST;
     - the SHA-256 of `uv.lock`.
   - The input checksums are already recorded.
   - As built (after an independent review), the record also:
     - names the commit its diff is against;
     - lists files git does not track, which count as uncommitted;
     - includes the package's data files (the epoch register);
     - lists files modified while the build ran;
     - says why, instead of failing, when git cannot answer.

     Attribute docstrings are also left out of the code digest.
   - Building from uncommitted code is allowed, because the diff makes it
     reproducible. Finding 4 could not have happened with this record.
2. **Report staleness, never fail, during development.**
   - `make reported-values` prints, per part, the loaded modules whose code
     changed since the build.
   - `questions.md` shows each summary's build commit, but not live staleness.
     (Changed while building it: a staleness line there would make the
     byte-for-byte freshness test fail on every code commit, which is the
     friction this rule exists to avoid. The emitter's printout and
     `make release-check` carry it instead.)
3. **Block only at a milestone.** A `make release-check` fails when any summary
   was built from uncommitted code, or when its loaded modules' code changed
   since. It runs before a draft goes to co-authors, before submission, and in
   the release workflow; not on every commit and not in CI. Only the stale parts
   are rebuilt.

**Left out on purpose:**
- a hash over the whole package;
- refusing builds from uncommitted code;
- requiring the five summaries to match each other.

Each of those would force rebuilds that the table shows would come almost daily.

**Limit.** This records the code that built the summary, not the code that made
its inputs (the fits and tables it reads). The inputs are pinned by checksum
only. Finding 3 is an input whose producing script is missing, and this rule
would not catch it. Each input's producing script is recorded by that run's own
provenance, where it has one.

### 5. Checks (tests)

1. Every value's claim id and figure tag is in `questions.py`. This also runs in
   `Value`.
2. Every question marked `computed` or `partly computed` has at least one value.
   A question marked `not computed` or `no numbers` has none. This catches only
   missing values, not values that answer another question (finding 7: claims 9
   and 12). So each status is set by hand and reviewed against what its values
   measure.
3. `questions.md` matches a fresh render, byte for byte, as `reported_values.tex`
   does now.
4. Every figure id has a section in `figure-pipeline.md`.
5. **Nothing lost in the move:** every marker key in the documents being retired
   (with or without a style suffix) ends up in one of two places:
   - cited by its macro in `main.tex`;
   - listed in a carry-over file with its reason, for example "supplementary
     table" or "not reported".

   This test is used while the documents are retired, then removed with them.
6. Every value tagged `paper` is cited in `main.tex`. This test is added when the
   Results are written. statespacecheck-paper's version requires every macro to
   appear in `main.tex` (`tests/test_reported_values.py:948`). Ours is narrower,
   because our macro file also holds supplement values.

## Draft question list

The text in the "Draft question" column is the assistant's drafting, for the
lead to edit. The claim text beside it is copied from `paper/claims_table.md`
(line given), with markers replaced by their keys. "Values" counts the summary
values whose `claims` names the id; "Figure tags" counts their `used_in` figure
tags.

| Id | Line | Claim, as written | Draft question | Values | Figure tags | Placement, as written (start) |
|---|---|---|---|---|---|---|
| **Part 1** | | | | | | |
| 10 | 34 | The decoder is accurate in general and out-of-sample (NOT a positive control for the events) | Is the decoder accurate, in sample and on held-out data? | 0 | — | Supp · have (in-sample); GAP (held-out + immobile) |
| 11 | 35 | **(CV)** Held-out likelihood and label stability favor the 4-state model over ContFrag / reduced | Do held-out likelihood and event-label stability favor the four-state model over the Continuous–Fragmented and reduced models? | 0 | — | Supp · GAP |
| 12a | 36 | **[M4]** The detector has a low false-positive rate / is specific | On simulated data, how often does the detector report an event where no replay was inserted, and how does it compare with the Widloski detector on replays of known shape? | 99 | memoryless comparison (99) | Supp · GAP |
| 3 | 37 | **[M5]** The non-SWR fraction is robust to detector thresholds | How do the non-SWR fraction and the close/remote split change across the non-local and SWR-z thresholds? | 0 | — | Supp · GAP |
| 8 | 38 | **(W3 definitive)** Close events are not place-field-extent leakage | Are close events produced by the extent of place fields? | 0 | — | Supp · GAP (GPU subset) |
| **Part 2a** | | | | | | |
| 1 | 41 | Non-local events occur without an SWR | Do non-local events occur without an SWR? (examples) | 0 | — | Main Fig 2a · have |
| 2 | 42 | A large fraction of non-local events carry no SWR | What fraction of immobile non-local events overlaps no SWR? | 31 | 2b (11), main, left out (20) | Main Fig 2b · have |
| 4 | 43 | SWR-detection and non-local-detection are separate axes | How do SWR detection and non-local detection overlap, in events and in time points? | 98 | main, other side of the overlap (14) | Main Fig 3 Venn · have, standardize with Fig 2b |
| 23 | 44 | Content-based detection is specific to trajectory content; LFP detection is content-agnostic | For each other detector, what share of its events overlaps no non-local event, and how do those divide by decoder state; for the ones that do overlap, what share of their time bins does each state take? | 98 | main, other side of the overlap (14) | Main (Venn / gallery; Kay only) · values in `paper/reported/part2.json`; panel GAP; … |
| 24 | 45 | Detection spans the ripple-power continuum | How is SWR z-score distributed at immobile Local and Non-Local time points, and how much of the non-local mass lies below the SWR-detection threshold? | 15 | 3b (8), one class or two (3) | Main Fig 3b · have; add threshold line & below-threshold % |
| **Part 2b** | | | | | | |
| 5 | 47 | **(W1)** Non-SWR events are not theta / movement-driven | How do theta z-score and speed compare across SWR events, non-SWR events and running? | 54 | 3b (8) | Supp · have (data in hand) |
| 6 | 48 | **(W2a · [M1])** Events are spike-driven by multiple tetrodes (not a no-spike / ground-process event) | How do per-event spike count, MUA z-score and active tetrodes compare across non-SWR events, SWR events and matched immobile non-event windows? | 14 | spike support (14) | Main reality fig · have: all `[part2.evidence_epochs]` epochs … |
| 7 | 49 | **(W2b + W3 · CENTERPIECE)** The decoded displacement is supported by spatially informative spikes, not just the prior | Do an event's spikes favor its decoded location over the animal's position, against matched windows and on held-out tetrodes? | 65 | spike support (29) | Main reality fig · have (`[typed: part2.evidence_epochs]` epochs, F1 2026-10-16); … |
| 7b | 50 | **(W2b)** Where event MAPs sit relative to the GPI peak | Where do event MAPs sit relative to the GPI peak, against matched quiet windows? | 4 | — | Supp · the `[typed: part2.evidence_epochs]`-epoch run measures the distance to the GPI peak … |
| 7c | 51 | The matched non-event immobility baseline is the key null for spike evidence | What spike evidence do matched immobile non-event windows have? | 7 | spike support (7) | Methods note (folds into #7) · … |
| 9 | 52 | **(W4)** Remote events are the clean non-local population | Do remote events pass the spike-support tests (6, 7)? | 33 | — | Main · follows from #13 + #6/#7 |
| **Part 2c** | | | | | | |
| 13 | 55 | **[M2]** Non-local events split into close and remote distance regimes | How is the decoded distance from the animal distributed, by SWR status? | 12 | 3, content occupancy (12) | Main Fig 3 distance_from_animal · raw plot exists; recut needed |
| 14 | 56 | Non-SWR content sits at reward wells more often than SWR content; SWR content sits mostly on paths between wells | Where does content sit relative to the animal and to the reward wells, SWR against non-SWR? | 20 | 3, content occupancy (12), 3 (8) | Main Fig 3 (content_occupancy) · have …; own-well gated on #8 |
| 15 | 57 | Non-SWR replay is slower than SWR replay | How does replay speed differ between SWR and non-SWR events? | 9 | 3c (8) | Main Fig 3c · have; annotate the animal count |
| 16 | 58 | Non-SWR replay is spatially compact | How does spatial extent differ between SWR and non-SWR events? | 17 | 3 (16) | Promote to Main Fig 3 · ECDF exists; main-panel recut needed |
| 16b | 59 | Non-SWR replay is **stationary** — the content barely moves; SWR sweeps | How much does content move within an event (pauses, largest step, reversals), SWR against non-SWR? | 24 | 3 (24) | Main Fig 3 · have (…); panel GAP |
| 17 | 60 | Non-SWR replay has a sharper, more spatially-concentrated posterior than SWR | How concentrated is the posterior (peak, HPD coverage, multimodality), SWR against non-SWR? | 25 | — | Supp · analysis exists; panel GAP |
| 18 | 61 | Non-SWR events are shorter than SWR events | How does event duration differ between SWR and non-SWR events? | 6 | — | Supp · data in hand; panel GAP |
| **Part 2d** | | | | | | |
| 21 | 63 | We detect non-local events that do not overlap any SWR | How many detected non-local events overlap no SWR? (The claims table reads these as the events SWR-triggered detection misses; that reading belongs in the Results.) | 31 | 2b (11), main, left out (20) | Main · have |
| 22 | 64 | Content-based event boundaries differ from SWR boundaries | How do non-local event starts and ends differ from those of the matched SWR? | 3 | 2d (3) | Main Fig 2d · have; do not claim better boundaries yet |
| 25 | 65 | **(candidate)** Content-targeted, not LFP-proxy-triggered | (framing, no numbers) | 0 | — | framing · candidate |
| **Part 3** | | | | | | |
| 19a | 68 | Content over-represents the **current arm** (near the animal but spike-supported and displaced) in SWR and non-SWR events alike, with **no consistent SWR difference** once each kind's arm baselines are accounted for; SWR content **shifts onto the trunk arms** and is **broader within the event** | Which arms does content over-represent, and does that differ between SWR and non-SWR events once each kind's arm baseline is accounted for? | 244 | content_estimators (80) | Main/Supp Fig 3 · have; #8 re-run of the decoder pending |
| 19b | 69 | Non-SWR content is **not selective for recently rewarded arms** (measured on the realized reward fraction, not on arm value) | Is non-SWR content selective for recently rewarded arms (realized reward fraction)? | 36 | — | Supp · have |
| 19c | 70 | Non-SWR non-local occurrence is **less reward-gated than SWR-coupled occurrence — at lag 1 of the reward kernel**, while the slower reward-memory baseline is shared | How does the reward-history kernel of event occurrence (lag 1 and the slower lags) compare between SWR-coupled and non-SWR events? | 468 | — | Main/Supp · have (kernel + paired test + self-history + model checking) |
| 19d | 71 | Content does **not** correlate with the upcoming choice once behaviour is in the model, with or without SWR | Does content correlate with the upcoming choice once behavior is in the model, with or without SWR? | 28 | — | Supp · have (negative) |
| 19e | 72 | Non-SWR content does **not** predict stay/leave | Does non-SWR content predict staying or leaving? | 17 | — | Supp · have (negative) |
| 19g | 73 | Content over-represents arms the animal is **uncertain about** (epistemic/explore axis); arm value's main effect is positive in the median but not across animals, and only uncertainty earns its parameter | Does content over-represent arms the animal is uncertain about, or arms of high value? | 165 | content_estimators (78), uncertainty_content (20) | Supp · have (uncertainty_content feature, panel E; within-arm horse-race) |
| 19f | 74 | Reward gates with-SWR **rate**; SWR content shows **no direct (length-adjusted) avoidance** of high-value arms under any estimator, and a **negative total contrast** (the non-SWR-content half of the old double dissociation is **withdrawn pending refit**) | How does SWR against non-SWR content depend on the arm's reward fraction (`CS4`) and, separately, on its value (`CS5`), as direct (length-adjusted) and total effects? (The claim's "high-value" covers both; the claims table's dropped list calls the `CS4` total effect avoidance of recently rewarded arms. The lead keeps the two apart or merges them.) | 136 | content_estimators (78) | Supp · have (new); interaction-block refit pending |
| **Part 4** | | | | | | |
| 20 | 76 | Non-SWR replay significantly modulates mPFC activity | Do non-SWR events modulate mPFC activity, against matched immobile non-event windows, and how does that compare with SWR events? | 604 | 4d (25), 4c (1) | Main Fig 4 · have for engagement; … |
| **Other tags** | | | | | | |
| introduction | — | (Part 0) | What detection and decoding parameters do published replay studies use? | 32 | — | introduction and Table S1 |
| 19h (was `synthesis Layer 2`) | — | (no row) | Does the ripple power of events depend on the recent patch-switch rate, the previous trial's reward outcome or the post-outcome reward prediction error? | 10 | — | `docs: replay_rate_content_swr_synthesis` only; becomes `supplement` |
| 12b (new) | — | (split from 12) | How often does the detector report non-local events on shuffled or thinned real spikes, and in windows where no replay is expected? | 0 | — | not computed |

## Where each piece of the retired documents goes

Line ranges are those of each file at commit `3bcca2e7`. "Results" means the
draft in the named subsection of `paper/main.tex`. The test in Checks item 5
confirms that every marker key arrives somewhere.

**`paper/reported/overview.md`** (deleted after the move)

| Lines | Section | Goes to |
|---|---|---|
| 1–15 | Purpose, and cautions that apply throughout (plain medians; epochs not independent; recording-day tests still to be added) | Methods, statistics (draft); the recording-day tests go to `TODO.md` |
| 16–50 | Part 1 | Results, Part 1 |
| 51–158 | Part 2 | Results, Part 2 |
| 159–274 | Part 3 | Results, Part 3 |
| 275–419 | Part 4 | Results, Part 4 |

**`paper/claims_table.md`** (deleted after the move)

| Lines | Section | Goes to |
|---|---|---|
| 1–29 | Purpose; the four parts; "descriptive, not categorical" | Results introduction (draft) |
| 30–77, Claim | Each row's claim | Results draft, as that question's paragraph claim |
| 30–77, Comparison | Each row's comparison | Results draft, and the question's panels in `questions.py` |
| 30–77, What it shows | Each row's reading | Results draft |
| 30–77, Placement | Each row's placement and status | `questions.py` status; `figure-pipeline.md` |
| 54 | Part 2c framing row, no id ("one contrast": focal and stationary against extended and moving) | Results, Part 2c introduction (draft) |
| 67 | Part 3 framing row, no id ("how to read every content number": the per-event allocation fraction estimand) | Methods, content models; one sentence in the Results, Part 3 introduction (draft) |
| 80–89 | Narrative order | The order of the Results paragraphs |
| 90–99 | Spike-support logic (W1–W4) | Results, Part 2b (draft) |
| 100–117 | Spike-support status | Results, Part 2b (draft); finding 3 decides the held-out split |
| 118–128 | Status caveats; the open close-against-remote question | `TODO.md` |
| 129–139 | Main against supplement | `figure-pipeline.md` |
| 141–170 | Considered and dropped | Lead decides: Discussion draft, or `docs/design/model_history.md` |

**`docs/replay_rate_content_swr_synthesis.md`** (deleted after the move)

| Lines | Section | Goes to |
|---|---|---|
| 1–40 | Status and refit note | Methods (cohort) |
| 41–109 | Provenance and methods | Methods, or `docs/rate_models.md` |
| 110–211 | Layer 1, rate | Results, Part 3 |
| 212–235 | Behavioral-state covariates on rate (pending) | `TODO.md` |
| 236–258 | Layer 2, SWR coupling | Depends on the `synthesis Layer 2` decision |
| 259–539 | Layer 3, content (its opening and five subsections) | Results, Part 3 |
| 540–576 | Distinction from Gillespie et al. 2021 | Discussion draft |
| 577–616 | What's new here | Discussion draft |
| 617–668 | Confirmed | Results, Part 3, each line checked against its question's values |
| 669–717 | Hypotheses; unknown and pending | `TODO.md` |
| 718–745 | Caveats and confounds | Methods, or Discussion limitations |
| 746–765 | Reproducibility | `docs/figure-pipeline.md` or `docs/reproduce.md`, where not already there |

**`docs/per_bin_rate_results.md`**: becomes a generated supplementary table.
The prose at lines 1–56 goes to `docs/rate_models.md`, and section 5 (lines
217–236, no values) goes to `TODO.md` if it is still pending. Its inbound links
include `scripts/rate/plot_per_bin_rate_summary.py`,
`src/continuum_swr_replay/reported/part3.py` and
`tests/unit/test_summarize_per_bin_rate_models.py`. These are code, so each is
checked before the file is removed.

**`docs/analysis_map.md`** (deleted after the move)

| Lines | Section | Goes to |
|---|---|---|
| 1–35 | Purpose | — |
| 36–40 | The flow (a link to `reproduce.md`) | — (`reproduce.md` already holds it) |
| 41–166 | Per part: what it tests, script, run, output, claims | The sections of `figure-pipeline.md`; analyses with no figure go in a "numbers with no figure" section there |
| 41–57, "No script" | Claims 3, 8, 11, 12 (real data) | `questions.py` status |
| 167–176 | Supporting scripts | `scripts/README.md`, where not already there |
| 177–215 | Removed as redundant; old run ids; old question labels | `docs/design/model_history.md` |

**`paper/figure_claims_plan.md`**: deleted; git keeps it. Three items in its
"Open decisions" (lines 152–158) go to `TODO.md` for the lead to confirm or
close:
- the scope of the close-against-remote headline;
- whether D needs a re-implementation of the traditional pipeline;
- Figure 3d's "reward zones".

**`paper/reported/values_index.md`**: replaced by `questions.md`. Its links in
`docs/reproduce.md`, `paper/README.md`, `scripts/reported/emit_reported_values.py`
and `tests/unit/test_manuscript_macros.py` are updated.

**Links.** These live files link to a retired document and are updated:
- `README.md`, `docs/README.md`, `paper/README.md`, `paper/main.tex`;
- `docs/figure-pipeline.md`, `docs/models_table.md`, `docs/rate_models.md`,
  `docs/behavior_models.md`;
- `scripts/README.md`, `scripts/qa/run_refits.py`,
  `scripts/exploratory/_verify_content_models.py`;
- `src/continuum_swr_replay/analysis/replay/model_specs.py`,
  `src/continuum_swr_replay/reported/__init__.py`, `src/continuum_swr_replay/reported/values.py`;
- `src/continuum_swr_replay/reported/part3.py`. Its `docs: replay_rate_content_swr_synthesis`
  tags (`:1854`, `:2699`) are replaced per decision 4;
- the design documents under `docs/design/`.

Plans and reviews under `.claude/docs/` are records and keep their links.
`paper/reported/changes.md` and `removed_part{1..4}.md` record the earlier
migration and stay.

## Decisions for the lead

1. **The figure tags in finding 1.** For each tag: which figure and panel, or a
   new supplementary figure?
2. **Figure 3:**
   - Is claim 5 in 3b?
   - Where do claims 14, 16 and 16b go?
   - Does claim 4's Venn belong in Figure 3 at all?
3. **Claim 7's SWR split (finding 3):**
   - Mark the 10 values pending until a tracked script produces them, or drop them?
   - Is the script on the data machine?
4. **The 25 values tagged only with a document (finding 5).**
   - Is `synthesis Layer 2` (`power_*`) a question of its own, or not reported?
   - Are the 11 claim-19c `rate_contrast_*` values and the 4 `docs: rate_models`
     values `paper`, `supplement`, or not reported?
5. **Claims 9 and 12 (finding 7).** Claim 9's values are the claim 7 held-out
   values. Does claim 9 need its own remote-subset values, or does it fold into
   7 and 13? Claim 12: is it "partly computed" (simulation only)?
6. **Considered and dropped:** Discussion, or design history?
7. **Provenance milestones (Design 4):** which events run `make release-check`?
   Proposed: a draft going to co-authors, submission, and a release.
8. **Rebuild timing.** Do Parts 3 and 4 get rebuilt from committed code before
   the Results are drafted? Drafting from them first means drafting from numbers
   whose code is not recorded.
9. **The draft question wording above.** Draft 19f in particular separates
   reward fraction from value, which the claim's "high-value" does not.

## Order of work

Each step is one commit, and the tests pass after each.

1. `questions.py`, with the list above as the lead edits it. Tests 1, 2 and 4
   start in report-only mode and list today's mismatches.
2. Builders use the question and figure ids; the stray tags are fixed per the
   lead's decisions. The summaries must then be rebuilt on the data machine, so
   tests 1 and 2 are made strict after that rebuild.
3. The emitter writes `questions.md` and stops writing `values_index.md`. Test 3.
4. Provenance:
   - the build records the loaded modules' digests, `uv.lock` and any diff;
   - the emitter and `questions.md` report staleness;
   - `make release-check` blocks.

   Nothing blocks during development.
5. **Deferred (lead, 2026-11-10):** retire the interpretation documents only
   once the Results are being written. Until then they stay as they are. One
   commit per retired document: move its content per the tables above, add its
   keys to the carry-over check (test 5), update its links, delete it.
   `paper/figure_claims_plan.md` and `analysis_map.md` hold no interpretation
   that needs the Results, so they can be retired earlier.
   The `TODO.md` item "Summary provenance" (refuse a dirty tree) is replaced by
   step 4.
6. Rebuild the summaries on the data machine; make tests 1 and 2 strict.
