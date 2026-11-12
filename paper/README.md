# Paper

The manuscript, its figures, and the summary files its numbers come from.
Read [main.pdf](main.pdf), or build it from the committed files:

```bash
# From the repository root:
make manuscript
```

This needs LaTeX with `latexmk` and BibTeX (TeX Live or MacTeX). It needs neither
Python nor the lab's data: it uses the committed figures and
`reported_values.tex`.

The [paper-questions review](../.claude/docs/reviews/2026-11-11-paper-questions-review.md)
assesses the argument, scope, and likely reviewer concerns. The proposed
[paper-development plan](../docs/design/2026-11-11-paper-development-plan.md)
orders the validation, analysis, and writing work, with completion criteria for
each stage.
The [proposed question outline](../docs/design/2026-11-12-paper-question-outline.md)
groups the main questions, subquestions, and supporting controls, with replay
explicitly including stationary representations as well as trajectories.

## Files

| File | Purpose |
| --- | --- |
| `main.tex` | The manuscript and its preamble |
| `references.bib` | The bibliography (BibTeX) |
| `reported_values.tex` | Generated: one LaTeX macro per reported number, read by `main.tex` |
| `reported/<summary>.json` | The summary files every reported number is read from, named for what they hold: `survey` (the literature numbers the introduction quotes), `detection`, `events`, `behavior`, `mpfc` |
| `reported/inputs/detection/` | The tracked input files the detection summary reads: the simulation sweep summaries (`detection_{sweep,controls,heldout}_summary.csv`), `reference_check_summary.csv` and the labeled recording's numbers (`memoryless_comparison_recording.json`) |
| `reported/questions.md` | Generated: every question of the paper, in its outline, with the values that answer it (macro, value, meaning, source, the builder function that made it) and the values by figure |
| `figures/<topic>/` | Figures, one folder per figure, written by the figure scripts; only Figure 1's files are kept in the repository |
| `tables/` | `table_s1_literature_parameters.csv`, the <!--v:survey.papers-->57<!--/v-->-paper survey (Supplementary Data 1, edited by hand), and `table_s1_literature.tex`, Supplementary Table S1 generated from it |
| `Makefile`, `.latexmkrc` | Build settings |
| `LICENSE` | CC BY 4.0 for the manuscript text and figures |

## Numbers

Do not type a result number into `main.tex`. Use the macro that
`reported_values.tex` defines for it, for example
`\DetectionSweepReplays{}`; [reported/questions.md](reported/questions.md)
lists every value under the question it answers, with its macro (or, until its
summary is rebuilt, among the values with tag problems). The macro for summary key `detection.sweep_replays` is
the summary's name capitalized (`Detection`) followed by the key in CamelCase.

When a summary changes, regenerate the macros, then rebuild:

```bash
make summaries        # rebuild reported/<summary>.json (all but survey need the lab's processed data)
make reported-values  # write the macros and the docs' inline values
make manuscript
```

`make manuscript` rebuilds when `reported_values.tex` changes, but does not
regenerate it. `tests/reported/test_manuscript_macros.py` checks that every macro
`main.tex` uses is defined.

## Figures

`main.tex` draws each main figure from `figures/figureNN/figureNN.pdf`. A
figure that does not exist yet is drawn as a framed box naming the file and the
script that makes it, so the PDF always builds. `make figures` from the
repository root regenerates the main figures (it needs the lab's processed
data).

Git ignores the figure files under `figures/` by default. When a figure is
final, add a `!paper/figures/<topic>/<name>.pdf` line to `.gitignore`, as
Figure 1 has, and commit it with the rebuilt `main.pdf`.

## Supplementary tables

Supplementary Table S1 is the survey of replay-detection and decoding parameters
in <!--v:survey.papers-->57<!--/v--> published studies. `tables/table_s1_literature_parameters.csv` is the
paper's one copy of the survey, and it is published as Supplementary Data 1.
Each value was checked against the study's methods, supplement or released code
(first here on 2026-09-21, then in the `ripple_detection` repository through
2026-09-25, whose evidence file records the source of every value). Correct a
value in the CSV, then regenerate:

```bash
uv run --frozen python scripts/figures/generate_supp_literature_table.py
```

The introduction's literature numbers (modes, ranges and shares) are computed
from the same survey into `reported/survey.json`; after regenerating the table, run
`uv run --frozen python scripts/reported/summarize.py --summary survey` and
`make reported-values`. `tests/reported/test_literature_table.py` and
`tests/reported/test_reported_survey.py` fail when the committed table or summary no
longer matches the CSV.

The lab's shared Google Sheet (`12mMgiYWO7tKRhs5qhMXMZ5RggGi1J24wxIkcs4knTUI`)
does not feed the paper: a change made only in the Sheet does not reach it.

## Bibliography

`references.bib` is exported from Eric Denovellis's Zotero library by Better
BibTeX: the **Non-Local-Paper** collection, with "Keep updated", so Zotero
rewrites the file whenever the collection changes. Do not edit it by hand; fix
a reference in Zotero. To cite a new work, add it to the collection and use its
citation key in `\cite{}`. The export omits `abstract`, `file`, `urldate` and
`copyright` (Better BibTeX settings, Export → Fields), so no local file paths
enter the repository. A work in the collection that the text does not cite is
harmless: BibTeX prints only cited works. The style is `unsrtnat` (numbered, in
order of citation) until a journal is chosen.

## Direct LaTeX commands

From this directory:

```bash
latexmk -pdf main.tex
latexmk -pvc -pdf main.tex   # rebuild as the source changes
make clean                   # remove main.pdf and the build files
```
