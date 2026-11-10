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

## Files

| File | Purpose |
| --- | --- |
| `main.tex` | The manuscript and its preamble |
| `references.bib` | The bibliography (BibTeX) |
| `reported_values.tex` | Generated: one LaTeX macro per reported number, read by `main.tex` |
| `reported/part{n}.json` | The four summary files every reported number is read from |
| `reported/values_index.md`, `reported/overview.md` | Every reported value with its meaning; the values read question by question |
| `figures/<topic>/` | Figures, one folder per figure, with their caption drafts |
| `claims_table.md`, `figure_claims_plan.md` | Each claim, the comparison that supports it, and where it goes |
| `Makefile`, `.latexmkrc` | Build settings |
| `LICENSE` | CC BY 4.0 for the manuscript text and figures |

## Numbers

Do not type a result number into `main.tex`. Use the macro that
`reported_values.tex` defines for it, for example
`\PartOneSweepReplays{}`; [reported/values_index.md](reported/values_index.md)
lists every key. The macro for summary key `part1.sweep_replays` is
`\PartOne` followed by the key in CamelCase.

When a summary changes, regenerate the macros, then rebuild:

```bash
make summaries        # rebuild reported/part{n}.json (needs the lab's processed data)
make reported-values  # write the macros and the docs' inline values
make manuscript
```

`make manuscript` rebuilds when `reported_values.tex` changes, but does not
regenerate it. `tests/unit/test_manuscript_macros.py` checks that every macro
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

## Bibliography

Until the text cites a work, BibTeX reports that it found no citations; the
build still succeeds. The style is `unsrtnat` (numbered, in order of citation)
until a journal is chosen.

## Direct LaTeX commands

From this directory:

```bash
latexmk -pdf main.tex
latexmk -pvc -pdf main.tex   # rebuild as the source changes
make clean                   # remove main.pdf and the build files
```
