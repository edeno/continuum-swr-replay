# Non-local hippocampal events outside sharp-wave ripples (working title)

Source code, analyses, and manuscript for a paper on non-local hippocampal
events: periods when the hippocampus represents a place other than the animal's
own, found with a switching state-space decoder. The paper characterizes these
events in rats performing a spatial bandit task: how they compare with
sharp-wave ripples (SWRs) and with other replay detectors, how they relate to
the animal's choices, and how prefrontal neurons respond to them, comparing
events inside and outside SWRs throughout.

- **Read:** [manuscript PDF](paper/main.pdf) (in progress)
- **Reproduce:** [how to rerun the analyses](docs/reproduce.md)
- **Find a figure's code:** [figure pipeline](docs/figure-pipeline.md)

## Build the manuscript

The reported-value macros and the finished figures (so far Figure 1) are
committed; a figure not yet committed builds as a placeholder box. With LaTeX and
`latexmk` installed, run from the repository root:

```bash
make manuscript
```

This builds `paper/main.pdf` without Python or the recording data.

## Run the analyses

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
make sync-dev   # install the locked environment (uv.lock) with the dev tools
make check      # format check, lint, and the test suite, as CI runs them
make help       # list every task
```

The analyses read the Frank lab's Spyglass database and its processed data, so
they currently run only on lab machines. [Reproduction](docs/reproduce.md) lists
what each step needs, and gives the full rebuild from the recordings to the PDF.

## Find your way around

| Location | Contents |
| --- | --- |
| `paper/` | LaTeX source, compiled paper, figures, and the summary files every reported number is read from |
| `src/continuum_swr_replay/` | The analysis package: data loading, decoding, event tables, models, plotting |
| `scripts/` | Command-line entry points, grouped by domain ([index](scripts/README.md)) |
| `docs/` | Methods, design specs, and guides ([index](docs/README.md)) |
| `tests/` | The test suite ([guide](tests/README.md)) |

| Guide | Purpose |
| --- | --- |
| [Reproduction](docs/reproduce.md) | Requirements, the full rebuild in order, and what is committed |
| [Figure pipeline](docs/figure-pipeline.md) | Each figure's claims, script, inputs, and tests |
| [Data lineage](docs/data-lineage.md) | Where each stage reads and writes, from the NWB files to the summaries |
| [Analysis map](docs/analysis_map.md) | The paper's four parts and every analysis under each |
| [Development](docs/development.md) | Setup, code layout, and the project's analysis conventions |
| [Manuscript](paper/README.md) | Building the PDF, and writing numbers and figures into it |

## License and citation

The code is under the [MIT License](LICENSE). The manuscript text and figures in
`paper/` are under [CC BY 4.0](paper/LICENSE). To cite this work, see
[CITATION.cff](CITATION.cff).
