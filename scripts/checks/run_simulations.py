#!/usr/bin/env python
"""Run a model's scenario list through the simulation engine.

Every number this script writes is a simulation number.

``--model`` names an entry of :data:`MODELS`: a module beside the model that
holds its scenario list. The module has ``simulation(quick: bool, workers:
int) -> regression.simulation.Simulation`` (the models as the model's code
builds them, their rows, the scenarios, the data sets, the bootstrap draws
and the seed; ``workers`` the processes it may use to build them) and may
have ``table(result, simulation) -> pd.DataFrame``, which
gives its own columns from the engine's tables; without it the table is the
engine's summary (``regression.simulation.summarize``). The engine draws
only the outcome and runs the real fit, estimands and reduction.

Written: ``<model>.parquet`` in ``--output-dir`` (by default
``Processed-Data/replay_content/models_convention/simulations/``), with, in
its ``attrs``: ``model``, ``seed``, ``n_data_sets``, ``n_draws``,
``scenarios`` (those run), ``commit`` (``runs.provenance.described_commit``),
``log_penalties`` (each model's ``Design.log_penalty_record``: the declared
penalties), the simulation's own ``attrs`` and ``seconds``. The table is
printed, with a warning line for each scenario, model and estimand in which
a session fit did not converge or a session has no value.
``--reference PATH`` writes the table and its ``attrs`` as JSON as well, the
form of a fixed-seed test fixture.

::

    python scripts/checks/run_simulations.py --model declared_penalty --quick
    python scripts/checks/run_simulations.py --model declared_penalty \\
        --workers 6 --output-dir OUT
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from continuum_swr_replay.runs.threads import pin_blas_threads  # noqa: E402

pin_blas_threads()

import pandas as pd  # noqa: E402

from continuum_swr_replay.analysis.regression import simulation as engine  # noqa: E402
from continuum_swr_replay.paths import replay_content_dir  # noqa: E402
from continuum_swr_replay.runs.device import default_jax_platforms  # noqa: E402
from continuum_swr_replay.runs.provenance import described_commit  # noqa: E402
from continuum_swr_replay.tables.io import write_table  # noqa: E402

#: Model name -> the module of its scenario list.
MODELS = {
    "declared_penalty": "scripts.checks.simulate_declared_penalty",
}


def output_dir() -> Path:
    """The default folder of the tables, read when called."""
    return replay_content_dir() / "models_convention" / "simulations"


def run(
    name: str,
    *,
    quick: bool = False,
    scenarios: list[str] | None = None,
    n_data_sets: int | None = None,
    seed: int | None = None,
    workers: int = 1,
) -> pd.DataFrame:
    """The table of model ``name`` with its ``attrs`` (module docstring)."""
    started = time.perf_counter()
    module = importlib.import_module(MODELS[name])
    simulation = module.simulation(quick, workers)
    if n_data_sets is not None:
        simulation = replace(simulation, n_data_sets=n_data_sets)
    if seed is not None:
        simulation = replace(simulation, seed=seed)
    result = engine.run(simulation, workers, scenarios)
    tabulate = getattr(module, "table", None)
    table = result.summary if tabulate is None else tabulate(result, simulation)
    run_names = [
        s.name for s in simulation.scenarios if scenarios is None or s.name in scenarios
    ]
    table.attrs = {
        "model": name,
        "seed": simulation.seed,
        "n_data_sets": simulation.n_data_sets,
        "n_draws": simulation.n_draws,
        "scenarios": run_names,
        "commit": described_commit(),
        "log_penalties": {
            model.name: model.design.log_penalty_record() for model in simulation.models
        },
        **simulation.attrs,
        "seconds": time.perf_counter() - started,
    }
    table.attrs["warnings"] = warnings(result.summary)
    return table


def warnings(summary: pd.DataFrame) -> list[str]:
    """One line for each row of the engine's summary in which a session fit
    did not converge or a session has no value."""
    out = []
    for row in summary.itertuples():
        where = f"scenario {row.scenario}, model {row.model}, estimand {row.estimand}"
        if row.share_not_converged > 0:
            out.append(
                f"warning: {where}: {100 * row.share_not_converged:.1f} % of the "
                "session fits tried did not converge"
            )
        if row.share_without_value > 0:
            causes = ", ".join(
                f"{column.removeprefix('without_value_')} {100 * share:.1f} %"
                for column, share in row._asdict().items()
                if column.startswith("without_value_") and share > 0
            )
            out.append(
                f"warning: {where}: {100 * row.share_without_value:.1f} % of the "
                f"sessions have no value ({causes}); an estimate used as few as "
                f"{row.n_sessions_used_min} of the {row.n_sessions} sessions"
            )
    return out


def main(argv: list[str] | None = None) -> None:
    default_jax_platforms("cpu")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--scenarios", type=lambda text: text.split(","), default=None)
    parser.add_argument("--n-data-sets", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reference", type=Path, default=None)
    args = parser.parse_args(argv)
    table = run(
        args.model,
        quick=args.quick,
        scenarios=args.scenarios,
        n_data_sets=args.n_data_sets,
        seed=args.seed,
        workers=args.workers,
    )
    path = write_table(table, (args.output_dir or output_dir()) / args.model)
    with pd.option_context("display.width", 200, "display.max_columns", 60):
        print(table.round(4).to_string(index=False))
    for line in table.attrs["warnings"]:
        print(line)
    if args.reference is not None:
        args.reference.parent.mkdir(parents=True, exist_ok=True)
        command = "python scripts/checks/run_simulations.py " + " ".join(
            sys.argv[1:] if argv is None else argv
        )
        args.reference.write_text(
            json.dumps(
                {
                    "command": command,
                    "attrs": table.attrs,
                    "table": json.loads(table.to_json(orient="records")),
                },
                indent=1,
                default=str,
            )
            + "\n"
        )
    print(f"wrote {path} in {table.attrs['seconds']:.0f} s")


if __name__ == "__main__":
    main()
