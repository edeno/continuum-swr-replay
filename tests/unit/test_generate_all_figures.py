"""``generate_all_figures`` runs each figure script with this interpreter and,
when one fails, logs that script's whole stderr."""

from __future__ import annotations

import logging
import sys

from scripts.figures import generate_all_figures


def _config(script, tmp_path) -> dict:
    return {"script": str(script), "output_dir": str(tmp_path / "out")}


def test_a_figure_script_runs_with_this_interpreter(tmp_path):
    seen = tmp_path / "executable.txt"
    script = tmp_path / "figure.py"
    script.write_text(f"import sys\nopen({str(seen)!r}, 'w').write(sys.executable)\n")

    succeeded, _ = generate_all_figures.run_figure_script(
        "figure", _config(script, tmp_path), None
    )

    assert succeeded
    assert seen.read_text() == sys.executable


def test_a_failing_figure_script_logs_its_whole_stderr(tmp_path, caplog):
    lines = "\n".join(f"stderr line {i:03d}" for i in range(100))  # ~1.7 kB
    script = tmp_path / "figure.py"
    script.write_text(f"import sys\nsys.stderr.write({lines!r})\nsys.exit(3)\n")

    with caplog.at_level(logging.ERROR):
        succeeded, _ = generate_all_figures.run_figure_script(
            "figure", _config(script, tmp_path), None
        )

    assert not succeeded
    assert lines in caplog.text
