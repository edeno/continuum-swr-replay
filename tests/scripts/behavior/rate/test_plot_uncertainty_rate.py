"""The uncertainty-rate figure's command line."""

from __future__ import annotations

from scripts.behavior.rate import plot_uncertainty_rate as mod


def test_the_parser_reads_the_smoke_fit_and_the_provenance_opt_out():
    parser = mod.build_parser()
    default = parser.parse_args([])
    assert not default.smoke and not default.allow_stale
    both = parser.parse_args(["--smoke", "--allow-stale"])
    assert both.smoke and both.allow_stale
